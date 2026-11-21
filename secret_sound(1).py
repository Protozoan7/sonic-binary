import os
import hashlib
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt, chirp, fftconvolve
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

rate = 48000
bit_time = 0.03
password = "snu-physics-2026"

# 4 pairs of tones: (tone for 0, tone for 1). bits take turns using them
tones = [(17600, 17900), (18200, 18500), (18800, 19100), (19400, 19700)]

filt = butter(6, [17000, 20500], btype="bandpass", fs=rate, output="sos")


def make_key(pw):
    return hashlib.sha256(pw.encode()).digest()[:16]


def encrypt(msg, pw):
    nonce = os.urandom(12)
    return nonce + AESGCM(make_key(pw)).encrypt(nonce, msg.encode(), None)


def decrypt(data, pw):
    return AESGCM(make_key(pw)).decrypt(data[:12], data[12:], None).decode()


# AES-GCM scrambles each byte on its own, so letters can be unscrambled as they arrive.
# The full message is still checked with decrypt() at the end.
def decrypt_part(nonce, part, pw):
    c = Cipher(algorithms.AES(make_key(pw)), modes.CTR(nonce + b"\x00\x00\x00\x02")).decryptor()
    return c.update(part).decode(errors="replace")


def smooth(x, ms):
    n = int(rate * ms / 1000)
    ramp = np.linspace(0, 1, n)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def tone(f, length):
    t = np.arange(int(rate * length)) / rate
    return smooth(np.sin(2 * np.pi * f * t), 2)


def start_sound():
    t = np.arange(int(rate * 0.1)) / rate
    return smooth(chirp(t, 17300, 0.1, 20200), 5)


def to_bits(data):
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def to_bytes(bits):
    data = []
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = byte * 2 + b
        data.append(byte)
    return bytes(data)


# ---------------- SEND ----------------

def send():
    msg = input("Enter message: ")[:200]
    data = encrypt(msg, password)
    bits = to_bits([len(data)] + list(data))

    sound = [np.zeros(int(rate * 0.3)), start_sound(), np.zeros(int(rate * 0.02))]
    for pair in tones:
        for f in pair:
            sound.append(tone(f, 2 * bit_time))
    for i in range(len(bits)):
        f = tones[i % 4][bits[i]]
        sound.append(tone(f, bit_time))
    sound.append(np.zeros(int(rate * 0.3)))
    sound = 0.6 * np.concatenate(sound)

    print("Encrypted:", data.hex(" ")[:47], "...")
    print("Sending", len(bits), "bits, takes", round(len(sound) / rate, 1), "seconds")
    sd.play(sound, rate)
    sd.wait()
    print("Sent!")


# ---------------- RECEIVE ----------------

def energy(x, f):
    w = x * np.hanning(len(x))
    t = np.arange(len(x)) / rate
    return abs(np.sum(w * np.exp(-2j * np.pi * f * t))) ** 2


def find_start(audio):
    ref = start_sound()
    x = sosfiltfilt(filt, audio)
    match = np.abs(fftconvolve(x, ref[::-1], mode="valid"))
    power = fftconvolve(x ** 2, np.ones(len(ref)), mode="valid")
    # how much the sound looks like the start sound (0 to 1), not just how loud it is
    match = match / np.sqrt(np.sum(ref ** 2) * np.maximum(power, 0) + 1e-12)
    match = match[:len(match) - int(rate * 0.05)]
    if len(match) == 0 or match.max() < 0.4:
        return None
    return int(np.argmax(match >= 0.8 * match.max()))


def read_slot(x, pos, length):
    bit = int(rate * bit_time)
    end = pos + length - int(rate * 0.002)
    return x[end - int(bit * 0.7) + int(rate * 0.002): end]


def learn_room(x, start):
    noise = []
    n = int(rate * bit_time * 0.7)
    before = x[max(0, start - int(rate * 0.3)): start - int(rate * 0.02)]
    for i in range(0, len(before) - n, n):
        noise.append(energy(before[i:i + n], 18500))
    noise = np.median(noise) if noise else 1e-12

    gain = {}
    pos = start + int(rate * 0.12)
    for pair in tones:
        for f in pair:
            gain[f] = energy(read_slot(x, pos, int(rate * 2 * bit_time)), f)
            pos += int(rate * 2 * bit_time)
    loud = np.median(list(gain.values()))
    if loud < 10 * noise:
        return None, pos
    for f in gain:
        gain[f] = max(gain[f], 0.01 * loud)
    return gain, pos


def read_bits(x, pos, gain, first, count):
    bits = []
    bit = int(rate * bit_time)
    for i in range(first, first + count):
        p0, p1 = tones[i % 4]
        slot = read_slot(x, pos + i * bit, bit)
        e0 = energy(slot, p0) / gain[p0]
        e1 = energy(slot, p1) / gain[p1]
        bits.append(1 if e1 > e0 else 0)
    return bits


def receive():
    print("Listening... (Ctrl+C to stop)")
    audio = np.zeros(0)
    start = None
    size = 0
    got = b""
    shown = ""
    bit = int(rate * bit_time)
    try:
        with sd.InputStream(samplerate=rate, channels=1) as mic:
            while True:
                chunk, _ = mic.read(int(rate * 0.25))
                audio = np.concatenate([audio, chunk[:, 0]])

                if start is None:
                    audio = audio[-rate * 2:]
                    start = find_start(audio)
                    if start is not None:
                        print("Start signal heard, receiving...")
                    continue

                need = start + int(rate * 0.62) + bit * 8 + rate // 10
                if len(audio) < need:
                    continue
                x = sosfiltfilt(filt, audio)

                if size == 0:
                    gain, pos = learn_room(x, start)
                    size = to_bytes(read_bits(x, pos, gain, 0, 8))[0] if gain else 0
                    if size < 29:
                        print("False alarm, still listening...")
                        audio = audio[start + int(rate * 0.1):]
                        start = None
                        size = 0
                        continue
                    print("Message: ", end="", flush=True)

                have = min(size, (len(audio) - rate // 10 - pos) // (8 * bit) - 1)
                if have > len(got):
                    bits = read_bits(x, pos, gain, 8 * (1 + len(got)), 8 * (have - len(got)))
                    got += to_bytes(bits)

                if len(got) > 12:
                    text = decrypt_part(got[:12], got[12:min(len(got), size - 16)], password)
                    print(text[len(shown):], end="", flush=True)
                    shown = text

                if len(got) == size:
                    try:
                        msg = decrypt(got, password)
                        print("\n\nMESSAGE:", msg, "(checked OK)\n")
                    except Exception:
                        print("\n\nCould not decrypt it properly (wrong password or damaged)\n")
                    return
    except KeyboardInterrupt:
        print("Stopped.")


while True:
    print("\n1. Send")
    print("2. Receive")
    print("3. Exit")
    choice = input("Choose: ")
    if choice == "1":
        send()
    elif choice == "2":
        receive()
    elif choice == "3":
        break
