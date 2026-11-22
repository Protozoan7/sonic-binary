import os
import time
import queue
import hashlib
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt, chirp, fftconvolve
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

rate = 48000
bit_time = 0.03
password = input("Password: ")

# (tone for 0, tone for 1). bits take turns using these 4 pairs so echoes dont mix up the bits
tones = [(17600, 17900), (18200, 18500), (18800, 19100), (19400, 19700)]

filt = butter(6, [17000, 20500], btype="bandpass", fs=rate, output="sos")
bit = int(rate * bit_time)


def key():
    return hashlib.sha256(password.encode()).digest()[:16]


def lock(msg):
    nonce = os.urandom(12)
    return nonce + AESGCM(key()).encrypt(nonce, msg.encode(), None)


def unlock(data):
    return AESGCM(key()).decrypt(data[:12], data[12:], None).decode()


def unlock_part(nonce, part):
    c = Cipher(algorithms.AES(key()), modes.CTR(nonce + b"\x00\x00\x00\x02")).decryptor()
    return c.update(part).decode(errors="replace")


def tone(f, length):
    t = np.arange(int(rate * length)) / rate
    x = np.sin(2 * np.pi * f * t)
    n = int(rate * 0.002)
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def start_sound():
    t = np.arange(int(rate * 0.1)) / rate
    x = chirp(t, 17300, 0.1, 20200)
    n = int(rate * 0.005)
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def send():
    msg = input("Enter message: ")[:200]
    data = lock(msg)
    data = bytes([len(data)]) + data

    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    sound = [np.zeros(int(rate * 0.3)), start_sound(), np.zeros(int(rate * 0.02))]
    for i in range(len(bits)):
        sound.append(tone(tones[i % 4][bits[i]], bit_time))
    sound.append(np.zeros(int(rate * 0.3)))
    sound = 0.6 * np.concatenate(sound)

    print("Sending", len(bits), "bits, takes", round(len(sound) / rate, 1), "seconds")
    sd.play(sound, rate)
    sd.wait()
    print("Sent!")


def strength(x, f):
    x = x * np.hanning(len(x))
    t = np.arange(len(x)) / rate
    return abs(np.sum(x * np.exp(-2j * np.pi * f * t))) ** 2


def find_start(audio):
    ref = start_sound()
    x = sosfiltfilt(filt, audio)
    match = np.abs(fftconvolve(x, ref[::-1], mode="valid"))
    power = np.maximum(fftconvolve(x ** 2, np.ones(len(ref)), mode="valid"), 0)
    # 1 = sounds exactly like the start sound, 0 = nothing like it
    match = match / np.sqrt(np.sum(ref ** 2) * power + 1e-12)
    match = match[:len(match) - int(rate * 0.05)]
    if len(match) == 0 or match.max() < 0.4:
        return None
    return int(np.argmax(match >= 0.8 * match.max()))


def read_byte(x, first, n):
    byte = 0
    loudest = []
    for i in range(n * 8, n * 8 + 8):
        # skip the first 30% of the bit, thats where the echo of the last bit is
        part = x[first + i * bit + int(bit * 0.3): first + (i + 1) * bit - int(rate * 0.002)]
        f0, f1 = tones[i % 4]
        e0 = strength(part, f0)
        e1 = strength(part, f1)
        byte = byte * 2 + (1 if e1 > e0 else 0)
        loudest.append(max(e0, e1))
    return byte, np.median(loudest)


def receive():
    print("Listening... (Ctrl+C to stop)")
    audio = np.zeros(0)
    start = None
    got = b""
    shown = ""
    q = queue.Queue()

    # the mic puts sound in the queue in the background, so nothing is lost while we calculate
    def mic_callback(data, frames, t, status):
        q.put(data[:, 0].copy())

    try:
        with sd.InputStream(samplerate=rate, channels=1, callback=mic_callback):
            while True:
                time.sleep(0.25)
                while not q.empty():
                    audio = np.concatenate([audio, q.get()])

                if start is None:
                    audio = audio[-rate * 2:]
                    start = find_start(audio)
                    continue

                first = start + int(rate * 0.12)
                ready = (len(audio) - rate // 10 - first) // (8 * bit)
                if ready < 1:
                    continue
                x = sosfiltfilt(filt, audio)

                if got == b"":
                    size, loud = read_byte(x, first, 0)
                    quiet = np.median([strength(x[j:j + bit], 18500) for j in range(max(0, start - rate // 4), start - bit, bit)] or [1])
                    if size < 29 or loud < 10 * quiet:
                        audio = audio[start + int(rate * 0.1):]
                        start = None
                        continue
                    got = bytes([size])
                    print("Start signal heard")
                    print("Message: ", end="", flush=True)

                size = got[0]
                while len(got) < min(ready, size + 1):
                    got += bytes([read_byte(x, first, len(got))[0]])

                if len(got) > 13:
                    text = unlock_part(got[1:13], got[13:min(len(got), size - 15)])
                    print(text[len(shown):], end="", flush=True)
                    shown = text

                if len(got) == size + 1:
                    try:
                        print("\n\nMESSAGE:", unlock(got[1:]), "(checked OK)\n")
                    except Exception:
                        print("\n\nCould not unlock it (wrong password or damaged)\n")
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
