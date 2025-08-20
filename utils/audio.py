# utils/audio.py
import io, base64, wave
from scipy.io import wavfile
import numpy as np

def to_wav_bytes(signal: np.ndarray, fs: int) -> bytes:
    """signal in [-1,1], mono float -> 16-bit PCM WAV bytes."""
    sig = np.clip(signal, -1.0, 1.0)
    pcm = (sig * 32767.0).astype("<i2")  # little-endian int16
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()

def wav_data_url(signal: np.ndarray, fs: int) -> str:
    return "data:audio/wav;base64," + base64.b64encode(to_wav_bytes(signal, fs)).decode("ascii")

def read_mono_audio(path):
    fs, mono =  wavfile.read(path)
    if len(mono.shape) > 1:
        mono = mono[:,0]
    if 'int' in str(mono.dtype):
        mono = mono.astype("float32")
        # remove offset
        mono -= np.mean(mono)
        max_abs = np.max(np.abs(mono))
        mono /= max_abs
    return fs, mono
