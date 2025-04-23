# pages/advanced_noise_reduction.py

from flask import Blueprint, render_template, request, current_app
import os, uuid, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from io import BytesIO
import pywt
from scipy.signal import wiener, butter, filtfilt

advanced_noise_reduction_bp = Blueprint("advanced_noise_reduction", __name__)
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def generate_time_domain_plot(audio, denoised, duration):
    t = np.linspace(0, duration, len(audio))
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0].plot(t, audio)
    ax[0].set_title("Original Audio (Time Domain)")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[1].plot(t, denoised, color="orange")
    ax[1].set_title("Denoised Audio (Time Domain)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_frequency_domain_plot(audio, denoised, sample_rate):
    fft_orig = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1/sample_rate)
    mag_orig = 20 * np.log10(np.abs(fft_orig) + 1e-6)
    fft_den = np.fft.rfft(denoised)
    mag_den = 20 * np.log10(np.abs(fft_den) + 1e-6)
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0].plot(freqs, mag_orig)
    ax[0].set_title("Original Audio Spectrum")
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("Magnitude (dB)")
    ax[1].plot(freqs, mag_den, color="orange")
    ax[1].set_title("Denoised Audio Spectrum")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude (dB)")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_gain_plot(audio, denoised, sample_rate):
    fft_orig = np.fft.rfft(audio)
    fft_den = np.fft.rfft(denoised)
    gain = np.abs(fft_den) / (np.abs(fft_orig) + 1e-6)
    gain_db = 20 * np.log10(gain + 1e-6)
    freqs = np.fft.rfftfreq(len(audio), d=1/sample_rate)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(freqs, gain_db)
    ax.set_title("Effective Filter Gain (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_wavelet_coef_plot(original_coeffs, thresholded_coeffs):
    detail_orig = original_coeffs[-1]
    detail_thr  = thresholded_coeffs[-1]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(detail_orig, label="Original Coeffs")
    ax.plot(detail_thr,  label="Thresholded Coeffs", color="orange")
    ax.set_title(f"Wavelet Detail Coeffs (Level {len(original_coeffs)-1})")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def process_noise_reduction(method, sample_rate, audio):
    if method == "wiener":
        denoised = wiener(audio)
        coef_plot = None
    elif method == "spectral":
        fft_audio = np.fft.rfft(audio)
        noise_floor = 0.05 * np.max(np.abs(fft_audio))
        fft_den = np.where(np.abs(fft_audio) < noise_floor, 0, fft_audio)
        denoised = np.fft.irfft(fft_den)
        coef_plot = None
    elif method == "wavelet":
        coeffs = pywt.wavedec(audio, 'db8', level=6)
        original = coeffs[:]
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(audio)))
        thr_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
        denoised = pywt.waverec(thr_coeffs, 'db8')
        coef_plot = generate_wavelet_coef_plot(original, thr_coeffs)
    else:
        denoised = audio
        coef_plot = None

    gain_plot = generate_gain_plot(audio, denoised, sample_rate)
    return denoised, method, coef_plot, gain_plot

@advanced_noise_reduction_bp.route("/", methods=["GET", "POST"])
def advanced_noise_reduction():
    time_plot = freq_plot = gain_plot = coef_plot = output_audio = None
    error = None

    if request.method == "POST":
        choice = request.form.get("audio_choice", "default")
        try:
            # 1) Load WAV
            if choice == "default":
                path = os.path.join(current_app.root_path, "static", "audio", "example.wav")
                sr, audio = wavfile.read(path)
            else:
                if "audio_file" not in request.files:
                    raise ValueError("Please upload an audio file.")
                f = request.files["audio_file"]
                fn = f"{uuid.uuid4()}.wav"
                fp = os.path.join(UPLOAD_FOLDER, fn)
                f.save(fp)
                sr, audio = wavfile.read(fp)

            # 2) Preprocess
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = audio.astype(np.float32)
            duration = len(audio) / sr
            if duration > 10:
                raise ValueError("Audio longer than 10 seconds.")

            # 3) Denoise
            method = request.form.get("method", "wiener")
            denoised, meth, comp_coef, gain_plot = process_noise_reduction(method, sr, audio)

            # 4) Plots
            time_plot = generate_time_domain_plot(audio, denoised, duration)
            freq_plot = generate_frequency_domain_plot(audio, denoised, sr)
            coef_plot = comp_coef

            # 5) Output WAV
            buf = BytesIO()
            dn = denoised / np.max(np.abs(denoised))
            wavfile.write(buf, sr, np.int16(dn * 32767))
            buf.seek(0)
            output_audio = base64.b64encode(buf.read()).decode()

        except Exception as e:
            error = str(e)

    return render_template("advanced_noise_reduction.html",
                           time_plot=time_plot,
                           freq_plot=freq_plot,
                           gain_plot=gain_plot,
                           coef_plot=coef_plot,
                           output_audio=output_audio,
                           error=error)
