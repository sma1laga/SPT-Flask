from flask import Blueprint, render_template, request
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from io import BytesIO
import base64
import pywt
from scipy.signal import wiener, butter, filtfilt

advanced_noise_reduction_bp = Blueprint("advanced_noise_reduction", __name__)
UPLOAD_FOLDER = "temp_uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
    time_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return time_plot

def generate_frequency_domain_plot(audio, denoised, sample_rate):
    fft_orig = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1/sample_rate)
    mag_orig = 20 * np.log10(np.abs(fft_orig) + 1e-6)
    
    fft_denoised = np.fft.rfft(denoised)
    mag_denoised = 20 * np.log10(np.abs(fft_denoised) + 1e-6)
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0].plot(freqs, mag_orig)
    ax[0].set_title("Original Audio Spectrum")
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("Magnitude (dB)")
    ax[1].plot(freqs, mag_denoised, color="orange")
    ax[1].set_title("Denoised Audio Spectrum")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude (dB)")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    freq_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return freq_plot

def generate_gain_plot(audio, denoised, sample_rate):
    # Compute effective gain at each frequency.
    fft_orig = np.fft.rfft(audio)
    fft_denoised = np.fft.rfft(denoised)
    gain = np.abs(fft_denoised) / (np.abs(fft_orig) + 1e-6)
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
    gain_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return gain_plot

def generate_wavelet_coef_plot(original_coeffs, thresholded_coeffs):
    # Plot the detail coefficients from the last level.
    detail_orig = original_coeffs[-1]
    detail_thresh = thresholded_coeffs[-1]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(detail_orig, label="Original Coefficients")
    ax.plot(detail_thresh, label="Thresholded Coefficients", color="orange")
    ax.set_title("Wavelet Detail Coefficients (Level {})".format(len(original_coeffs)-1))
    ax.set_xlabel("Index")
    ax.set_ylabel("Coefficient Value")
    ax.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    coef_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return coef_plot

def process_noise_reduction(method, sample_rate, audio):
    if method == "wiener":
        denoised = wiener(audio)
        method_name = "Wiener Filtering"
        coef_plot = None  # Not applicable for Wiener; will use gain plot instead.
    elif method == "spectral":
        fft_audio = np.fft.rfft(audio)
        noise_floor = 0.05 * np.max(np.abs(fft_audio))
        fft_denoised = np.where(np.abs(fft_audio) < noise_floor, 0, fft_audio)
        denoised = np.fft.irfft(fft_denoised)
        method_name = "Spectral Subtraction"
        coef_plot = None
    elif method == "wavelet":
        coeffs = pywt.wavedec(audio, 'db8', level=6)
        original_coeffs = coeffs[:]  # Save original coefficients for reference.
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(audio)))
        thresholded_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
        denoised = pywt.waverec(thresholded_coeffs, 'db8')
        method_name = "Wavelet Denoising"
        coef_plot = generate_wavelet_coef_plot(original_coeffs, thresholded_coeffs)
    else:
        denoised = audio
        method_name = "No Filter"
        coef_plot = None
        
    gain_plot = generate_gain_plot(audio, denoised, sample_rate)
    return denoised, method_name, coef_plot, gain_plot

@advanced_noise_reduction_bp.route("/", methods=["GET", "POST"])
def advanced_noise_reduction():
    time_plot = None
    freq_plot = None
    gain_plot = None
    coef_plot = None
    output_audio = None
    error = None

    if request.method == "POST":
        try:
            if "audio_file" not in request.files:
                error = "Please upload an audio file (WAV, maximum 10 seconds)."
            else:
                audio_file = request.files["audio_file"]
                filename = str(uuid.uuid4()) + ".wav"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                audio_file.save(filepath)
                sample_rate, audio = wavfile.read(filepath)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                audio = audio.astype(np.float32)
                duration = len(audio) / sample_rate
                if duration > 10:
                    error = "The uploaded audio is longer than 10 seconds."
            
            if not error:
                method = request.form.get("method", "wiener")
                denoised, method_name, computed_coef_plot, gain_plot = process_noise_reduction(method, sample_rate, audio)
                time_plot = generate_time_domain_plot(audio, denoised, duration)
                freq_plot = generate_frequency_domain_plot(audio, denoised, sample_rate)
                # For wavelet method, computed_coef_plot is provided; otherwise it is None.
                coef_plot = computed_coef_plot  
                
                out_buffer = BytesIO()
                denoised_norm = denoised / np.max(np.abs(denoised))
                denoised_int16 = np.int16(denoised_norm * 32767)
                wavfile.write(out_buffer, sample_rate, denoised_int16)
                out_buffer.seek(0)
                output_audio = base64.b64encode(out_buffer.read()).decode("utf-8")
        except Exception as e:
            error = str(e)
            
    return render_template("advanced_noise_reduction.html",
                           time_plot=time_plot,
                           freq_plot=freq_plot,
                           gain_plot=gain_plot,
                           coef_plot=coef_plot,
                           output_audio=output_audio,
                           error=error)
