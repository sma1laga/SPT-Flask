# pages/speech_filter_input.py

from flask import Blueprint, render_template, request, session, current_app, jsonify
import os, uuid, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from io import BytesIO
from scipy.signal import butter, filtfilt

speech_filter_input_bp = Blueprint("speech_filter_input", __name__)
TEMP_UPLOAD_FOLDER = "temp_uploads"
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)

def plot_pole_zero(b, a):
    fig, ax = plt.subplots(figsize=(5,5))
    zeros = np.roots(b)
    poles = np.roots(a)
    ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Zeros')
    ax.scatter(np.real(poles), np.imag(poles), marker='x', color='red',  label='Poles')
    theta = np.linspace(0,2*np.pi,300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)
    ax.set_title("Pole–Zero Plot")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal','box')
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def process_audio(filter_type, order, cutoff_str, sample_rate, audio, duration):
    # design
    cuts = [float(x) for x in cutoff_str.split(",")]
    nyq = sample_rate / 2.0
    if filter_type in ["bandpass","bandstop"]:
        norm = sorted([c/nyq for c in cuts])
    else:
        norm = cuts[0]/nyq
    b, a = butter(order, norm, btype=filter_type)
    # filter
    filtered = filtfilt(b, a, audio)
    # spectrogram
    fig1, ax1 = plt.subplots(2,1,figsize=(6,4))
    ax1[0].specgram(audio, NFFT=256, Fs=sample_rate, noverlap=128, cmap='viridis')
    ax1[0].set_title("Original Spectrogram")
    ax1[1].specgram(filtered, NFFT=256, Fs=sample_rate, noverlap=128, cmap='viridis')
    ax1[1].set_title("Filtered Spectrogram")
    buf1 = BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format='png', dpi=75)
    buf1.seek(0)
    spec_plot = base64.b64encode(buf1.getvalue()).decode("utf-8")
    plt.close(fig1)
    # waveform
    t = np.linspace(0, duration, len(audio))
    fig2, ax2 = plt.subplots(2,1,figsize=(6,4))
    ax2[0].plot(t, audio);       ax2[0].set_title("Original Waveform")
    ax2[1].plot(t, filtered, color='orange'); ax2[1].set_title("Filtered Waveform")
    buf2 = BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png', dpi=75)
    buf2.seek(0)
    wave_plot = base64.b64encode(buf2.getvalue()).decode("utf-8")
    plt.close(fig2)
    # pole-zero
    pz_plot = plot_pole_zero(b, a)
    return filtered, spec_plot, wave_plot, pz_plot

@speech_filter_input_bp.route("/", methods=["GET", "POST"])
def speech_filter_input():
    spec_plot = wave_plot = pz_plot = output_audio_data = None
    error = None

    if request.method == "POST":
        try:
            # --- load audio ---
            choice = request.form.get("audio_choice", "default")
            if choice == "default":
                path = os.path.join(current_app.root_path, "static", "audio", "example.wav")
                sr, audio = wavfile.read(path)
            else:
                if "audio_file" not in request.files:
                    raise ValueError("No audio file uploaded.")
                f = request.files["audio_file"]
                fn = f"{uuid.uuid4()}.wav"
                fp = os.path.join(TEMP_UPLOAD_FOLDER, fn)
                f.save(fp)
                session["uploaded_audio_file"] = fn
                sr, audio = wavfile.read(fp)

            # --- preprocess ---
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = audio.astype(np.float32)
            duration = len(audio) / sr
            if duration > 10:
                raise ValueError("Audio longer than 10 seconds.")

            # --- filter params ---
            if request.form.get("use_standard_filter"):
                preset = request.form.get("standard_filter", "lowpass_std")
                if preset == "lowpass_std":
                    filter_type, order, cutoff_str = "lowpass", 3, "1000"
                elif preset == "bandpass_std":
                    filter_type, order, cutoff_str = "bandpass", 4, "300,3400"
                elif preset == "highpass_std":
                    filter_type, order, cutoff_str = "highpass", 2, "80"
                elif preset == "telephone_filter":
                    filter_type, order, cutoff_str = "bandpass", 4, "300,3400"
                elif preset == "podcast_filter":
                    filter_type, order, cutoff_str = "lowpass", 3, "3000"
                else:
                    filter_type, order, cutoff_str = "bandstop", 2, "50,150"
            else:
                filter_type = request.form.get("filter_type", "lowpass")
                order       = int(request.form.get("order", 4))
                cutoff_str  = request.form.get("cutoff", "100,400")

            # --- process ---
            filtered, spec_plot, wave_plot, pz_plot = process_audio(
                filter_type, order, cutoff_str, sr, audio, duration
            )

            # --- prepare audio for playback/download ---
            buf = BytesIO()
            wavfile.write(buf, sr,
                          np.int16(filtered / np.max(np.abs(filtered)) * 32767))
            buf.seek(0)
            output_audio_data = base64.b64encode(buf.read()).decode("utf-8")

        except Exception as e:
            error = str(e)

    return render_template("speech_filter_input.html",
                           spec_plot=spec_plot,
                           wave_plot=wave_plot,
                           pz_plot=pz_plot,
                           output_audio_data=output_audio_data,
                           error=error)

@speech_filter_input_bp.route("/live", methods=["GET"])
def live_audio():
    """
    AJAX endpoint: reapply filter to the same audio (uploaded or example)
    using query params and session to fetch the file.
    """
    try:
        # load from session or fallback to example
        if "uploaded_audio_file" in session:
            fn = session["uploaded_audio_file"]
            path = os.path.join(TEMP_UPLOAD_FOLDER, fn)
            sr, audio = wavfile.read(path)
        else:
            path = os.path.join(current_app.root_path, "static", "audio", "example.wav")
            sr, audio = wavfile.read(path)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        duration = len(audio) / sr

        # get filter params from querystring
        if request.args.get("use_standard_filter") == "1":
            preset = request.args.get("standard_filter", "lowpass_std")
            if preset == "lowpass_std":
                filter_type, order, cutoff_str = "lowpass", 3, "1000"
            elif preset == "bandpass_std":
                filter_type, order, cutoff_str = "bandpass", 4, "300,3400"
            elif preset == "highpass_std":
                filter_type, order, cutoff_str = "highpass", 2, "80"
            elif preset == "telephone_filter":
                filter_type, order, cutoff_str = "bandpass", 4, "300,3400"
            elif preset == "podcast_filter":
                filter_type, order, cutoff_str = "lowpass", 3, "3000"
            else:
                filter_type, order, cutoff_str = "bandstop", 2, "50,150"
        else:
            filter_type = request.args.get("filter_type", "lowpass")
            order       = int(request.args.get("order", 4))
            cutoff_str  = request.args.get("cutoff", "100,400")

        # re-process
        filtered, spec_plot, wave_plot, pz_plot = process_audio(
            filter_type, order, cutoff_str, sr, audio, duration
        )

        # prepare audio
        buf = BytesIO()
        wavfile.write(buf, sr,
                      np.int16(filtered / np.max(np.abs(filtered)) * 32767))
        buf.seek(0)
        audio_data = base64.b64encode(buf.read()).decode("utf-8")

        return jsonify({
            "spec_plot": spec_plot,
            "wave_plot": wave_plot,
            "pz_plot":  pz_plot,
            "audio":    audio_data
        })

    except Exception as e:
        return jsonify(error=str(e)), 400
