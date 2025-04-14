from flask import Blueprint, render_template, request, session, jsonify
import os
import uuid
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.io import wavfile
from io import BytesIO
import base64

speech_filter_input_bp = Blueprint("speech_filter_input", __name__)

# Directory to temporarily save uploaded audio files
TEMP_UPLOAD_FOLDER = "temp_uploads"
if not os.path.exists(TEMP_UPLOAD_FOLDER):
    os.makedirs(TEMP_UPLOAD_FOLDER)

def plot_pole_zero(b, a):
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    fig, ax = plt.subplots(figsize=(5,5))
    zeros = np.roots(b)
    poles = np.roots(a)
    ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Zeros')
    ax.scatter(np.real(poles), np.imag(poles), marker='x', color='red', label='Poles')
    
    # Redo the unit circle: plot it with good resolution and a dashed style.
    theta = np.linspace(0, 2*np.pi, 300)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)
    ax.plot(unit_circle_x, unit_circle_y, 'k--', linewidth=1, label='Unit Circle')
    
    ax.set_title("Pole–Zero Plot")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.legend()
    ax.grid(True)
    
    # Enforce equal scaling for the unit circle.
    ax.set_aspect('equal', 'box')
    
    # Optionally, if you want a "normal" numeric scale rather than tight around the circle,
    # you can set custom limits. For example:
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    pz_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return pz_plot



def process_audio(filter_type, order, cutoff_str, sample_rate, audio, duration):
    """Design the filter, process the audio, and generate three images: spectrograms, waveforms, and pole–zero plot."""
    # Process cutoff string.
    cutoff_vals = [float(x.strip()) for x in cutoff_str.split(",")]
    nyquist = sample_rate / 2.0
    normalized_cutoffs = [f / nyquist for f in cutoff_vals]
    if filter_type in ["bandpass", "bandstop"]:
        if len(normalized_cutoffs) != 2:
            raise ValueError("For bandpass/stop filters, provide two cutoff values separated by a comma.")
        normalized_cutoffs = sorted(normalized_cutoffs)
    else:
        if len(normalized_cutoffs) != 1:
            raise ValueError("For low/high pass filters, provide a single cutoff value.")
        normalized_cutoffs = normalized_cutoffs[0]
    
    # Design and apply the Butterworth filter.
    b, a = butter(order, normalized_cutoffs, btype=filter_type)
    filtered_audio = filtfilt(b, a, audio)
    
    # Generate spectrogram plot.
    fig_spec, ax_spec = plt.subplots(2, 1, figsize=(6, 4))
    ax_spec[0].specgram(audio, NFFT=256, Fs=sample_rate, noverlap=128, cmap='viridis')
    ax_spec[0].set_title("Spectrogram of Original Audio")
    ax_spec[0].set_xlabel("Time (s)")
    ax_spec[0].set_ylabel("Frequency (Hz)")
    ax_spec[1].specgram(filtered_audio, NFFT=256, Fs=sample_rate, noverlap=128, cmap='viridis')
    ax_spec[1].set_title("Spectrogram of Filtered Audio")
    ax_spec[1].set_xlabel("Time (s)")
    ax_spec[1].set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    buf_spec = BytesIO()
    plt.savefig(buf_spec, format="png", dpi=75)
    buf_spec.seek(0)
    spec_plot = base64.b64encode(buf_spec.getvalue()).decode("utf-8")
    plt.close(fig_spec)
    
    # Generate waveform plot.
    fig_wave, ax_wave = plt.subplots(2, 1, figsize=(6, 4))
    t = np.linspace(0, duration, len(audio))
    ax_wave[0].plot(t, audio)
    ax_wave[0].set_title("Waveform of Original Audio")
    ax_wave[0].set_xlabel("Time (s)")
    ax_wave[0].set_ylabel("Amplitude")
    ax_wave[1].plot(t, filtered_audio, color="orange")
    ax_wave[1].set_title("Waveform of Filtered Audio")
    ax_wave[1].set_xlabel("Time (s)")
    ax_wave[1].set_ylabel("Amplitude")
    plt.tight_layout()
    buf_wave = BytesIO()
    plt.savefig(buf_wave, format="png", dpi=75)
    buf_wave.seek(0)
    wave_plot = base64.b64encode(buf_wave.getvalue()).decode("utf-8")
    plt.close(fig_wave)
    
    # Generate pole–zero plot.
    pz_plot = plot_pole_zero(b, a)
    
    return filtered_audio, spec_plot, wave_plot, pz_plot

@speech_filter_input_bp.route("/", methods=["GET", "POST"])
def speech_filter_input():
    spec_plot = None
    wave_plot = None
    pz_plot = None
    output_audio_data = None
    error = None
    
    if request.method == "POST":
        try:
            filter_type = request.form.get("filter_type", "lowpass")
            order = int(request.form.get("order", 4))
            cutoff_str = request.form.get("cutoff", "100,400")
            
            if "audio_file" not in request.files:
                error = "No audio file uploaded."
            else:
                audio_file = request.files["audio_file"]
                filename = str(uuid.uuid4()) + ".wav"
                filepath = os.path.join(TEMP_UPLOAD_FOLDER, filename)
                audio_file.save(filepath)
                session["uploaded_audio_file"] = filename
                
                sample_rate, audio = wavfile.read(filepath)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                audio = audio.astype(np.float32)
                duration = len(audio) / sample_rate
                if duration > 10:
                    error = "The uploaded audio is longer than 10 seconds."
            
            if not error:
                filtered_audio, spec_plot, wave_plot, pz_plot = process_audio(
                    filter_type, order, cutoff_str, sample_rate, audio, duration)
                
                out_buffer = BytesIO()
                filtered_int16 = np.int16(filtered_audio/np.max(np.abs(filtered_audio)) * 32767)
                wavfile.write(out_buffer, int(sample_rate), filtered_int16)
                out_buffer.seek(0)
                output_audio_data = base64.b64encode(out_buffer.read()).decode("utf-8")
        except Exception as e:
            error = str(e)
            
    return render_template("speech_filter_input.html",
                           spec_plot=spec_plot,
                           wave_plot=wave_plot,
                           pz_plot=pz_plot,
                           output_audio_data=output_audio_data,
                           error=error)

@speech_filter_input_bp.route("/live")
def live_audio():
    try:
        if "uploaded_audio_file" not in session:
            return "No audio uploaded", 400
        filename = session["uploaded_audio_file"]
        filepath = os.path.join(TEMP_UPLOAD_FOLDER, filename)
        sample_rate, audio = wavfile.read(filepath)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        duration = len(audio) / sample_rate
        
        filter_type = request.args.get("filter_type", "lowpass")
        order = int(request.args.get("order", 4))
        cutoff_str = request.args.get("cutoff", "100,400")
        
        filtered_audio, spec_plot, wave_plot, pz_plot = process_audio(
            filter_type, order, cutoff_str, sample_rate, audio, duration)
        
        out_buffer = BytesIO()
        filtered_int16 = np.int16(filtered_audio/np.max(np.abs(filtered_audio)) * 32767)
        wavfile.write(out_buffer, int(sample_rate), filtered_int16)
        out_buffer.seek(0)
        audio_data = base64.b64encode(out_buffer.read()).decode("utf-8")
        
        return jsonify({
            "spec_plot": spec_plot,
            "wave_plot": wave_plot,
            "pz_plot": pz_plot,
            "audio": audio_data
        })
    except Exception as e:
        return str(e), 400
