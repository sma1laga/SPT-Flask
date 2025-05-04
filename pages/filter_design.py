# pages/filter_design.py
from flask import Blueprint, render_template, request, redirect, url_for
import numpy as np
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt
from io import BytesIO
import base64

filter_design_bp = Blueprint(
    "filter_design",
    __name__,
    template_folder="../templates",
    url_prefix="/filter_design",
)
@filter_design_bp.route("/", methods=["GET", "POST"])
def filter_design():
    plot_url = None
    error = None
    if request.form.get("go_image_filter") == "1":
        return redirect(url_for("image_filter.home"))
    if request.method == "POST":
        try:
            # Get form data with default values
            filter_type = request.form.get("filter_type", "lowpass")
            order = int(request.form.get("order", 4))
            cutoff_str = request.form.get("cutoff", "100")
            sample_rate = float(request.form.get("sample_rate", 1000))
            noise_amplitude = float(request.form.get("noise_amplitude", 1))
            num_samples = int(request.form.get("num_samples", 1000))
            
            # Process cutoff frequency values.
            # For band filters, expect two comma-separated values.
            cutoff_vals = [float(x.strip()) for x in cutoff_str.split(",")]
            nyquist = sample_rate / 2
            normalized_cutoffs = [f / nyquist for f in cutoff_vals]
            
            if filter_type in ["bandpass", "bandstop"]:
                if len(normalized_cutoffs) != 2:
                    raise ValueError("For bandpass and bandstop filters, provide two cutoff frequencies separated by a comma.")
                normalized_cutoffs = sorted(normalized_cutoffs)
            else:
                if len(normalized_cutoffs) != 1:
                    raise ValueError("For lowpass and highpass filters, please provide a single cutoff frequency.")
                normalized_cutoffs = normalized_cutoffs[0]
            
            # Design a Butterworth filter
            b, a = butter(order, normalized_cutoffs, btype=filter_type)
            
            # Compute the filter's frequency response for plotting.
            w, h = freqz(b, a, worN=8000, fs=sample_rate)
            h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
            
            # Generate a white noise signal
            noise = noise_amplitude * np.random.randn(num_samples)
            
            # Apply the filter to the noise signal
            filtered_signal = filtfilt(b, a, noise)
            
            # Compute FFT of the original noise signal (frequency domain)
            freqs = np.fft.rfftfreq(num_samples, 1/sample_rate)
            orig_fft = np.abs(np.fft.rfft(noise))
            orig_fft_db = 20 * np.log10(np.maximum(orig_fft, 1e-10))
            
            # Compute FFT of the filtered noise signal (frequency domain)
            fft_vals = np.abs(np.fft.rfft(filtered_signal))
            fft_db = 20 * np.log10(np.maximum(fft_vals, 1e-10))
            
            # Create a figure with three subplots:
            # 1) Spectrum of original noise; 2) Spectrum of filtered noise; 3) Filter frequency response.
            fig, ax = plt.subplots(3, 1, figsize=(8, 12))
            
            # Plot 1: FFT Spectrum of Original Noise Signal
            ax[0].plot(freqs, orig_fft_db)
            # ── FIXED SCALE ────────────────────────────────────────
            ax[0].set_xlim(0, sample_rate/2)
            ax[0].set_ylim(-80,  80)    # set your desired dB‐range here
            ax[0].set_title("Spectrum of Original Noise Signal")
            ax[0].set_xlabel("Frequency (Hz)")
            ax[0].set_ylabel("Magnitude (dB)")
            ax[0].grid(True)
            
            # Plot 2: FFT Spectrum of Filtered Noise Signal
            ax[1].plot(freqs, fft_db, color="orange")
            # ── FIXED SCALE ────────────────────────────────────────
            ax[1].set_xlim(0, sample_rate/2)
            ax[1].set_ylim(-80,  80)    # same y‐range for consistency
            ax[1].set_title("Spectrum of Filtered Noise Signal")
            ax[1].set_xlabel("Frequency (Hz)")
            ax[1].set_ylabel("Magnitude (dB)")
            ax[1].grid(True)
            
            # Plot 3: Filter Frequency Response
            ax[2].plot(w, h_db, color="green")
            # ── FIXED SCALE ────────────────────────────────────────
            ax[2].set_xlim(0, sample_rate/2)
            ax[2].set_ylim(-60,  5)     # adjust to cover your filter’s pass/stop‐band
            ax[2].set_title("Filter Frequency Response")
            ax[2].set_xlabel("Frequency (Hz)")
            ax[2].set_ylabel("Magnitude (dB)")
            ax[2].grid(True)
            
            plt.tight_layout()
            
            # Save plot to a BytesIO buffer and encode as Base64.
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            plot_data = buf.getvalue()
            plot_url = base64.b64encode(plot_data).decode("utf-8")
            plt.close(fig)
        except Exception as e:
            error = f"Error: {e}"
    
    return render_template("filter_design.html", plot_url=plot_url, error=error)

@filter_design_bp.route("/live")
def live_plot():
    try:
        # Use query parameters to customize the live plot, falling back to default values
        filter_type = request.args.get("filter_type", "lowpass")
        order = int(request.args.get("order", 4))
        cutoff_str = request.args.get("cutoff", "100")
        sample_rate = float(request.args.get("sample_rate", 1000))
        noise_amplitude = float(request.args.get("noise_amplitude", 1))
        num_samples = int(request.args.get("num_samples", 1000))
        
        cutoff_vals = [float(x.strip()) for x in cutoff_str.split(",")]
        nyquist = sample_rate / 2
        normalized_cutoffs = [f / nyquist for f in cutoff_vals]
        if filter_type in ["bandpass", "bandstop"]:
            if len(normalized_cutoffs) != 2:
                raise ValueError("For bandpass and bandstop filters, provide two cutoff frequencies separated by a comma.")
            normalized_cutoffs = sorted(normalized_cutoffs)
        else:
            if len(normalized_cutoffs) != 1:
                raise ValueError("For lowpass and highpass filters, please provide a single cutoff frequency.")
            normalized_cutoffs = normalized_cutoffs[0]
        
        b, a = butter(order, normalized_cutoffs, btype=filter_type)
        w, h = freqz(b, a, worN=8000, fs=sample_rate)
        h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
        
        noise = noise_amplitude * np.random.randn(num_samples)
        filtered_signal = filtfilt(b, a, noise)
        
        freqs = np.fft.rfftfreq(num_samples, 1/sample_rate)
        orig_fft = np.abs(np.fft.rfft(noise))
        orig_fft_db = 20 * np.log10(np.maximum(orig_fft, 1e-10))
        
        fft_vals = np.abs(np.fft.rfft(filtered_signal))
        fft_db = 20 * np.log10(np.maximum(fft_vals, 1e-10))
        
        fig, ax = plt.subplots(3, 1, figsize=(6, 9))
        
        ax[0].plot(freqs, orig_fft_db)
        ax[0].set_title("Spectrum of Original Noise Signal")
        ax[0].set_xlabel("Frequency (Hz)")
        ax[0].set_ylabel("Magnitude (dB)")
        ax[0].grid(True)
        
        ax[1].plot(freqs, fft_db, color="orange")
        ax[1].set_title("Spectrum of Filtered Noise Signal")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("Magnitude (dB)")
        ax[1].grid(True)
        
        ax[2].plot(w, h_db, color="green")
        ax[2].set_title("Filter Frequency Response")
        ax[2].set_xlabel("Frequency (Hz)")
        ax[2].set_ylabel("Magnitude (dB)")
        ax[2].grid(True)
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {'Content-Type': 'image/png'}
    except Exception as e:
        return f"Error: {e}", 400
