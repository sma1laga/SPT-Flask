# pages/demos/data.py
"""Shared demo metadata for menu and sidebar."""

DEMOS = {
    "Signals and Systems I": {
        "Lecture": [
            {
                "slug": "exponential",
                "title": "Chapter 2-1",
                "title_desc": "Exponential Function",
                "desc": "Visualize x(t)=|X̂| e^{σt} e^{j(ωt+φ)}; full or up to t.",
                "endpoint": "demos_exponential.page",
            },
            {
                "slug": "convolution",
                "title": "Chapter 2-2",
                "title_desc": "Convolution",
                "desc": "Visualize the convolution of two signals.",
                "endpoint": "demos_convolution.page",
            },
            {
                "slug": "fouriertransformation",
                "title": "Chapter 3",
                "title_desc": "Fourier Transform",
                "desc": "Analytical pairs: rect, si, triangle, si²; interactive scaling & shifting.",
                "endpoint": "demos_fouriertransformation.page",
            },
            {
                "slug": "systems-time-audio",
                "title": "Chapter 5",
                "title_desc": "Systems - Time Domain (Audio)",
                "desc": "FIR/IIR systems (moving average, echo, comb, one-pole, custom b/a); listen to input vs. output; impulse response.",
                "endpoint": "demos_systems_time_audio.page",
            },
            {
                "slug": "bandpass",
                "title": "Chapter 8",
                "title_desc": "Ideal Band-Pass",
                "desc": "Design a band-pass and apply it to audio. See |H(e^{jω})| and listen to input vs. output.",
                "endpoint": "demos_bandpass.page",
            },
            {
                "slug": "stability-feedback",
                "title": "Chapter 10",
                "title_desc": "Stability and Feedback Systems",
                "desc": "Interactive pole plots; background color indicates stability. Parameters: a, b, K.",
                "endpoint": "stability_feedback.page",
            },
            {
                "slug": "sampling",
                "title": "Chapter 11",
                "title_desc": "Sampling",
                "desc": "x(t)=ω_g/(2π)·si²(½ω_g t); spectra X, X_a, Y; adjustable ω_g and ω_a; optional sinc components.",
                "endpoint": "sampling.page",
            },
        ],
        "Tutorial": [],
    },
    "Signals and Systems II": {
        "Lecture": [
            {
                "slug": "kapitel2",
                "title": "Chapter 2",
                "title_desc": "Echo (Convolution)",
                "desc": "Audio-Eingang + Echo-Impulsantwort (A, τ), Ausgabe via Faltung.",
                "endpoint": "demos_kapitel2.page",
            },
            {
                "slug": "kapitel4",
                "title": "Chapter 4",
                "title_desc": "DFT (Audio)",
                "desc": "Audio wählen, Fensterposition verschieben, |DFT| anzeigen + Audio abspielen.",
                "endpoint": "demos_kapitel4.page",
            },
            {
                "slug": "kapitel6",
                "title": "Chapter 6",
                "title_desc": "Image-Filter",
                "desc": "h[k]=a·δ[k]+b·δ[k−1]; zeilen- oder spaltenweise Faltung des Bildes.",
                "endpoint": "demos_kapitel6.page",
            },
            {
                "slug": "kapitel8_audio",
                "title": "Chapter 8-1",
                "title_desc": "Low-Pass (Audio)",
                "desc": "Audio-LP: M, Ωg/π, optional Hann; DFT(x), h[k], DFT(y), |H(e^{jΩ})| + Audio.",
                "endpoint": "demos_kapitel8_audio.page",
            },
            {
                "slug": "kapitel8_2",
                "title": "Chapter 8-2",
                "title_desc": "Band-Pass (Image)",
                "desc": "Bandpass im Ortsbereich: M und Ωg/π; Originalbild, h[n], Gefiltertes Bild, |H(e^{jΩ})|.",
                "endpoint": "demos_kapitel8_2.page",
            },
            {
                "slug": "kapitel11",
                "title": "Chapter 11",
                "title_desc": "Noise Analysis",
                "desc": "Interaktiv: Sprache/Rauschen/etc. anzeigen als Zeitverlauf, DFT, AKF oder LDS.",
                "endpoint": "demos_kapitel11.page",
            },
        ],
        "Tutorial": [
            {
                "slug": "dtft_impulses",
                "title": "DTFT",
                "title_desc": "Discrete Cosine",
                "desc": "Zweifenster-Demo: Zeitsignal und Impuls-Spektrum bei ±ω₀ (ω₀/π ∈ [0,1]).",
                "endpoint": "dtft_impulses.page",
            },
            {
                "slug": "dtft_dft",
                "title": "DTFT & DFT",
                "title_desc": "Discrete Cosine",
                "desc": "Vergleich: DTFT-Impulse und DFT einer Länge-M-Ausschnitts des Cosinus.",
                "endpoint": "dtft_dft.page",
            },
            {
                "slug": "z_trafo",
                "title": "DTFT & z-Transform",
                "title_desc": "Damped Cosine",
                "desc": "x[k]=a^k·cos(ω₀ k)·u[k]; Zeitfolge, DTFT und PN-Diagramm (z).",
                "endpoint": "demos_z_trafo.page",
            },
            {
                "slug": "iir",
                "title": "IIR Echo",
                "title_desc": r"\(y[k] = x[k] + a·y[k-K]\)",
                "desc": "Convolution view via impulse response h plus inline audio for x and y.",
                "endpoint": "demos_iir.page",
            },
            {
                "slug": "filter",
                "title": "Filter Demo",
                "title_desc": "Filter Analysis",
                "desc": "Apply IIR/FIR to audio; interactive plots + audio preview.",
                "endpoint": "demos_filter.page",
            },
        ],
    },
    "Image and Video Compression": {
        "Lecture": [
            {
                "slug": "spatial-prediction-1",
                "title": "Spatial Prediction 1",
                "title_desc": "Linear Predictors",
                "desc": "Compare 2D linear predictors on Lenna with entropy and compression of the error image.",
                "endpoint": "demos_spatial_prediction.page",
            },
            {
                "slug": "compression",
                "title": "Compression",
                "title_desc": "Image & Video",
                "desc": "Compare lossless ZIP with weak and strong JPEG compression using the classic Lenna image.",
                "endpoint": "demos_compression.page",
            },
            {
                "slug": "lloyd-max",
                "title": "Lloyd-Max",
                "title_desc": "Iterative Quantizer",
                "desc": "Step through Lloyd-Max iterations on Lenna and compare to uniform 16-level quantization.",
                "endpoint": "demos_lloyd_max.page",
            },
            {
                "slug": "huffman",
                "title": "Huffman coding",
                "title_desc": "Entropy coding",
                "desc": "Visualize histograms, entropy, and codeword lengths for Lenna Y/Cr channels and white noise.",
                "endpoint": "demos_huffman.page",
            },
            {
                "slug": "zonal-dct-coding",
                "title": "Zonal DCT Coding",
                "title_desc": "8×8 Masked Quantization",
                "desc": "Compare multiple zonal quantization masks on Lenna with entropy, SNR, PSNR, and coefficient heatmaps.",
                "endpoint": "zonal_dct.page",
            }
        ],
        "Tutorial": [],
    },
}
