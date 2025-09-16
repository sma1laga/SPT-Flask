# pages/demos/data.py
"""Shared demo metadata for menu and sidebar."""

DEMOS = {
    "Signals and Systems I": {
        "Lecture": [
            {
                "slug": "exponential",
                "title": "Complex Exponential Function",
                "title_desc": "3D spiral + Re/Im/|x|/phase",
                "desc": "Visualize x(t)=|X̂| e^{σt} e^{j(ωt+φ)}; full or up to t.",
                "endpoint": "demos_exponential.page",
            },
            {
                "slug": "fouriertransformation",
                "title": "Fourier Transform (Lecture 4)",
                "title_desc": "Magnitude & Phase of X(jω)",
                "desc": "Analytical pairs: rect, si, triangle, si²; interactive scaling & shifting.",
                "endpoint": "demos_fouriertransformation.page",
            },
            {
                "slug": "bandpass",
                "title": "Band-Pass Filter (Audio)",
                "title_desc": "FIR windowed-sinc & IIR biquad",
                "desc": "Design a band-pass and apply it to audio. See |H(e^{jω})| and listen to input vs. output.",
                "endpoint": "demos_bandpass.page",
            },
            {
                "slug": "systems-time-audio",
                "title": "Systems — Time Domain (Audio)",
                "title_desc": "LTI with real audio",
                "desc": "FIR/IIR systems (moving average, echo, comb, one-pole, custom b/a); listen to input vs. output; impulse response.",
                "endpoint": "demos_systems_time_audio.page",
            },
            {
                "slug": "stability-feedback",
                "title": "Stability & Feedback Systems",
                "title_desc": "Poles of H(s) and closed-loop Q(s)",
                "desc": "Interactive pole plots; background color indicates stability. Parameters: a, b, K.",
                "endpoint": "stability_feedback.page",
            },
            {
                "slug": "sampling",
                "title": "Sampling (Ch. 11)",
                "title_desc": "Time & frequency before/after sampling",
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
                "title_desc": "\(y[k] = x[k] + a·y[k-K]\)",
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
}
