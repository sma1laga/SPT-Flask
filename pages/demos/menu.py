# pages/demos/menu.py
from flask import Blueprint, render_template

demos_menu_bp = Blueprint(
    "demos_menu", __name__, template_folder="../../templates"
)

@demos_menu_bp.route("/", methods=["GET"])
def index():
    """Render the demo overview.

    The demos are grouped in a nested dictionary so that the template can
    display them in separate "Course" and "Tutorial" cards for each of the
    two lecture parts (Signals and Systems I & II).
    """

    demos = {
        "Signals and Systems I": {
            "Course": [
                {
                "slug": "exponential",
                "title": "Complex Exponential Function",
                "title_desc": "3D spiral + Re/Im/|x|/phase",
                "desc": "Visualize x(t)=|X̂| e^{σt} e^{j(ωt+φ)}; full or up to t.",
                "endpoint": "demos_exponential.page",
                },
                ],
            "Tutorial": [],
        },
        "Signals and Systems II": {
            "Course": [
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
                    "desc": "Vergleich: DTFT‑Impulse und DFT einer Länge‑M‑Ausschnitts des Cosinus.",
                    "endpoint": "dtft_dft.page",
                },
                {
                    "slug": "z_trafo",
                    "title": "DTFT & z-Transform",
                    "title_desc": "Damped cosine",
                    "desc": "x[k]=a^k·cos(ω₀ k)·u[k]; Zeitfolge, DTFT und PN‑Diagramm (z).",
                    "endpoint": "demos_z_trafo.page",
                },
                {
                "slug": "iir",
                "title": "IIR Echo",
                "title_desc": "y[k] = x[k] + a·y[k−K]",
                "desc": "Convolution view via impulse response h plus inline audio for x and y.",
                "endpoint": "demos_iir.page",
                },
                {
                "slug": "filter",
                "title": "Filter Demo",
                "title_desc": "Magnitude, Phase, Pole–Zero, Impulse",
                "desc": "Apply IIR/FIR to audio; interactive plots + audio preview.",
                "endpoint": "demos_filter.page",
                },

            ],
        },
    }
    return render_template("demos/menu.html", demos=demos)
