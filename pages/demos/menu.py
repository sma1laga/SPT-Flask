# pages/demos/menu.py
from flask import Blueprint, render_template

demos_menu_bp = Blueprint(
    "demos_menu", __name__, template_folder="../../templates"
)

@demos_menu_bp.route("/", methods=["GET"])
def index():
    demos = [
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
            "slug": "kapitel8_2",
            "title": "Chapter 8.2",
            "title_desc": "Bandpass (Image)",
            "desc": "Bandpass im Ortsbereich: M und Ωg/π; Originalbild, h[n], Gefiltertes Bild, |H(e^{jΩ})|.",
            "endpoint": "demos_kapitel8_2.page",
        },
        {
            "slug": "kapitel8_audio",
            "title": "Chapter 8",
            "title_desc": "Low-Pass (Audio)",
            "desc": "Audio-LP: M, Ωg/π, optional Hann; DFT(x), h[k], DFT(y), |H(e^{jΩ})| + Audio.",
            "endpoint": "demos_kapitel8_audio.page",
        },
        {
            "slug": "kapitel11",
            "title": "Chapter 11",
            "title_desc": "Signal-Analysis (Time/DFT/AKF/LDS)",
            "desc": "Interaktiv: Sprache/Rauschen/etc. anzeigen als Zeitverlauf, DFT, AKF oder LDS.",
            "endpoint": "demos_kapitel11.page",
        },
        {
        "slug": "dtft_dft",
        "title": "DTFT & DFT",
        "title_desc": "Oscillation & Crop",
        "desc": "Vergleich: DTFT‑Impulse und DFT einer Länge‑M‑Ausschnitts des Cosinus.",
        "endpoint": "dtft_dft.page",
        },
        {
        "slug": "dtft_impulses",
        "title": "DTFT",
        "title_desc": "Analytic Cosine (Impulses)",
        "desc": "Zweifenster-Demo: Zeitsignal und Impuls-Spektrum bei ±ω₀ (ω₀/π ∈ [0,1]).",
        "endpoint": "dtft_impulses.page",
        },
    ]
    return render_template("demos/menu.html", demos=demos)
