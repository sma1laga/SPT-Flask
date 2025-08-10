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
            "title": "Kapitel 2 – Echo (Faltung x*h)",
            "desc": "Audio-Eingang + Echo-Impulsantwort (A, τ), Ausgabe via Faltung.",
            "endpoint": "demos_kapitel2.page",
        },
        {
            "slug": "kapitel4",
            "title": "Kapitel 4 – DFT (Fenster & Betragsspektrum)",
            "desc": "Audio wählen, Fensterposition verschieben, |DFT| anzeigen + Audio abspielen.",
            "endpoint": "demos_kapitel4.page",
        },
        {
            "slug": "kapitel6",
            "title": "Kapitel 6 – Bildfilter (2-Tap, zeilen/spalten)",
            "desc": "h[k]=a·δ[k]+b·δ[k−1]; zeilen- oder spaltenweise Faltung des Bildes.",
            "endpoint": "demos_kapitel6.page",
        },
        {
            "slug": "kapitel8_2",
            "title": "Kapitel 8.2 – Bandpass (Bild)",
            "desc": "Bandpass im Ortsbereich: M und Ωg/π; Originalbild, h[n], Gefiltertes Bild, |H(e^{jΩ})|.",
            "endpoint": "demos_kapitel8_2.page",
        },
        {
            "slug": "kapitel8_audio",
            "title": "Kapitel 8 – Tiefpass (Audio)",
            "desc": "Audio-LP: M, Ωg/π, optional Hann; DFT(x), h[k], DFT(y), |H(e^{jΩ})| + Audio.",
            "endpoint": "demos_kapitel8_audio.page",
        },
        {
            "slug": "kapitel11",
            "title": "Kapitel 11 – Signalanalyse (Zeit/DFT/AKF/LDS)",
            "desc": "Interaktiv: Sprache/Rauschen/etc. anzeigen als Zeitverlauf, DFT, AKF oder LDS.",
            "endpoint": "demos_kapitel11.page",
        },
    ]
    return render_template("demos/menu.html", demos=demos)
