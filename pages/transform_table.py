# pages/transform_table.py
from flask import Blueprint, render_template

transform_table_bp = Blueprint("transform_table", __name__)
TABLES = [
    {
        "name": "Fourier Transform",
        "description": "Reference table for the continuous-time Fourier transform.",
        "file": "docs/continuous_fourier.pdf",
    },
    {
        "name": "Laplace Transform",
        "description": "Common Laplace transform pairs and properties.",
        "file": "docs/continuous_laplace.pdf",
    },
    {
        "name": "z-Transform",
        "description": "Useful z-transform relationships for discrete systems.",
        "file": "docs/z_transform.pdf",
    },
    {
        "name": "Discrete Fourier Transform (DFT)",
        "description": "DFT properties and identities for finite-length sequences.",
        "file": "docs/dft.pdf",
    },
]


@transform_table_bp.route("/")
def transform_table():
    """Display the available transform tables."""
    return render_template("transform_table.html", tables=TABLES)