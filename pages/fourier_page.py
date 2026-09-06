from flask import Blueprint, render_template

fourier_bp = Blueprint("fourier", __name__)


@fourier_bp.route("/", methods=["GET"])
def fourier():
    """Render the page. The transform itself is computed in the browser by
    static/js/fourier_compute.js, so there is no server-side counterpart to keep
    in sync -- an earlier duplicate here had drifted apart from the client in
    threshold, grid size and delta width."""
    return render_template("fourier.html")
