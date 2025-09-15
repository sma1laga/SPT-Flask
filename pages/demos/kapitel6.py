# pages/demos/kapitel6.py
from flask import Blueprint, render_template, request, jsonify, current_app
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.style.use("fast")
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.signal import convolve2d
from utils.img import fig_to_base64

demos_kapitel6_bp = Blueprint(
    "demos_kapitel6", __name__, template_folder="../../templates"
)

# Notebook image options -> filenames under static/demos/images/
IMAGE_MAP = {
    "Arnold": "arnold.png",
    "Jack":   "jack.png",
}

# ---- helpers (faithful to the notebook) ----

def _image_path(filename: str) -> str:
    static_root = current_app.static_folder  # .../static
    return os.path.join(static_root, "demos", "images", filename)

def _load_gray_float(path: str):
    x = imread(path, as_gray=True)
    x = x.astype(np.float32)
    return x

def _impulse_response(peak2: float):
    # h = [1 - max(0, b), b], exactly like the nb
    return np.array([1.0 - max(0.0, peak2), peak2], dtype=np.float32)

def _filter_image(x: np.ndarray, h: np.ndarray, row_wise: bool):
    ker = h[None, :] if row_wise else h[:, None]
    y = convolve2d(x, ker, mode="same")
    return y

# ---- routes ----

@demos_kapitel6_bp.route("/", methods=["GET"])
def page():
    defaults = {
        "x_type": "Arnold",
        "peak2": 0.0,                # b in [-1, 0.5]
        "filtering": "Row-wise",  # or Spaltenweise
    }
    return render_template(
        "demos/kapitel6.html",
        image_options=list(IMAGE_MAP.keys()),
        defaults=defaults
    )

@demos_kapitel6_bp.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True) or {}
        x_type     = (data.get("x_type", "Arnold")).strip()
        peak2      = float(data.get("peak2", 0.0))
        filtering  = (data.get("filtering", "Row-wise")).strip()
        row_wise   = True if filtering == "Row-wise" else False

        filename = IMAGE_MAP.get(x_type)
        if not filename:
            return jsonify(error=f"Unknown image: {x_type}"), 400

        path = _image_path(filename)
        if not os.path.exists(path):
            return jsonify(error=f"Image file not found: {filename}", path=path), 500

        x = _load_gray_float(path)
        h = _impulse_response(peak2)
        y = _filter_image(x, h, row_wise=row_wise)

        fig, axs = plt.subplots(2, 2, figsize=(9.0, 5.4), gridspec_kw={'height_ratios': [1, 2]}, layout='constrained')
        x_axis = axs[1, 0]
        y_axis = axs[1, 1]
        gs = axs[0, 0].get_gridspec()
        for ax in axs[0, :]:
            ax.remove()
        h_axis = fig.add_subplot(gs[0, :])

        h_axis.set_xlabel("Index $k$")
        h_axis.set_ylabel("$h[k]$")
        h_axis.set_title("Impulse Response")
        h_axis.set_ylim(-1.1, 1.1)
        h_axis.set_xlim(-1.1, 6.1)
        h_axis.set_yticks(np.arange(-1, 1.1, 0.5))
        h_axis.hlines(0, -1.1, 6.1, color='black', linewidth=1)
        h_axis.grid(True)

        nz = np.nonzero(h)[0]
        if nz.size > 0:
            for k in nz:
                h_axis.vlines([k], 0, h[k], linewidth=2)
        # marker line (add zeros around to mimic nb's look)
        h_line_x = np.arange(-1, 7)
        h_line_y = np.array([0, *h, 0, 0, 0, 0, 0], dtype=float)
        h_axis.plot(h_line_x, h_line_y, 'o', markersize=7)

        # images
        x_axis.imshow(x, cmap="gray", interpolation='none')
        x_axis.set_title("Input")
        x_axis.axis("off")

        y_axis.imshow(np.abs(y), cmap="gray", interpolation='none')
        y_axis.set_title("Output (Magnitude)")
        y_axis.axis("off")

        png = fig_to_base64(fig)

        return jsonify({"image": png})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500
