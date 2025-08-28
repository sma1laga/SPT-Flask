# image_filter.py
from __future__ import annotations
import io, os, base64
from functools import lru_cache

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Blueprint, current_app, render_template, request
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from scipy import ndimage, signal

image_filter_bp = Blueprint(
    "image_filter",
    __name__,
    template_folder="./templates",
    url_prefix="/image_filter",
)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff"}
MAX_SIDE    = 512
DEMO_PATH   = "static/images/achilles.png"
# Maximum upload size in bytes (5 MB)
MAX_FILE_SIZE = 5 * 1024 * 1024
# ───────── helpers ────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _demo() -> np.ndarray:
    path = os.path.join(current_app.root_path, DEMO_PATH)
    return _prep(Image.open(path))

def _prep(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.asarray(img, dtype=np.float32) / 255.0

def _png(arr: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.axis("off"); ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    buf = io.BytesIO(); plt.savefig(buf, format="png",
                                    bbox_inches="tight", pad_inches=0); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ───────── Butterworth core ───────────────────────────────────────
def _butter(shape, cutoff, high=False, order=2):
    rows, cols = shape
    cy, cx = rows//2, cols//2
    y, x = np.ogrid[:rows,:cols]
    dist = np.hypot(y-cy, x-cx)
    norm = dist/dist.max()
    H = 1/(1+(norm/cutoff)**(2*order))
    return 1-H if high else H

def _ifft(img, H):
    F = np.fft.fft2(img)
    out = np.real(np.fft.ifft2(F * np.fft.ifftshift(H)))
    out -= out.min(); m = out.max()
    return out/m if m else out

def apply_custom_freq(img, kind, cutoff, order):
    if kind=="lowpass":
        return _ifft(img, _butter(img.shape, cutoff, False, order))
    if kind=="highpass":
        return _ifft(img, _butter(img.shape, cutoff, True, order))
    if kind=="bandpass":
        low, high = cutoff
        H = _butter(img.shape, high, False, order) * _butter(img.shape, low, True, order)
        return _ifft(img, H)
    return img

# ───────── preset spatial filters ─────────────────────────────────
PRESETS = {
    "blur":    lambda im: ndimage.gaussian_filter(im, sigma=3),
    "sharpen": lambda im: np.clip(im + (im - ndimage.gaussian_filter(im, 1)), 0, 1),
    "edge":    lambda im: np.sqrt(ndimage.sobel(im,0)**2 + ndimage.sobel(im,1)**2) / np.sqrt(2),
    "emboss":  lambda im: np.clip(ndimage.convolve(
                  im, np.array([[-2,-1,0],[-1,1,1],[0,1,2]])), 0, 1)
}

# ───────── plots & metrics ────────────────────────────────────────
def _dft(arr):
    F = np.fft.fftshift(np.fft.fft2(arr))
    mag = 20*np.log10(np.abs(F)+1e-3); mag-=mag.min(); mag/=mag.max()
    return mag

def _autocorr(arr):
    F = np.fft.fft2(arr)
    c = np.fft.ifft2(np.abs(F)**2)
    c = np.fft.fftshift(np.real(c)); c-=c.min(); c/=c.max()
    return c

def _psd_plot(arr, y):
    f, P = signal.welch(arr[y,:], nperseg=min(256, arr.shape[1]))
    fig, ax = plt.subplots(figsize=(3,2), dpi=110)
    ax.plot(f, P); ax.set_xlabel("Normalised frequency"); ax.set_ylabel("PSD")
    fig.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def radial_profile(img_fft_mag):
    rows, cols = img_fft_mag.shape
    cy, cx = rows//2, cols//2
    y, x = np.indices((rows, cols))
    r = np.hypot(x-cx, y-cy).astype(int)
    tbin = np.bincount(r.ravel(), img_fft_mag.ravel())
    nr   = np.bincount(r.ravel())
    return tbin/np.maximum(nr, 1)

# ───────── main route ─────────────────────────────────────────────
@image_filter_bp.route("/", methods=["GET", "POST"])
def home():
    # initialise form context
    context = {
        "img_choice":"default","use_standard_filter":False,
        "standard_filter":"blur","filter_type":"lowpass",
        "order":2,"cutoff":"0.1","y_scan":0,"error":None,
        "diff_img":None,
        "orig_prof": [], "filt_prof":  [],
    }
    try:
        if request.method=="POST":
            # 1) choose image
            context["img_choice"]=img_choice=request.form.get("img_choice","default")
            if img_choice=="upload":
                f = request.files.get("image_file")
                if not f or not f.filename:
                    raise ValueError("No image uploaded")
                filename = secure_filename(f.filename)
                if not filename:
                    raise ValueError("Invalid filename")
                ext = filename.rsplit(".", 1)[-1].lower()
                if ext not in ALLOWED_EXT:
                    raise ValueError("Unsupported file type")
                f.stream.seek(0, os.SEEK_END)
                if f.stream.tell() > MAX_FILE_SIZE:
                    raise ValueError("File too large")
                f.stream.seek(0)
                try:
                    img_file = Image.open(f.stream)
                except UnidentifiedImageError:
                    raise ValueError("Invalid image file")
                if img_file.format.lower() not in ALLOWED_EXT:
                    raise ValueError("Unsupported file type")
                img = _prep(img_file)
            else:
                img=_demo()

            # 2) scan‑line
            try: y=int(request.form.get("y_scan",-1))
            except: y=-1
            if not 0<=y<img.shape[0]: y=img.shape[0]//2
            context["y_scan"]=y

            # 3) filtering branch
            if request.form.get("use_standard_filter"):
                context["use_standard_filter"]=True
                std=request.form.get("standard_filter","blur"); context["standard_filter"]=std
                if std not in PRESETS: raise ValueError("Unknown preset")
                filtered=PRESETS[std](img)
                # mask preview (spatial kernel)
                if std=="blur":
                    x=np.linspace(-1,1,21);X,Y=np.meshgrid(x,x)
                    ker=np.exp(-(X*X+Y*Y)/(2*(0.3)**2)); ker/=ker.sum()
                elif std=="sharpen":
                    ker=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],float)
                elif std=="edge":
                    ker=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)
                else:
                    ker=np.array([[-2,-1,0],[-1,1,1],[0,1,2]],float)
                norm_ker=(ker-ker.min())/(ker.max()-ker.min()+1e-8)
                mask_img=_png(norm_ker)
                mask_label=f"Spatial kernel for '{std}' preset."
            else:
                kind=request.form.get("filter_type","lowpass"); context["filter_type"]=kind
                order=int(request.form.get("order",2));       context["order"]=order
                cutstr=request.form.get("cutoff","0.1");      context["cutoff"]=cutstr
                cuts=[float(c.strip()) for c in cutstr.split(",") if c.strip()]
                if kind=="bandpass" and len(cuts)!=2:
                    raise ValueError("Need two cutoffs for bandpass")
                cutoff=cuts if kind=="bandpass" else cuts[0]
                filtered=apply_custom_freq(img, kind, cutoff, order)
                # freq‑domain mask preview
                if kind in ("lowpass","highpass"):
                    H=_butter(img.shape, cutoff, high=(kind=="highpass"), order=order)
                else:
                    H=_butter(img.shape, cuts[1], False, order) * _butter(img.shape, cuts[0], True, order)
                mask_img=_png(H); mask_label="Frequency-domain mask H(u,v)."

            # 4) difference image
            diff=np.abs(filtered-img); diff/=diff.max() or 1
            context["diff_img"]=_png(diff)

            # 5) radial profiles
            F0=np.fft.fftshift(np.fft.fft2(img))
            prof0=radial_profile(np.log10(np.abs(F0)+1e-3))
            Ff=np.fft.fftshift(np.fft.fft2(filtered))
            proff=radial_profile(np.log10(np.abs(Ff)+1e-3))
            context["orig_prof"]=prof0.tolist(); context["filt_prof"]=proff.tolist()

            # 6) main images & plots
            orig_img=_png(img); filt_img=_png(filtered)
            dft_img=_png(_dft(filtered)); acorr_img=_png(_autocorr(filtered))
            psd_img=_psd_plot(filtered, y)

            # 7) render
            return render_template("image_filter.html",
                orig_img=orig_img, filt_img=filt_img,
                dft_img=dft_img, acorr_img=acorr_img, psd_img=psd_img,
                mask_img=mask_img, mask_label=mask_label, **context
            )
    except Exception as e:
        context["error"]=str(e)

    # GET or error
    return render_template("image_filter.html", **context)
