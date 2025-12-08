"""Huffman coding demo for luminance, chrominance, white noise"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, render_template
from imageio.v2 import imread, imwrite
from skimage import color

@dataclass(frozen=True)
class HuffmanVariant:
    key: str
    label: str
    description: str
    entropy: float
    average_length: float
    original_size: int
    compressed_size: int
    image_uri: str
    histogram_uri: str
    length_uri: str


ROOT = Path(__file__).resolve().parent.parent.parent
LENNA_PATH = ROOT / "static" / "demos" / "images" / "lenna.png"
_rng = np.random.default_rng(7)


def _buffer_to_uri(buf: io.BytesIO) -> str:
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _array_to_uri(array: np.ndarray) -> str:
    buf = io.BytesIO()
    imwrite(buf, array.astype(np.uint8), format="png")
    return _buffer_to_uri(buf)


def _huffman_lengths(freqs: Iterable[int]) -> Dict[int, int]:
    heap: List[Tuple[int, int, Tuple[int, int]]] = []
    counter = count()
    for symbol, count_val in enumerate(freqs):
        if count_val > 0:
            heappush(heap, (int(count_val), next(counter), (symbol, -1)))

    if not heap:
        return {}

    if len(heap) == 1:
        symbol = heap[0][2][0]
        return {symbol: 1}

    while len(heap) > 1:
        w1, _, n1 = heappop(heap)
        w2, _, n2 = heappop(heap)
        heappush(heap, (w1 + w2, next(counter), (-1, len(heap)) + (n1, n2)))

    _, _, root = heap[0]

    lengths: Dict[int, int] = {}

    def _walk(node: Tuple[int, ...], depth: int) -> None:
        symbol = node[0]
        if symbol >= 0:
            lengths[symbol] = max(depth, 1)
            return
        _walk(node[2], depth + 1)
        _walk(node[3], depth + 1)

    _walk(root, 0)
    return lengths


def _entropy(freqs: np.ndarray) -> float:
    total = freqs.sum()
    if total == 0:
        return 0.0
    probs = freqs / total
    mask = probs > 0
    return float((-probs[mask] * np.log2(probs[mask])).sum())


def _plot_histogram(freqs: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    values = np.arange(freqs.size)
    ax.bar(values, freqs, color="#4a90e2", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Grey level")
    ax.set_ylabel("Frequency of occurrence")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return _buffer_to_uri(buf)


def _plot_lengths(lengths_map: Dict[int, int]) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    values = np.arange(256)
    lengths = np.array([lengths_map.get(v, 0) for v in values])
    ax.bar(values, lengths, color="#7a68c2", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Grey level")
    ax.set_ylabel("Codeword length [bit]")
    ax.set_xticks(np.arange(0, 256, 32))
    ax.set_xlim(-0.5, 255.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return _buffer_to_uri(buf)


def _huffman_stats(channel: np.ndarray, label: str, description: str, key: str) -> HuffmanVariant:
    flat = channel.ravel().astype(np.uint8)
    freqs = np.bincount(flat, minlength=256)
    lengths_map = _huffman_lengths(freqs)
    entropy = _entropy(freqs)

    if lengths_map:
        max_len = max(lengths_map.values())
        length_hist = np.zeros(max_len + 1, dtype=int)
        total_bits = 0
        for symbol, count_val in enumerate(freqs):
            if count_val == 0:
                continue
            length = lengths_map.get(symbol, 0)
            length_hist[length] += int(count_val)
            total_bits += int(count_val) * length
        avg_len = total_bits / flat.size if flat.size else 0.0
    else:
        length_hist = np.zeros(1, dtype=int)
        avg_len = 0.0
        total_bits = 0

    original_size = flat.size
    compressed_size = math.ceil(total_bits / 8) if total_bits else 0

    histogram_uri = _plot_histogram(freqs)
    length_uri = _plot_lengths(lengths_map)
    image_uri = _array_to_uri(channel)

    return HuffmanVariant(
        key=key,
        label=label,
        description=description,
        entropy=entropy,
        average_length=avg_len,
        original_size=original_size,
        compressed_size=compressed_size,
        image_uri=image_uri,
        histogram_uri=histogram_uri,
        length_uri=length_uri,
    )


def _prepare_variants() -> Dict[str, HuffmanVariant]:
    if not LENNA_PATH.exists():
        raise FileNotFoundError(f"Lenna image missing at {LENNA_PATH}")
    rgb = imread(LENNA_PATH)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    if rgb.dtype != np.uint8:
        rgb = (255 * np.clip(rgb, 0, 1)).astype(np.uint8)

    ycbcr = color.rgb2ycbcr(rgb)
    y_channel = np.clip(ycbcr[..., 0], 0, 255).astype(np.uint8)
    cr_channel = np.clip(ycbcr[..., 2], 0, 255).astype(np.uint8)

    noise = _rng.integers(0, 256, size=y_channel.shape, dtype=np.uint8)

    return {
        "luminance": _huffman_stats(
            y_channel,
            label="Huffman coding of luminance component Y",
            description="Distribution and coding efficiency for the Lenna image's Y channel.",
            key="luminance",
        ),
        "chrominance": _huffman_stats(
            cr_channel,
            label="Huffman coding of chrominance component Cr",
            description="Histogram and Huffman code lengths for the Lenna image's Cr channel.",
            key="chrominance",
        ),
        "noise": _huffman_stats(
            noise,
            label="Huffman coding of uniformly distributed white noise",
            description="Reference for a flat histogram with uniformly distributed white noise.",
            key="noise",
        ),
    }


_VARIANTS = _prepare_variants()


demos_huffman_bp = Blueprint("demos_huffman", __name__, template_folder="../../templates")


@demos_huffman_bp.route("/", methods=["GET"], endpoint="page")
def page() -> str:
    return render_template(
        "demos/huffman.html",
        variants={key: variant.__dict__ for key, variant in _VARIANTS.items()},
    )