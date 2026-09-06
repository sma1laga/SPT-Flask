"""Pins the browser-side signal helpers against ``utils/math_utils.py``.

``static/js/signal_functions.js`` is the single source of truth for the JS
modules. These tests keep it from drifting away from the Python definitions --
that drift is how ``si()`` ended up as ``sin(pi*t)/(pi*t)`` in the Fourier module
while the rest of the toolkit used ``sin(t)/t`` -- and make sure no module
reintroduces a private copy of the helpers.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from utils import math_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
JS_DIR = REPO_ROOT / "static" / "js"
SHARED = JS_DIR / "signal_functions.js"

# Helpers that must only ever be defined in signal_functions.js.
SHARED_NAMES = ["rect", "tri", "step", "sign", "delta", "inv_t", "si", "exp_iwt"]


def test_shared_library_defines_the_helpers():
    source = SHARED.read_text(encoding="utf-8")
    for name in SHARED_NAMES:
        assert re.search(rf"function {name}\s*\(", source), f"{name} missing from signal_functions.js"


@pytest.mark.parametrize("js_file", sorted(JS_DIR.glob("*.js")))
def test_no_module_redefines_the_helpers(js_file):
    """A private copy is how the definitions drift apart -- keep them out."""
    if js_file.name in {"signal_functions.js", "plotly.min.js", "fft.js"}:
        pytest.skip("library file")
    source = js_file.read_text(encoding="utf-8", errors="replace")
    duplicated = [n for n in SHARED_NAMES if re.search(rf"function {n}\s*\(\s*t\s*[,)]", source)]
    assert not duplicated, f"{js_file.name} redefines {duplicated}; import from SPTSignals instead"


# --- numeric parity with utils/math_utils.py --------------------------------

_T = np.linspace(-3.0, 3.0, 61)

# exp_iwt is complex in Python and real-only in JS; delta is a grid-dependent
# approximation whose default width differs per module, so it is compared at a
# fixed eps to pin the formula rather than the default.
_CASES = {
    "rect": math_utils.rect(_T),
    "tri": math_utils.tri(_T),
    "step": math_utils.step(_T),
    "sign": math_utils.sign(_T),
    "cos": math_utils.cos(_T),
    "sin": math_utils.sin(_T),
    "si": math_utils.si(_T),
    "inv_t": math_utils.inv_t(_T),
    "delta": math_utils.delta(_T, eps=1e-3),
    "exp_iwt": np.real(math_utils.exp_iwt(_T)),
}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_helpers_match_python():
    script = f"""
    global.window = {{}};
    require({json.dumps(str(SHARED))});
    const S = global.window.SPTSignals;
    const t = {json.dumps(_T.tolist())};
    const names = {json.dumps(sorted(_CASES))};
    const out = {{}};
    for (const n of names) out[n] = t.map(v => S[n](v));
    process.stdout.write(JSON.stringify(out));
    """
    proc = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=60, cwd=REPO_ROOT
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)

    for name, expected in _CASES.items():
        np.testing.assert_allclose(
            got[name], expected, rtol=1e-9, atol=1e-9,
            err_msg=f"{name}() differs between signal_functions.js and utils/math_utils.py",
        )
