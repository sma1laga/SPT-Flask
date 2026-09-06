"""Page contract for the Fourier module.

The transform itself is tested in ``test_fourier_js.py``; what is checked here is
that the page still hands the browser everything its inline script binds to. A
renamed id makes ``getElementById(...).addEventListener`` throw, which kills the
rest of the script and leaves the page silently dead.
"""

import re

import pytest

from main import create_app

# Ids the inline script in templates/fourier.html looks up unconditionally.
REQUIRED_IDS = [
    "func", "errorMsg", "fourierForm", "computeBtn", "clearBtn",
    "shift", "scale", "modulate", "phase",              # Fourier property sliders
    "shiftOut", "scaleOut", "modulateOut", "phaseOut",  # their readouts
    "resetProps", "normalize",
    "specView", "phaseMode", "omegaRange", "downloadCsv",
    "timePlot", "magPlot", "phasePlot",
    "leakageNote", "fourier_transformation_label",
]


@pytest.fixture(scope="module")
def page():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/fourier/")
        assert resp.status_code == 200
        return resp.get_data(as_text=True)


@pytest.mark.parametrize("element_id", REQUIRED_IDS)
def test_control_is_present(page, element_id):
    assert re.search(rf'id="{re.escape(element_id)}"', page), f'missing id="{element_id}"'


def test_every_looked_up_id_exists(page):
    """Catches a control that is renamed in the markup but not in the script."""
    looked_up = set(re.findall(r"""getElementById\(\s*["']([A-Za-z0-9_\-]+)["']\s*\)""", page))
    present = set(re.findall(r"""\bid\s*=\s*["']([^"']+)["']""", page))
    # base.html looks up the demo tab bar on every page and guards the result.
    optional = {"tab-sisy1", "tab-sisy2", "menu-sisy1", "menu-sisy2"}
    assert (looked_up - optional) <= present


def test_property_sliders_start_neutral(page):
    """A stale default would show a transformed signal as if it were x(t)."""
    for element_id, value in [("shift", "0"), ("scale", "1"), ("modulate", "0"), ("phase", "0")]:
        pattern = rf'id="{element_id}"[^>]*value="{value}"'
        assert re.search(pattern, page), f"{element_id} does not default to {value}"


def test_page_offers_the_correspondence_table(page):
    assert "continuous_fourier.pdf" in page
