"""Page contract for the continuous convolution module."""

import re

import pytest

from main import create_app

REQUIRED_IDS = ["func1", "func2", "convForm", "computeBtn", "clearBtn", "errorMsg"]


@pytest.fixture(scope="module")
def page():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/convolution/")
        assert resp.status_code == 200
        return resp.get_data(as_text=True)


@pytest.mark.parametrize("element_id", REQUIRED_IDS)
def test_control_is_present(page, element_id):
    assert re.search(rf'id="{re.escape(element_id)}"', page), f'missing id="{element_id}"'


def test_every_looked_up_id_exists(page):
    looked_up = set(re.findall(r"""getElementById\(\s*["']([A-Za-z0-9_\-]+)["']\s*\)""", page))
    present = set(re.findall(r"""\bid\s*=\s*["']([^"']+)["']""", page))
    optional = {"tab-sisy1", "tab-sisy2", "menu-sisy1", "menu-sisy2"}
    assert (looked_up - optional) <= present
