"""Whole-site smoke tests.

The toolkit serves well over a hundred pages, and the usual way one breaks is
quiet: a route keeps pointing at a template or a static file that was deleted
somewhere else. ``/info/terms`` sat broken in production that way, and
``image_filter.html`` pulled a script that never existed in the repository. The
sweeps below make both classes of failure fail the build instead.
"""

import os
import re
from urllib.parse import urlparse
from xml.etree import ElementTree

import pytest

from main import create_app

app = create_app()
app.config["TESTING"] = True

STATIC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
STATIC_REFERENCE = re.compile(r"""["'(]/static/([^"'()\s>]+)""")


def _public_get_rules():
    """Every page a visitor can reach with a plain GET and no parameters."""
    skip = {"static", "sitemap", "robots", "favicon", "google_site_verification"}
    return sorted(
        (r.rule for r in app.url_map.iter_rules()
         if "GET" in r.methods and not r.arguments
         and r.endpoint not in skip and not r.rule.startswith("/static")),
    )


PUBLIC_RULES = _public_get_rules()


@pytest.fixture(scope="module")
def client():
    with app.test_client() as c:
        yield c


# --- the pages that have to be up ------------------------------------------

@pytest.mark.parametrize("url", [
    "/",
    "/plot_function/",
    "/fourier/",
    "/convolution/",
    "/convolution/dynamic/",
    "/autocorrelation/",
    "/discrete/autocorrelation/",
    "/discrete/plot_functions/",
    "/discrete/dft/",
    "/block_diagram/",
    "/discrete/direct_plot/",
    "/inverse_z/",
    "/inverse_laplace/",
    "/demos/",
    "/info/about",
    "/info/impressum",
    "/info/privacy",
    "/info/news",
    "/info/hall-of-fame",
])
def test_core_pages_render(client, url):
    resp = client.get(url)
    assert resp.status_code == 200, f"{url} returned {resp.status_code}"


def test_removed_pages_are_not_advertised_as_broken():
    """A deleted page must disappear from the routing table, not 500 forever."""
    assert "/info/terms" not in PUBLIC_RULES


# --- nothing anywhere may raise --------------------------------------------

@pytest.fixture(scope="module")
def visited(client):
    """One pass over the whole site, shared by every sweep below.

    Some demo pages render images on GET and take seconds each, so fetching them
    once per test would triple the runtime of the suite.
    """
    pages = {}
    for url in PUBLIC_RULES:
        resp = client.get(url)
        html = resp.get_data(as_text=True) if resp.mimetype == "text/html" else None
        pages[url] = (resp.status_code, html)
    return pages


@pytest.mark.parametrize("url", PUBLIC_RULES)
def test_no_page_raises(visited, url):
    """A 4xx can be correct (a download endpoint without parameters), a 5xx never is."""
    status, _ = visited[url]
    assert status < 500, f"{url} returned {status}"


# --- what the pages reference must exist -----------------------------------

def _rendered_html_pages(visited):
    for url, (status, html) in visited.items():
        if status == 200 and html is not None:
            yield url, html


def test_referenced_static_files_exist(visited):
    """A missing asset is a silent 404 in the browser, not a failing page."""
    missing = {}
    checked = set()
    for url, html in _rendered_html_pages(visited):
        for reference in STATIC_REFERENCE.findall(html):
            filename = reference.split("?")[0]
            if filename in checked:
                continue
            checked.add(filename)
            if not os.path.exists(os.path.join(STATIC_ROOT, *filename.split("/"))):
                missing.setdefault(filename, url)
    assert checked, "no static references found -- the extraction is broken"
    assert not missing, "missing static files: " + ", ".join(
        f"{name} (referenced by {where})" for name, where in sorted(missing.items())
    )


# --- what search engines are told ------------------------------------------

def test_sitemap_lists_only_real_routes(client):
    resp = client.get("/sitemap.xml")
    assert resp.status_code == 200
    root = ElementTree.fromstring(resp.get_data())
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    paths = [urlparse(loc.text).path for loc in root.findall(".//sm:loc", namespace)]
    assert paths, "the sitemap is empty"
    unknown = [p for p in paths if p not in PUBLIC_RULES]
    assert not unknown, f"sitemap advertises routes that do not exist: {unknown}"


def test_robots_points_at_the_sitemap(client):
    resp = client.get("/robots.txt")
    assert resp.status_code == 200
    assert "Sitemap:" in resp.get_data(as_text=True)
