"""The SymPy-backed parsers must never execute Python.

``sympy.parse_expr`` and ``sympify`` compile their input to Python and eval it
with every builtin in scope, so ``__import__("os").getpid()`` used to run on the
server.  An empty ``global_dict`` does not help either: attribute chains such as
``j.__class__.__mro__[-1].__subclasses__()`` escape.  Every entry point now goes
through ``utils.eval_helpers.safe_parse_expr`` (arithmetic over a whitelist,
builtins disabled), and ``^`` is turned into ``**`` before the exponent limits
run, so ``2^2^2^2`` is caught as well.  These tests pin that down for the helper
itself, for every module-level parser and for every route that takes an expression.
"""

import os

import numpy as np
import pytest
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from main import create_app
from pages import direct_plot, discrete_direct_plot, ztransform_page
from pages.block_diagram import utils as bd_utils
from utils import eval_helpers as eh
from utils import laplace_utils, sympy_utils, z_utils
from utils.eval_helpers import UnsafeExpressionError, parse_sympy_str, safe_parse_expr

CANARY = "SPT_EVAL_CANARY"
PID_PAYLOAD = '__import__("os").getpid()'
# no comma or blank, so that list-splitting parsers see it as one item
CANARY_PAYLOAD = f'__import__("os").environ.update({CANARY}="1")'
DUNDER_PAYLOAD = "j.__class__.__mro__[-1].__subclasses__()"
NESTED_POWER = "2^2^2^2"

# payload, phrase of the error message, exception type shown by the pages
ATTACKS = [
    pytest.param(PID_PAYLOAD, "Unknown symbol", "ValueError", id="getpid"),
    pytest.param(CANARY_PAYLOAD, "Unknown symbol", "ValueError", id="canary"),
    pytest.param(DUNDER_PAYLOAD, "Unsupported expression element", "ValueError", id="dunder-chain"),
    pytest.param(NESTED_POWER, "Nested exponentiation is blocked", "UnsafeExpressionError", id="nested-power"),
]

app = create_app()
app.config["TESTING"] = True


@pytest.fixture(scope="module")
def client():
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def clear_canary(monkeypatch):
    monkeypatch.delenv(CANARY, raising=False)


def _assert_not_executed():
    assert CANARY not in os.environ, "user input was executed as Python"


# --- the helper itself ----------------------------------------------------------------

def test_plain_sympy_parser_executes_python_which_is_why_it_is_wrapped():
    """Documents the hazard and proves the canary payload is a real probe."""
    parse_expr(CANARY_PAYLOAD)
    assert os.environ.pop(CANARY, None) == "1"


S = sp.Symbol("s", complex=True)
LOCALS = {"s": S, "j": sp.I, "pi": sp.pi, "e": sp.E, "sqrt": sp.sqrt, "exp": sp.exp}


@pytest.mark.parametrize("text, expected", [
    ("2s", 2 * S),
    ("2 s", 2 * S),
    ("2(s+1)", 2 * (S + 1)),
    ("(s+1)(s+2)", (S + 1) * (S + 2)),
    ("(s+1) (s+2)", (S + 1) * (S + 2)),
    ("s(s+1)", S * (S + 1)),
    ("2s(s+1)^2", 2 * S * (S + 1) ** 2),
    ("(s+1)^2", (S + 1) ** 2),
    ("s^2 + 2 s + 1", S ** 2 + 2 * S + 1),
    ("2pi", 2 * sp.pi),
    ("pi(s+1)", sp.pi * (S + 1)),
    ("2e^(-s)", 2 * sp.exp(-S)),
    ("2e", 2 * sp.E),
    ("2exp(s)", 2 * sp.exp(S)),
    ("1e3s", 1000.0 * S),
    ("1e-3", sp.Float(1e-3)),
    ("sqrt(2)s", sp.sqrt(2) * S),
    ("2j", 2 * sp.I),
    ("1+2j", 1 + 2 * sp.I),
    (".5s", 0.5 * S),
    ("-s", -S),
])
def test_safe_parse_expr_accepts_engineering_notation(text, expected):
    assert sp.expand(safe_parse_expr(text, local_dict=LOCALS) - expected) == 0


@pytest.mark.parametrize("text", [
    PID_PAYLOAD,
    CANARY_PAYLOAD,
    DUNDER_PAYLOAD,
    "e.__class__",
    "(1).__class__",
    'len("ab")',
    "s.real",
    "s[0]",
    "lambda: 1",
    "[1 for _ in range(3)]",
    "s if s else s",
    "s and s",
    "s == s",
    "s & s",
    "s := 1",
    'f"{s}"',
    '"abc"',
    "True",
    "None",
    "__builtins__",
    "x + 1",
    "sin(s)",
    "Integer(3)",
    'Symbol("x")',
    "sqrt(s, evaluate=False)",
    "5!",
    "",
    "   ",
])
def test_safe_parse_expr_rejects_code_and_unknown_names(text):
    with pytest.raises(ValueError):
        safe_parse_expr(text, local_dict=LOCALS)
    _assert_not_executed()


@pytest.mark.parametrize("text", ["2^2^2^2", "2**2**2", "s^(2^2)", "10^10^10"])
def test_nested_exponents_are_blocked_with_caret_notation(text):
    with pytest.raises(UnsafeExpressionError, match="Nested exponentiation"):
        safe_parse_expr(text, local_dict=LOCALS)


def test_large_exponent_is_blocked():
    with pytest.raises(UnsafeExpressionError, match="too large"):
        safe_parse_expr("s^1001", local_dict=LOCALS)
    assert safe_parse_expr("s^1000", local_dict=LOCALS) == S ** 1000


def test_overlong_input_is_rejected():
    with pytest.raises(ValueError, match="too long"):
        safe_parse_expr("s+" * 600 + "1", local_dict=LOCALS)


def test_parser_namespace_has_no_builtins():
    namespace = eh.sympy_parse_globals()
    assert namespace["__builtins__"] == {}
    with pytest.raises(Exception):
        parse_expr(CANARY_PAYLOAD, local_dict={}, global_dict=namespace)
    _assert_not_executed()


def test_parse_sympy_str_round_trips_printer_output():
    t, k = sp.Symbol("t"), sp.Symbol("k")
    assert parse_sympy_str("2*0 + exp(-t)*Heaviside(t)") == sp.exp(-t) * sp.Heaviside(t)
    assert parse_sympy_str("-1/2 + sqrt(3)*I/2") == sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2
    assert parse_sympy_str("Piecewise((1, k >= 0), (0, True))") == sp.Piecewise((1, k >= 0), (0, True))
    assert parse_sympy_str("zoo") == sp.zoo


@pytest.mark.parametrize("text", [PID_PAYLOAD, CANARY_PAYLOAD, DUNDER_PAYLOAD, 'Symbol("x")', "Float(1, precision=3)"])
def test_parse_sympy_str_rejects_code(text):
    with pytest.raises(ValueError):
        parse_sympy_str(text)
    _assert_not_executed()


# --- every module-level parser ----------------------------------------------------------

N, Z = sp.symbols("n", integer=True), sp.symbols("z", complex=True)

PARSERS = [
    pytest.param(direct_plot._str_to_coeffs, id="direct_plot"),
    pytest.param(discrete_direct_plot._str_to_coeffs, id="discrete_direct_plot"),
    pytest.param(laplace_utils.parse_input, id="laplace_utils.parse_input"),
    pytest.param(lambda text: laplace_utils.parse_input(f"[{text}]"), id="laplace_utils.parse_input-list"),
    pytest.param(z_utils.parse_input, id="z_utils.parse_input"),
    pytest.param(lambda text: z_utils.parse_input(f"[{text}]"), id="z_utils.parse_input-list"),
    pytest.param(bd_utils.parse_poly, id="block_diagram.parse_poly"),
    pytest.param(bd_utils._parse_root_list, id="block_diagram._parse_root_list"),
    pytest.param(lambda text: bd_utils.gain_expr({"type": "TF", "params": {"num": text, "den": "s+1"}}),
                 id="block_diagram.gain_expr-TF"),
    pytest.param(lambda text: bd_utils.gain_expr({"type": "Gain", "params": {"k": text}}),
                 id="block_diagram.gain_expr-Gain"),
    pytest.param(lambda text: bd_utils.gain_expr({"type": "PID", "params": {"kp": text, "ki": 0, "kd": 0}}),
                 id="block_diagram.gain_expr-PID"),
    pytest.param(lambda text: bd_utils.gain_expr({"type": "ZeroPole", "params": {"zeros": text, "poles": "-1"}}),
                 id="block_diagram.gain_expr-ZeroPole"),
    pytest.param(lambda text: bd_utils.gain_expr(
        {"type": "Input", "params": {"kind": "custom", "num": text, "den": "s"}}), id="block_diagram.gain_expr-Source"),
    pytest.param(lambda text: ztransform_page.parse_sequence(text, N, Z), id="ztransform.parse_sequence"),
]


@pytest.mark.parametrize("payload, phrase, _exc", ATTACKS)
@pytest.mark.parametrize("parse", PARSERS)
def test_module_parsers_reject_payloads(parse, payload, phrase, _exc):
    with pytest.raises(ValueError, match=phrase):
        parse(payload)
    _assert_not_executed()


def test_module_parsers_still_accept_engineering_notation():
    assert list(direct_plot._str_to_coeffs("(s+3)(s+1)^2")) == [1, 5, 7, 3]
    assert list(discrete_direct_plot._str_to_coeffs("0.5(z+2)")) == [0.5, 1.0]
    assert sp.expand(laplace_utils.parse_input("2e^(-2)s + pi") - (2 * sp.exp(-2) * S + sp.pi)) == 0
    assert sp.expand(laplace_utils.parse_input("[1, 2j, sin(1)]") - (S ** 2 + 2 * sp.I * S + sp.sin(1))) == 0
    assert sp.expand(z_utils.parse_input("(z-1)(z-0.5)") - (Z - 1) * (Z - 0.5)) == 0
    assert bd_utils.parse_poly("2s^2 + 3") == [2.0, 0.0, 3.0]
    assert bd_utils.parse_poly("(z-1)(z-0.5)") == [1.0, -1.5, 0.5]
    assert bd_utils._parse_root_list("-1+2i, -1-2j") == [-1 + 2j, -1 - 2j]
    assert bd_utils.gain_expr({"type": "Gain", "params": {"k": "2*pi"}}) == 2 * sp.pi
    assert bd_utils.gain_expr({"type": "Gain", "params": {"k": 2.5}}) == 2.5
    seq = ztransform_page.parse_sequence("Piecewise((1, (n>=0)&(n<=5)), (0, True)) + 0.5^n*Heaviside(n)", N, Z)
    assert seq.subs(N, 2) == 1.25


def test_printer_round_trips_keep_working():
    t = sp.Symbol("t", real=True)
    assert sympy_utils.render_number(sp.DiracDelta(0) + 1) == 2
    values = laplace_utils.eval_expression(sp.DiracDelta(t) + sp.exp(-t), np.array([0.0, 1.0]), t)
    assert values == pytest.approx([1.0, np.exp(-1.0)])


# --- every route -------------------------------------------------------------------------

FORM_ROUTES = [
    pytest.param("/direct_plot/", {"denominator": "[1, 1]", "direct_form": "2"}, "message", id="direct_plot"),
    pytest.param("/discrete/direct_plot/", {"denominator": "[1, 1]", "direct_form": "2"}, "message",
                 id="discrete_direct_plot"),
    pytest.param("/inverse_laplace/", {"denominator": "[1, 1]", "response_type": "impulse"}, "message",
                 id="inverse_laplace"),
    # the inverse-z page only shows the exception type
    pytest.param("/inverse_z/", {"denominator": "[1]", "roc_type": "causal"}, "type", id="inverse_z"),
]


@pytest.mark.parametrize("payload, phrase, exc_name", ATTACKS)
@pytest.mark.parametrize("url, extra, shows", FORM_ROUTES)
def test_form_routes_reject_payloads(client, url, extra, shows, payload, phrase, exc_name):
    resp = client.post(url, data={"numerator": payload, **extra})
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert (phrase if shows == "message" else exc_name) in body
    _assert_not_executed()


@pytest.mark.parametrize("payload, phrase, exc_name", ATTACKS)
@pytest.mark.parametrize("url, extra, shows", FORM_ROUTES)
def test_form_routes_reject_payloads_in_denominator(client, url, extra, shows, payload, phrase, exc_name):
    data = {"numerator": "1", **extra, "denominator": payload}
    resp = client.post(url, data=data)
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert (phrase if shows == "message" else exc_name) in body
    _assert_not_executed()


def test_form_routes_still_compute_valid_input(client):
    body = client.post("/direct_plot/", data={"numerator": "(s+3)(s+1)", "denominator": "[1, 6, -10]",
                                              "direct_form": "2"}).get_data(as_text=True)
    assert "Error:" not in body and "H(s)=" in body
    body = client.post("/discrete/direct_plot/", data={"numerator": "(z+3)(z+1)", "denominator": "[1, 6, -10]",
                                                       "direct_form": "2"}).get_data(as_text=True)
    assert "Error:" not in body and "H(z)=" in body
    body = client.post("/inverse_laplace/", data={"numerator": "1", "denominator": "(s+1)^2",
                                                  "response_type": "impulse"}).get_data(as_text=True)
    assert "Error" not in body
    body = client.post("/inverse_z/", data={"numerator": "z", "denominator": "(z-0.5)",
                                            "roc_type": "causal"}).get_data(as_text=True)
    assert "Error" not in body


def _node(node_id, node_type, params):
    return {"id": node_id, "type": node_type, "x": 0, "y": 0, "params": params}


def _chain(*middle, source=None):
    """Input -> middle blocks -> Output, connected in series."""
    nodes = [_node(1, "Input", source or {"kind": "impulse"}), *middle, _node(9, "Output", {})]
    ids = [n["id"] for n in nodes]
    edges = [{"from": a, "to": b, "sign": "+"} for a, b in zip(ids, ids[1:])]
    return {"nodes": nodes, "edges": edges, "domain": "s"}


BLOCK_INJECTIONS = [
    pytest.param(lambda p: _chain(_node(2, "TF", {"num": p, "den": "s+1"})), id="tf-num"),
    pytest.param(lambda p: _chain(_node(2, "TF", {"num": "1", "den": p})), id="tf-den"),
    pytest.param(lambda p: _chain(_node(2, "ZeroPole", {"zeros": p, "poles": "-1", "k": 1})), id="zeropole-zeros"),
    pytest.param(lambda p: _chain(_node(2, "ZeroPole", {"zeros": "", "poles": "-1", "k": p})), id="zeropole-k"),
    pytest.param(lambda p: _chain(_node(2, "Gain", {"k": p})), id="gain-k"),
    pytest.param(lambda p: _chain(_node(2, "PID", {"kp": p, "ki": 0, "kd": 0})), id="pid-kp"),
    pytest.param(lambda p: _chain(_node(2, "TF", {"num": "1", "den": "s+1"}),
                                  source={"kind": "custom", "num": p, "den": "s"}), id="source-custom"),
]


@pytest.mark.parametrize("payload, phrase, _exc", ATTACKS)
@pytest.mark.parametrize("build", BLOCK_INJECTIONS)
def test_block_diagram_compile_rejects_payloads(client, build, payload, phrase, _exc):
    resp = client.post("/block_diagram/compile", json=build(payload))
    assert resp.status_code == 400
    assert phrase in resp.get_json()["error"]
    _assert_not_executed()


def test_block_diagram_compile_still_accepts_engineering_notation(client):
    graph = _chain(_node(2, "TF", {"num": "2(s+1)", "den": "(s+2)(s+3)"}))
    resp = client.post("/block_diagram/compile", json=graph)
    assert resp.status_code == 200, resp.get_data(as_text=True)
    result = resp.get_json()
    assert result["loop_tf"]["num"] == [2.0, 2.0]
    assert result["loop_tf"]["den"] == [1.0, 5.0, 6.0]
