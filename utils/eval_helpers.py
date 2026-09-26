"""Helpers for evaluating user-entered mathematical expressions safely.

Two evaluation paths exist in the app:

* ``safe_eval``: Python ``eval`` of a numeric formula against an explicit name
  table (plot pages).
* ``safe_parse_expr``: SymPy parsing of a symbolic expression (transfer
  functions, polynomials, sequences).  ``sympy.parse_expr`` (and ``sympify``)
  compile their input to Python and ``eval`` it.  The default global namespace
  contains every builtin (``__import__("os").getpid()`` simply runs), and even an
  empty ``global_dict`` does not help because attribute access such as
  ``j.__class__.__mro__[-1].__subclasses__()`` escapes.  The helpers below
  therefore (1) rewrite the user's notation into plain Python, (2) accept only an
  arithmetic syntax tree over whitelisted names, (3) apply the exponent limits
  and (4) evaluate with explicit namespaces and builtins disabled.
"""
import ast
import inspect
import re
import traceback
from typing import Collection, Optional

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations


class UnsafeExpressionError(ValueError):
    """Raised when user input contains intentionally blocked expression patterns"""


def error_data(prefix: str, exc: Exception) -> dict:
    """Return standardized error info with message and position."""
    pos = None
    if isinstance(exc, SyntaxError) and getattr(exc, "offset", None):
        try:
            pos = int(exc.offset) - 1
        except Exception:
            pos = None
    msg = f"{prefix}{exc}" if prefix else str(exc)
    msg += " Make sure to use * when multiplying!"
    return {"error": msg, "pos": pos}

def _contains_power(node: ast.AST) -> bool:
    return any(isinstance(child, ast.BinOp) and isinstance(child.op, ast.Pow) for child in ast.walk(node))


def validate_expression_safety(expression: str) -> None:
    """Reject nested or huge exponents (``10**10**10``, ``s**5000``).

    Only ``**`` is inspected.  Callers that accept ``^`` for exponentiation must
    replace it with ``**`` first (``normalize_expression_text`` does that);
    otherwise ``^`` is parsed as bitwise xor and the checks never fire.
    """
    tree = ast.parse(expression, mode="eval")

    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)):
            continue

        # Blocks expressions like 10**10**10 (and other nested-power forms)
        # that can explode into extremely expensive big-int calculations.
        if _contains_power(node.right):
            raise UnsafeExpressionError(
                "Nested exponentiation is blocked to prevent expensive computations."
            )

        # Bound plain numeric exponents to a practical range.
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
            if abs(node.right.value) > 1000:
                raise UnsafeExpressionError(
                    "Exponent is too large. Please use an absolute value <= 1000."
                )



def safe_eval(expression, allowed_names):
    validate_expression_safety(expression)
    bytecode = compile(expression, "<string>", "eval")

    # check for not explicitly allowed names
    for name in bytecode.co_names:
        if name not in allowed_names:
            raise NameError(f"Use of {name} not allowed!")
    return eval(bytecode, {"__builtins__": {}}, allowed_names)


# ---------------------------------------------------------------------------
# Hardened SymPy parsing
# ---------------------------------------------------------------------------

DEFAULT_INPUT_HELP = (
    "Only numbers, the listed symbols and functions, + - * / ^ and parentheses are allowed."
)
MAX_EXPRESSION_LENGTH = 1000

_ARITHMETIC_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ARITHMETIC_UNARYOPS = (ast.UAdd, ast.USub)
# ``&``, ``|``, ``~`` and comparisons combine SymPy relationals (Piecewise conditions).
_RELATIONAL_BINOPS = (ast.BitAnd, ast.BitOr)
_RELATIONAL_UNARYOPS = (ast.Invert,)
_RELATIONAL_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)

_NUMBER_RE = r'\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?'
_IDENT_RE = r'[A-Za-z_][A-Za-z0-9_]*'


def callable_names(namespace: dict) -> set:
    """Names in ``namespace`` bound to functions or classes, i.e. things a user may call."""
    return {
        name for name, obj in namespace.items()
        if isinstance(obj, type) or inspect.isroutine(obj)
    }


def normalize_expression_text(expr_str: str, functions: Collection[str] = ()) -> str:
    """Turn the user's notation into plain Python syntax.

    ``^`` becomes ``**`` and implicit products are made explicit: ``2s``,
    ``2(s+1)``, ``2pi``, ``2e^(-s)``, ``1e3s``, ``(s+1)(s+2)``, ``(s+1)s``,
    ``s(s+1)``, ``pi(s+1)``.  A name directly followed by ``(`` stays a call only
    when it is listed in ``functions``; every other name is multiplied.
    """
    text = expr_str.strip().replace('^', '**')
    # number followed by a letter or parenthesis: 2s, 2(s+1), 2pi, 1e3s.  e/E are excluded
    # from the lookahead so that the exponent of 1e5 is never split off by backtracking ...
    text = re.sub(rf'({_NUMBER_RE})\s*(?=[A-DF-Za-df-z_(])', r'\1*', text)
    # ... and handled here: a digit followed by e/E that does not start an exponent
    # (2e, 2e**(-s), 2exp(s); 1e5 and 1e-5 are left alone)
    text = re.sub(r'(\d)\s*(?=[eE](?![0-9]|[+-]\d))', r'\1*', text)
    # closing parenthesis followed by a letter, number or parenthesis: (s+1)(s+2), (s+1)s, (s+1).5
    # ("(s+1).real" is left alone so that the AST check reports the attribute access)
    text = re.sub(r'\)\s*(?=[A-Za-z0-9_(]|\.\d)', ')*', text)

    # a name followed by "(" is a product unless the name is a known function: s(s+1), pi(s+1)
    def _name_before_paren(match):
        name = match.group(1)
        return f'{name}(' if name in functions else f'{name}*('

    text = re.sub(rf'(?<![A-Za-z0-9_.])({_IDENT_RE})\s*\(', _name_before_paren, text)
    return text


def check_expression_ast(
    text: str,
    *,
    allowed_names: Optional[Collection[str]],
    allowed_calls: Optional[Collection[str]],
    allow_relational: bool = False,
    help_text: str = DEFAULT_INPUT_HELP,
) -> ast.Expression:
    """Reject anything that is not an arithmetic expression over the allowed names.

    This is the actual defence against code execution through SymPy's parser:
    only ``+ - * / **`` (plus ``& | ~`` and comparisons when ``allow_relational``),
    numeric constants, whitelisted names and calls of whitelisted functions with
    positional arguments survive.  Attribute access, subscripts, strings, keyword
    arguments, lambdas, comprehensions etc. are refused.

    ``allowed_names``/``allowed_calls`` set to ``None`` accept any identifier
    (used for text produced by SymPy's own printer); the structural rules always
    apply.  Returns the parsed tree.
    """
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as exc:
        raise ValueError(
            f"Could not read the expression near position {exc.offset}. {help_text}"
        ) from None
    except (RecursionError, MemoryError, ValueError):
        raise ValueError(f"Could not read the expression. {help_text}") from None

    binops = _ARITHMETIC_BINOPS + (_RELATIONAL_BINOPS if allow_relational else ())
    unaryops = _ARITHMETIC_UNARYOPS + (_RELATIONAL_UNARYOPS if allow_relational else ())
    unsupported = ValueError(f"Unsupported expression element. {help_text}")

    def check(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, binops):
            check(node.left)
            check(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, unaryops):
            check(node.operand)
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                if not allow_relational:
                    raise unsupported
            elif not isinstance(value, (int, float, complex)):
                raise unsupported
        elif isinstance(node, ast.Name):
            if allowed_names is not None and node.id not in allowed_names:
                raise ValueError(f"Unknown symbol '{node.id}'. {help_text}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise unsupported
            if allowed_calls is not None and node.func.id not in allowed_calls:
                raise ValueError(f"Unknown function '{node.func.id}'. {help_text}")
            for arg in node.args:
                check(arg)
        elif allow_relational and isinstance(node, ast.Compare) \
                and all(isinstance(op, _RELATIONAL_CMPOPS) for op in node.ops):
            check(node.left)
            for comparator in node.comparators:
                check(comparator)
        elif allow_relational and isinstance(node, ast.Tuple):
            for element in node.elts:
                check(element)
        else:
            raise unsupported

    try:
        check(tree.body)
    except RecursionError:
        raise ValueError(f"The expression is nested too deeply. {help_text}") from None
    return tree


def sympy_parse_globals() -> dict:
    """Global namespace for ``parse_expr``.

    It holds only the names SymPy's standard transformations generate
    (``Integer(2)``, ``Float('0.5')``, ``2j -> 2*I``, ``Add/Mul/Pow`` and the
    relational classes for ``evaluate=False``) and disables the builtins, which
    ``eval`` would otherwise inject.  ``Symbol`` and ``Function`` are left out on
    purpose: an identifier that is not in the caller's ``local_dict`` fails with a
    NameError instead of silently becoming a symbol.
    """
    return {
        '__builtins__': {},
        'Integer': sp.Integer,
        'Float': sp.Float,
        'Rational': sp.Rational,
        'I': sp.I,
        'Add': sp.Add,
        'Mul': sp.Mul,
        'Pow': sp.Pow,
        'And': sp.And,
        'Or': sp.Or,
        'Not': sp.Not,
        'Lt': sp.Lt,
        'Le': sp.Le,
        'Gt': sp.Gt,
        'Ge': sp.Ge,
        'Eq': sp.Eq,
        'Ne': sp.Ne,
    }


def safe_parse_expr(
    expr_str: str,
    *,
    local_dict: dict,
    evaluate: bool = True,
    allow_relational: bool = False,
    help_text: str = DEFAULT_INPUT_HELP,
    max_length: int = MAX_EXPRESSION_LENGTH,
) -> sp.Basic:
    """Parse a user-entered expression with SymPy without executing Python.

    Only arithmetic over the names in ``local_dict`` is accepted; the functions
    and classes among them may be called.  ``^`` and implicit multiplication
    (``2s``, ``(s+1)(s+2)``) are supported.  Raises ``ValueError`` (or its
    subclass ``UnsafeExpressionError`` for nested/huge exponents) on anything else.
    """
    if not isinstance(expr_str, str):
        raise ValueError(f"Expected an expression. {help_text}")
    if len(expr_str) > max_length:
        raise ValueError(f"Input too long (at most {max_length} characters).")
    functions = callable_names(local_dict)
    text = normalize_expression_text(expr_str, functions)
    if not text:
        raise ValueError(f"The expression is empty. {help_text}")
    check_expression_ast(
        text,
        allowed_names=set(local_dict),
        allowed_calls=functions,
        allow_relational=allow_relational,
        help_text=help_text,
    )
    validate_expression_safety(text)
    return parse_expr(
        text,
        local_dict=dict(local_dict),
        global_dict=sympy_parse_globals(),
        transformations=standard_transformations,
        evaluate=evaluate,
    )


# Names in sympy's public namespace that turn strings into code; never needed to
# re-read printed expressions.
_EVALUATORS = frozenset({
    'S', 'sympify', 'parse_expr', 'lambdify', 'var', 'symbols', 'preview', 'plot',
    'plot3d', 'plot_implicit', 'plot_parametric', 'plot3d_parametric_line',
    'plot3d_parametric_surface', 'plot_backends', 'textplot', 'test', 'doctest',
    'init_printing', 'init_session', 'interactive_traversal',
})
_SYMPY_NAMESPACE = None


def _sympy_namespace() -> dict:
    global _SYMPY_NAMESPACE
    if _SYMPY_NAMESPACE is None:
        namespace = {
            name: getattr(sp, name) for name in sp.__all__ if name not in _EVALUATORS
        }
        # as parse_expr's default namespace does
        namespace['max'] = sp.Max
        namespace['min'] = sp.Min
        namespace['__builtins__'] = {}
        _SYMPY_NAMESPACE = namespace
    return dict(_SYMPY_NAMESPACE)


def parse_sympy_str(text: str) -> sp.Basic:
    """Re-parse text produced by SymPy's own ``str`` printer.

    This is for round-tripping internal expressions (``str(expr)`` with a textual
    substitution applied), **not** for user input: identifiers resolve against
    SymPy's public namespace only, builtins are disabled, and attribute access,
    subscripts, strings and keyword arguments are still refused.
    """
    check_expression_ast(
        text,
        allowed_names=None,
        allowed_calls=None,
        allow_relational=True,
        help_text="The expression could not be re-read.",
    )
    return parse_expr(
        text,
        local_dict={},
        global_dict=_sympy_namespace(),
        transformations=standard_transformations,
    )
