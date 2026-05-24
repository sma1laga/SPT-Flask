import traceback
import ast

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