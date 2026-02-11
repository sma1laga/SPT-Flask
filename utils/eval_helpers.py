import traceback


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


def safe_eval(expression, allowed_names):
    bytecode = compile(expression, "<string>", "eval")

    # check for not explicitly allowed names
    for name in bytecode.co_names:
        if name not in allowed_names:
            raise NameError(f"Use of {name} not allowed")
    return eval(bytecode, {"__builtins__": {}}, allowed_names)