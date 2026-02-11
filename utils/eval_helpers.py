import ast

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


class _SafeExpressionValidator(ast.NodeVisitor):
    """Allow only a small math AST subset for user expressions"""

    _allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Subscript,
        ast.Slice,
        ast.Index,
        ast.keyword,
        # operators
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )

    def __init__(self, allowed_names):
        super().__init__()
        self.allowed_names = set(allowed_names)

    def generic_visit(self, node):
        if not isinstance(node, self._allowed_nodes):
            raise ValueError(f"Expression element '{type(node).__name__}' is not allowed.")
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id not in self.allowed_names:
            raise NameError(f"Use of {node.id} not allowed!")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")
        if node.func.id not in self.allowed_names:
            raise NameError(f"Use of {node.func.id} not allowed!")
        self.generic_visit(node)


def safe_eval(expression, allowed_names):
    parsed = ast.parse(expression, mode="eval")
    _SafeExpressionValidator(allowed_names).visit(parsed)
    bytecode = compile(parsed, "<string>", "eval")
    return eval(bytecode, {"__builtins__": {}}, allowed_names)