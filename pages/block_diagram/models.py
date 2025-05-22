"""
JSON-Schema definitions – keep validation concerns out of routes.py
Install jsonschema  →  pip install jsonschema
"""
from jsonschema import validate

NODE_SCHEMA = {
    "type": "object",
    "required": ["id", "type", "x", "y"],
    "properties": {
        "id": {"type": "integer"},
        "type": {"type": "string"},
        "params": {"type": "object"},
        "x": {"type": "number"},
        "y": {"type": "number"},
    },
}

GRAPH_SCHEMA = {
    "type": "object",
    "required": ["nodes", "edges"],
    "properties": {
        "nodes": {"type": "array", "items": NODE_SCHEMA},
        "edges": {"type": "array"},
        "domain": {"type": "string", "enum": ["s", "z"]},
    },
}


def validate_graph(data: dict):
    validate(instance=data, schema=GRAPH_SCHEMA)
