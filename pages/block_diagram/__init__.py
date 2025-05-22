# pages/block_diagram/__init__.py
from flask import Blueprint

block_diagram_bp = Blueprint("block_diagram", __name__)

# keep this at the bottom to avoid circular imports
from . import routes
