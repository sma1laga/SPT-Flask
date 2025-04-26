# pages/process_chain.py
from flask import Blueprint, render_template, request, jsonify
from .chain_blocks import interpret_chain  # <-- must exist
# from .chain_transforms import ... # if you want additional code

process_chain_bp = Blueprint("process_chain", __name__)

@process_chain_bp.route("/", endpoint="show_chain")
def chain_home():
    # Renders process_chain.html, your front-end
    return render_template("process_chain.html")

@process_chain_bp.route("/compute", methods=["POST"])
def compute_chain():
    data = request.get_json(force=True)
    until = data.pop("until", None)       # may be absent
    try:
        plot_data = interpret_chain(data, until_block=until)
        return jsonify({"plot_data": plot_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
