# main.py
from flask import Flask, render_template
from pages.plot_function import plot_function_bp
from pages.fourier_page import fourier_bp
from pages.convolution import convolution_bp
from pages.process_chain import process_chain_bp
from pages.filter_design import filter_design_bp
from pages.function_definitions import func_defs_bp
from pages.transform_table import transform_table_bp
from pages.theory import theory_bp

# Import training blueprints from the training subfolder:
from pages.training.training_convolution import training_convolution_bp
from pages.training.training_fourier import training_fourier_bp
from pages.training.training_processing_chain import training_processing_chain_bp

def create_app():
    app = Flask(__name__)

    # Register regular pages
    app.register_blueprint(plot_function_bp, url_prefix="/plot_function")
    app.register_blueprint(fourier_bp,       url_prefix="/fourier")
    app.register_blueprint(convolution_bp,   url_prefix="/convolution")
    app.register_blueprint(process_chain_bp, url_prefix="/process_chain")
    app.register_blueprint(filter_design_bp, url_prefix="/filter_design")
    app.register_blueprint(func_defs_bp,     url_prefix="/function_definitions")
    app.register_blueprint(transform_table_bp, url_prefix="/transform_table")
    app.register_blueprint(theory_bp,        url_prefix="/theory")

    # Register training pages (each with its own sub-URL)
    app.register_blueprint(training_convolution_bp, url_prefix="/training/convolution")
    app.register_blueprint(training_fourier_bp,     url_prefix="/training/fourier")
    app.register_blueprint(training_processing_chain_bp, url_prefix="/training/processing_chain")

    @app.route("/")
    def home():
        return render_template("home.html")
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
