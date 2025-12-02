# main.py
from flask import Flask, render_template, request
from werkzeug.exceptions import HTTPException
import crash_logging
from pages.plot_function import plot_function_bp
from pages.fourier_page import fourier_bp
from pages.convolution import convolution_bp
from pages.autocorrelation import autocorrelation_bp

from pages.process_chain import process_chain_bp
from pages.block_diagram import block_diagram_bp
from pages.direct_plot import direct_plot_bp

from pages.filter_design import filter_design_bp
from pages.image_filter import image_filter_bp
from pages.kernel_animator import bp as kernel_bp
from pages.modulation import mod_bp
from pages.digital_modulation import dig_bp
# analytics

from pages.speech_filter_input import speech_filter_input_bp
from pages.function_definitions import func_defs_bp
from pages.transform_table import transform_table_bp
from pages.theory import theory_bp
from pages.info import info_bp
from pages.bode_plot import bode_plot_bp
from pages.dynamic_convolution import dynamic_convolution_bp
from pages.inverse_z import inverse_z_bp
from pages.inverse_laplace import inverse_laplace_bp

# discrete
from pages.discrete_plot_functions import discrete_plot_functions_bp
from pages.dft_page import discrete_fourier_bp
from pages.discrete_convolution import discrete_convolution_bp
from pages.discrete_dynamic_convolution import discrete_dynamic_convolution_bp
from pages.fft import fft_bp
from pages.discrete_transform_intuition import transform_intuition_bp
from pages.discrete_direct_plot import discrete_direct_plot_bp
from pages.discrete_autocorrelation import discrete_autocorrelation_bp

# Import advanced noise reduction blueprint
from pages.advanced_noise_reduction import advanced_noise_reduction_bp

# Import training blueprints from the training subfolder:
from pages.training.training_convolution import training_convolution_bp
from pages.training.training_fourier import training_fourier_bp
from pages.training.training_processing_chain import training_processing_chain_bp

# Import training exams
from pages.exam_convolution import exam_convolution_bp
from pages.exam_fourier import exam_fourier_bp

# VL DEMOS SISY2
from pages.demos.menu import demos_menu_bp
from pages.demos.data import DEMOS
from pages.demos.kapitel2 import demos_kapitel2_bp
from pages.demos.kapitel4 import demos_kapitel4_bp
from pages.demos.kapitel6 import demos_kapitel6_bp
from pages.demos.kapitel8_2 import demos_kapitel8_2_bp
from pages.demos.kapitel8_audio import demos_kapitel8_audio_bp
from pages.demos.kapitel11 import demos_kapitel11_bp
# UEBUNG DEMOS SISY2
from pages.demos.dtft_impulses import dtft_impulses_bp
from pages.demos.dtft_dft import dtft_dft_bp
from pages.demos.z_trafo import demos_z_trafo_bp
from pages.demos.iir import demos_iir_bp
from pages.demos.filter_demo import demos_filter_bp
#VL DEMOS SISY1
from pages.demos.exponential import demos_exponential_bp
from pages.demos.convolution import demos_convolution_bp
from pages.demos.fouriertransformation import demos_fouriertransformation_bp
from pages.demos.systems_time_audio import demos_systems_time_audio_bp
from pages.demos.bandpass import demos_bandpass_bp
from pages.demos.stability_feedback import stability_feedback_bp
from pages.demos.sampling import sampling_bp
#VL IVC
from pages.demos.compression import demos_compression_bp
from pages.demos.huffman import demos_huffman_bp
from pages.demos.lloyd_max import demos_lloyd_max_bp
from pages.demos.spatial_prediction import demos_spatial_prediction_bp


def _build_demo_slug_map():
    """Create a lookup from demo slug to its parent section name - ist cooler"""

    slug_to_section = {}
    for section_name, categories in DEMOS.items():
        for demo_list in categories.values():
            for demo in demo_list:
                slug_to_section[demo["slug"]] = section_name
    return slug_to_section


DEMO_SLUG_TO_SECTION = _build_demo_slug_map()





def create_app():
    app = Flask(__name__)
    
    @app.context_processor
    def inject_demos_sidebar():
        """Expose demo metadata for building the section-aware demo sidebar."""

        if not request.path.startswith("/demos"):
            return {}

        parts = [part for part in request.path.split("/") if part]
        slug = parts[1] if len(parts) > 1 else None
        section_name = DEMO_SLUG_TO_SECTION.get(slug)
        section_data = DEMOS.get(section_name)

        return {
            "demos_sidebar": DEMOS,
            "demos_section": section_data,
            "demos_section_name": section_name,
        }

    @app.errorhandler(Exception)
    def _handle_exception(e):
        if isinstance(e, HTTPException):
            return e
        crash_logging.log_exception(e)
        return ("Internal Server Error", 500)

    # Register regular pages
    app.register_blueprint(plot_function_bp, url_prefix="/plot_function")
    
    app.register_blueprint(fourier_bp,       url_prefix="/fourier")
    app.register_blueprint(bode_plot_bp, url_prefix="/bode_plot")
    app.register_blueprint(convolution_bp,   url_prefix="/convolution")
    app.register_blueprint(dynamic_convolution_bp, url_prefix="/convolution/dynamic")
    app.register_blueprint(autocorrelation_bp, url_prefix="/autocorrelation")
    app.register_blueprint(mod_bp, url_prefix='/modulation')
    app.register_blueprint(dig_bp, url_prefix='/digital_modulation')

    
    app.register_blueprint(process_chain_bp, url_prefix="/process_chain")
    app.register_blueprint(block_diagram_bp, url_prefix="/block_diagram")
    app.register_blueprint(direct_plot_bp, url_prefix="/direct_plot")
    app.register_blueprint(inverse_z_bp, url_prefix="/inverse_z")
    app.register_blueprint(inverse_laplace_bp, url_prefix="/inverse_laplace")

    
    app.register_blueprint(filter_design_bp, url_prefix="/filter_design")
    app.register_blueprint(func_defs_bp,     url_prefix="/function_definitions")
    app.register_blueprint(speech_filter_input_bp, url_prefix="/filter_design/speech_filter_input")
    app.register_blueprint(image_filter_bp)
    app.register_blueprint(kernel_bp)


    
    app.register_blueprint(transform_table_bp, url_prefix="/transform_table")
    app.register_blueprint(theory_bp,        url_prefix="/theory")
    app.register_blueprint(info_bp, url_prefix='/info')

    # Register discrete part
    app.register_blueprint(discrete_plot_functions_bp, url_prefix='/discrete/plot_functions')
    app.register_blueprint(discrete_fourier_bp, url_prefix="/discrete/dft")        
    app.register_blueprint(discrete_convolution_bp, url_prefix="/discrete/convolution")
    app.register_blueprint(discrete_dynamic_convolution_bp, url_prefix='/discrete/dynamic')
    app.register_blueprint(discrete_autocorrelation_bp, url_prefix='/discrete/autocorrelation')
    app.register_blueprint(fft_bp, url_prefix='/discrete/fft')
    app.register_blueprint(transform_intuition_bp, url_prefix='/discrete/transform_intuition')
    app.register_blueprint(discrete_direct_plot_bp, url_prefix='/discrete/direct_plot')


    # Register training pages (each with its own sub-URL)
    app.register_blueprint(training_convolution_bp, url_prefix="/training/convolution")
    app.register_blueprint(training_fourier_bp,     url_prefix="/training/fourier")
    app.register_blueprint(training_processing_chain_bp, url_prefix="/training/processing_chain")
    
    # Register the advanced noise reduction module.
    app.register_blueprint(advanced_noise_reduction_bp, url_prefix="/advanced_noise_reduction")
    
    # Exams try
    app.secret_key = "some secret"  # needed for session
    app.register_blueprint(exam_convolution_bp, url_prefix="/exam/convolution")
    app.register_blueprint(exam_fourier_bp, url_prefix='/exam/fourier')


    # Demos section
    # SiSy2 lecture
    app.register_blueprint(demos_menu_bp,     url_prefix="/demos")
    app.register_blueprint(demos_kapitel2_bp, url_prefix="/demos/kapitel2")
    app.register_blueprint(demos_kapitel4_bp, url_prefix="/demos/kapitel4")
    app.register_blueprint(demos_kapitel6_bp, url_prefix="/demos/kapitel6")
    app.register_blueprint(demos_kapitel8_2_bp, url_prefix="/demos/kapitel8_2")
    app.register_blueprint(demos_kapitel8_audio_bp, url_prefix="/demos/kapitel8_audio")
    app.register_blueprint(demos_kapitel11_bp, url_prefix="/demos/kapitel11")
    # SiSy2 exercise
    app.register_blueprint(dtft_impulses_bp, url_prefix="/demos/dtft_impulses")
    app.register_blueprint(dtft_dft_bp, url_prefix="/demos/dtft_dft")
    app.register_blueprint(demos_z_trafo_bp, url_prefix="/demos/z_trafo")
    app.register_blueprint(demos_iir_bp, url_prefix="/demos/iir")
    app.register_blueprint(demos_filter_bp, url_prefix="/demos/filter")
    # SiSy1 lecture
    app.register_blueprint(demos_exponential_bp, url_prefix="/demos/exponential")
    app.register_blueprint(demos_convolution_bp, url_prefix="/demos/convolution")
    app.register_blueprint(demos_fouriertransformation_bp, url_prefix="/demos/fouriertransformation")
    app.register_blueprint(demos_systems_time_audio_bp, url_prefix="/demos/systems-time-audio")
    app.register_blueprint(demos_bandpass_bp, url_prefix="/demos/bandpass")
    app.register_blueprint(stability_feedback_bp, url_prefix="/demos/stability-feedback")
    app.register_blueprint(sampling_bp, url_prefix="/demos/sampling")
    # IVC 
    app.register_blueprint(demos_compression_bp, url_prefix="/demos/compression")
    app.register_blueprint(demos_huffman_bp, url_prefix="/demos/huffman")
    app.register_blueprint(demos_lloyd_max_bp, url_prefix="/demos/lloyd-max")
    app.register_blueprint(demos_spatial_prediction_bp, url_prefix="/demos/spatial-prediction-1")

    @app.route("/")
    def home():
        return render_template("home.html")
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
