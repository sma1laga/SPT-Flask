# main.py
from flask import Flask, render_template
from pages.plot_function import plot_function_bp
from pages.fourier_page import fourier_bp
from pages.convolution import convolution_bp

from pages.process_chain import process_chain_bp
from pages.block_diagram import block_diagram_bp
from pages.direct_plot import direct_plot_bp

from pages.filter_design import filter_design_bp
from pages.image_filter import image_filter_bp
from pages.kernel_animator import bp as kernel_bp
from pages.modulation import mod_bp
from pages.digital_modulation import dig_bp
# analytics
from analytics import analytics_bp

from pages.speech_filter_input import speech_filter_input_bp
from pages.function_definitions import func_defs_bp
from pages.transform_table import transform_table_bp
from pages.theory import theory_bp
from pages.info import info_bp
from pages.bode_plot import bode_plot_bp
from pages.dynamic_convolution import dynamic_convolution_bp


# discrete
from pages.discrete_plot_functions import discrete_plot_functions_bp
from pages.dft_page import discrete_fourier_bp
from pages.discrete_convolution import discrete_convolution_bp
from pages.discrete_dynamic_convolution import discrete_dynamic_convolution_bp
from pages.fft import fft_bp
from pages.discrete_transform_intuition import transform_intuition_bp
from pages.discrete_direct_plot import discrete_direct_plot_bp




# Import advanced noise reduction blueprint
from pages.advanced_noise_reduction import advanced_noise_reduction_bp



# Import training blueprints from the training subfolder:
from pages.training.training_convolution import training_convolution_bp
from pages.training.training_fourier import training_fourier_bp
from pages.training.training_processing_chain import training_processing_chain_bp

# Import training exams
from pages.exam_convolution import exam_convolution_bp
from pages.exam_fourier import exam_fourier_bp




def create_app():
    app = Flask(__name__)
    app.register_blueprint(analytics_bp)

    # Register regular pages
    app.register_blueprint(plot_function_bp, url_prefix="/plot_function")
    
    app.register_blueprint(fourier_bp,       url_prefix="/fourier")
    app.register_blueprint(bode_plot_bp, url_prefix="/bode_plot")
    app.register_blueprint(convolution_bp,   url_prefix="/convolution")
    app.register_blueprint(dynamic_convolution_bp, url_prefix="/convolution/dynamic")
    app.register_blueprint(mod_bp, url_prefix='/modulation')
    app.register_blueprint(dig_bp, url_prefix='/digital_modulation')

    
    app.register_blueprint(process_chain_bp, url_prefix="/process_chain")
    app.register_blueprint(block_diagram_bp, url_prefix="/block_diagram")  
    app.register_blueprint(direct_plot_bp, url_prefix="/direct_plot")
    
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

    


    @app.route("/")
    def home():
        return render_template("home.html")
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
