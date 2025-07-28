# Signal Processing Toolkit

**Signal Processing Toolkit (SPT-Flask)** is an interactive, browser-based environment for exploring classic and modern DSP concepts.  The project grew from a simple plotting tool into a full learning platform with simulation and training modules.

**GitHub Repository:** [SPT-Flask](https://github.com/sma1laga/SPT-Flask)

---

## 📚 About the Project

The application is organised as a set of small Flask blueprints, each covering a
particular DSP topic. Students and hobbyists can:

- Plot and analyse continuous or discrete signals in real time
- Perform convolution, autocorrelation and Fourier analysis
- Visualise Bode plots, inverse transforms and kernel animations
- Model systems with process chains or editable block diagrams
- Design filters with speech and image demos and experiment with modulation
- Practise using the built-in training pages and exam modules

The goal is a **practical, hands-on toolkit** that supports university courses and self-study alike.



## 🔧 Built With

- [Python 3](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyWavelets](https://pywavelets.readthedocs.io/)
- [control](https://python-control.readthedocs.io/) (for Bode plots)
- [Bootstrap](https://getbootstrap.com/) – responsive UI with optional dark mode

## ✨ Features

### Plotting & Analysis
- Time‑domain function plotter with interactive controls
- Fourier Transform calculators (including FFT and DFT viewers)
- Autocorrelation visualiser and kernel animator
- Bode‑plot generator for control system analysis
- Inverse Laplace and Z‑transform calculators
- Image filter preview and transform table reference

### Convolution & Filtering
- Convolution calculator with dynamic and discrete variants
- Custom filter designer with speech and image examples
- Advanced noise reduction module using wavelet processing

### System Modeling
- Process‑chain simulator for chaining basic DSP blocks
- Block‑diagram editor and direct‑form visualiser

### Modulation
- Analog and digital modulation examples

### Discrete Tools
- Discrete‑time plotting, convolution and autocorrelation utilities
- FFT, DFT and discrete dynamic convolution demos
- Z‑transform viewer and transform intuition page

### Learning Aids
- Training modules for convolution, Fourier transforms and processing chains
- Small exam modules for practice sessions

All pages include downloadable plots/data, and the interface offers an optional dark mode.


## 🔄 Installation

### 1. Clone the repository
```bash
git clone https://github.com/sma1laga/SPT-Flask.git
cd SPT-Flask
```

### 2. Set up a Python virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install the required packages
```bash
pip install -r requirements.txt
```

### 4. Run the Flask application
```bash
python main.py
```

### 5. Open the application
Navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

## 🧪 Running Tests
Install dependencies and run `pytest`. The included GitHub Actions workflow
executes the test suite automatically on every push and pull request.

```bash
pip install -r requirements.txt
pytest -q
```

## 🚀 Contributing
Pull requests are welcome! If you want to propose a feature or fix a bug:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/yourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/yourFeature`)
5. Open a Pull Request

Everyone who contributes to this repository will be mentioned in the Hall of Fame on our website.

## 👤 License
Distributed under the Apache 2.0 License. See `LICENSE` for more information.

---

Feel free to contribute, report issues, and help grow this project!

