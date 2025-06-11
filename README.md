# Signal Processing Toolkit (SPT-Flask)

**Signal Processing Toolkit (SPT-Flask)** is an interactive, browser-based environment for exploring classic and modern DSP concepts.  The project grew from a simple plotting tool into a full learning platform with simulation and training modules.

**GitHub Repository:** [SPT-Flask](https://github.com/sma1laga/SPT-Flask)

---

## 📚 About the Project

The application exposes a collection of blueprints that cover a broad range of signal processing topics.  Students and hobbyists can:

- Plot fundamental signals and observe the results in real time
- Perform convolutions and Fourier analysis—both continuous and discrete
- Model systems using process chains or editable block diagrams
- Design filters and experiment with various modulation schemes

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
- Bode‑plot generator for control system analysis

### Convolution & Filtering
- Convolution calculator and dynamic convolution demo
- Custom filter designer and speech filter input
- Advanced noise reduction module using wavelet processing

### System Modeling
- Process‑chain simulator for chaining basic DSP blocks
- Block‑diagram editor with direct‑form visualiser

### Modulation
- Analog and digital modulation examples

### Discrete Tools
- Discrete‑time plotting and convolution utilities
- FFT and discrete convolution demonstrations

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

## 👤 License
Distributed under the Apache 2.0 License. See `LICENSE` for more information.

---

Feel free to contribute, report issues, and help grow this project!

