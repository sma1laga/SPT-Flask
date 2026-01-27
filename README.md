# Signal Processing Toolkit (SPT‑Flask)

**SPT‑Flask** is a modern, browser‑based environment for learning and experimenting with digital and analogue signal‑processing concepts. It was created at the Lehrstuhl für Multimediakommunikation und Signalverarbeitung (LMS), FAU Erlangen‑Nürnberg and is provided licence‑free for educational and research use. The current hosted instance is available at https://lms-spt.e-technik.uni-erlangen.de/.

## Project goals

* Make complex DSP topics approachable through interactive visualisation.
* Provide a one‑stop toolkit for plotting, transforms, convolution, filtering and modulation.
* Support discrete‑ and continuous‑time analysis within a unified interface.
* Offer training modules and exam‑style quizzes for self‑assessment.
* Encourage community contributions and recognise those who improve the toolkit.

## Features

### Plotting & analysis

* Time‑domain function plotter with adjustable parameters.
* Continuous and discrete Fourier transform calculators (including FFT and DFT viewers).
* Autocorrelation visualiser and kernel animator.
* Bode‑plot generator for continuous‑time systems.
* Inverse Laplace and Z‑transform calculators.
* Image spectrum viewer and transform‑table reference.

### Convolution & filtering

* Continuous, discrete and dynamic convolution calculators.
* Custom filter designer with downloadable impulse responses.
* Speech filter design and spectrogram analysis.
* Image‑filter demo supporting low‑pass/high‑pass/band‑pass Butterworth filters.
* Advanced noise‑reduction module using wavelet processing.

### System modelling

* Process‑chain simulator that lets you chain basic DSP blocks and view intermediate outputs.
* Block‑diagram editor and direct‑form visualiser.
* Kernel animator to illustrate impulse responses.

### Modulation

* Analog modulation examples (AM, FM).
* Digital modulation demos (ASK, PSK, FSK) with adjustable carrier frequency, bit rate and deviation.

### Discrete‑time tools

* Plot discrete sequences and explore Z‑transform intuition.
* Perform discrete convolution and autocorrelation.
* FFT/DFT and discrete dynamic convolution demos.

### Learning aids

* Training modules for convolution, Fourier transforms and process‑chain design.
* Small exam modules with timed questions and feedback.
* Downloadable plots and data for offline study.
* All pages support dark mode and colour‑blind mode.

### Analytics & stability

* A lightweight analytics component collects anonymous usage statistics (path, user agent, country) to improve the service.
* Crash logging records unhandled exceptions for debugging; no personal data are stored.

## 📦 Module Availability

> ⚙️ *Modules marked “Under verification” are implemented and currently being validated before public release.*

## 📦 Module Availability Overview

| Category       | Module                         | Path                            | Description                                            | Status             |
| -------------- | ------------------------------ | ------------------------------- | ------------------------------------------------------ | --------------------- |
| **Continuous** | Plot Function                  | `/plot_function`                | Function Plotter                                       | Available          |
|                | Convolution                    | `/convolution`                  | Time-domain convolution explorer                       | Available          |
|                | Dynamic Convolution            | `/convolution/dynamic`          | Time-domain convolution explorer (demo)                | Available          |
|                | Autocorrelation                | `/autocorrelation`              | Signal self-correlation toolkit                        | Available          |
|                | Inverse Laplace                | `/inverse_laplace`              | Transform inversion practice                           | Available          |
|                | Fourier Analysis               | `/fourier`                      | Core Fourier-transform plotting module                 | Under verification |
|                | Bode Plot                      | `/bode_plot`                    | Amplitude/phase visualization for transfer functions   | Under verification |
|                | Modulation                     | `/modulation`                   | Analog modulation plotter and calculator               | Under verification |
|                | Digital Modulation             | `/digital_modulation`           | Digital modulation plotter and calculator              | Under verification |
|                | Process Chain                  | `/process_chain`                | Cascade-processing and block-model builder             | Under verification |
|                | Block Diagram                  | `/block_diagram`                | Cascade-processing and block-model builder             | Under verification |
|                | Direct Plot                    | `/direct_plot`                  | Direct form (I, II, III) plotting utility              | Under verification |
|                | Filter Design                  | `/filter_design`                | Filter synthesis and speech-specific input module      | Under verification |
|                | Image Filter / Kernel Animator | `/image_filter` / —             | Spatial-filter experimentation tools                   | Under verification |
|                | Advanced Noise Reduction       | `/advanced_noise_reduction`     | Denoising methods showcase                             | Under verification |
| **Discrete**   | Plot Functions                 | `/discrete/plot_functions`      | Discrete-time signal visualization                     | Available          |
|                | Fourier / DFT                  | `/discrete/dft`                 | Spectrum analysis tools                                | Available          |
|                | Convolution                    | `/discrete/convolution`         | Discrete convolution                                   | Available          |
|                | Dynamic Discrete Convolution   | `/discrete/dynamic`             | Discrete convolution (interactive)                     | Available          |
|                | Discrete Autocorrelation       | `/discrete/autocorrelation`     | Correlation for sampled signals                        | Available          |
|                | Transform Intuition            | `/discrete/transform_intuition` | Conceptual guide to discrete transforms                | Available          |
|                | Inverse Z-Transform            | `/inverse_z`                    | Transform inversion practice                           | Available          |
|                | Discrete Direct Plot           | `/discrete/direct_plot`         | Direct form (I, II, III) plotting for discrete systems | Under verification |
| **Training**   | Convolution Training           | `/training/convolution`         | Practice generator and quizzes                         | Available          |
|                | Fourier Training               | `/training/fourier`             | Fourier Transform-focused training set                 | Under verification |
|                | Processing Chain Training      | `/training/processing_chain`    | Multistage system training module                      | Under verification |

## Installation

```bash
git clone https://github.com/sma1laga/SPT-Flask.git
cd SPT-Flask
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Then open your browser to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## Running Tests

Install dependencies and run `pytest`. The included GitHub Actions workflow
executes the test suite automatically on every push and pull request.

```bash
pip install -r requirements.txt
pytest -q
```

## Contributing

Pull requests are welcome! If you want to propose a feature or fix a bug:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/yourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/yourFeature`)
5. Open a Pull Request

> **Everyone who contributes to this repository will be mentioned in the Hall of Fame on our website.**

## 👤 License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

---

Feel free to contribute, report issues, and help grow this project!
