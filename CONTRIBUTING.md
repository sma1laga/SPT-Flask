# Contributing to SPT-Flask

Thank you for considering a contribution to the Signal Processing Toolkit. This guide describes the steps needed to get your environment ready and the conventions we follow.

## Getting Started

1. **Fork the repository** on GitHub and clone your fork locally.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On windows use: venv\Scripts\activate
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the test suite** to ensure the project works on your system:
   ```bash
   pytest -q
   ```
5. **Start the application** if you want to experiment locally:
   ```bash
   python main.py
   ```
   Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Making Changes

- Create a feature branch for your work:
  ```bash
  git checkout -b feature/your-feature
  ```
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style. Use four spaces for indentation and keep lines under 100 characters to match the existing codebase.
- Add unit tests for new functionality and make sure all tests pass with `pytest -q`.
- If you plan a large change, open an issue first so we can discuss your approach.

## Opening a Pull Request

1. Commit your changes with clear messages:
   ```bash
   git commit -m "Describe your change"
   ```
2. Push your branch and open a pull request against the `main` branch.
3. Describe the motivation for your changes and reference any related issues.

## Code of Conduct

Please be respectful and collaborative. We welcome contributions from everyone regardless of background or experience.

---

We appreciate your help in improving SPT-Flask!