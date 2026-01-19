# DEMO_GUIDE.md

## Contributing a new teaching demo

This project includes an interactive **demo library** under `/demos` that is used in *Signals and Systems I & II* lectures and tutorials.  
You can add your own demo (for a lecture, tutorial, or experiment) by following the steps below.

> **Overview**  
> A demo consists of:
> - a **Flask blueprint** (Python) that implements the logic and endpoints  
> - a **Jinja/HTML template** that renders the UI and calls the endpoints  
> - a **blueprint registration** in `main.py`  
> - a **metadata entry** in `pages/demos/data.py` so it appears in the `/demos` menu  

---

## 0. Prerequisites

1. Fork and clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app locally to verify everything works **before** you start:
   ```bash
   python main.py
   ```
4. Open the demos overview in your browser:
   ```text
   http://127.0.0.1:5000/demos
   ```

---

## 1. Create the demo blueprint (Python)

All demo blueprints live under `pages/demos/`.

Create a new file, for example:

```text
pages/demos/my_demo.py
```

Use this as a starting point:

```python
# pages/demos/my_demo.py
from flask import Blueprint, render_template, request, jsonify

my_demo_bp = Blueprint(
    "my_demo",  # blueprint name
    __name__,
    template_folder="../../templates"
)

@my_demo_bp.route("/", methods=["GET"])
def page():
    # Render the main demo page.
    defaults = {"param": 1.0}
    return render_template(
        "demos/my_demo.html",
        defaults=defaults,
    )

@my_demo_bp.route("/compute", methods=["POST"])
def compute():
    """Ajax endpoint: receives parameters, returns JSON for plotting."""
    data = request.get_json(force=True) or {}

    # Extract parameters (with defaults)
    param = float(data.get("param", 1.0))

    # TODO: put your signal processing / control logic here
    # Example: simple y = param * t
    import numpy as np
    t = np.linspace(0.0, 1.0, 200)
    y = param * t

    return jsonify(
        t=t.tolist(),
        y=y.tolist(),
        label=f"y(t) = {param:.2f} · t"
    )
```

**Notes:**

- Use a **unique blueprint name** (here: `"my_demo"`).
- The view that renders the page is conventionally called `page()`.
- The Ajax endpoint is conventionally named `compute()` and returns JSON.
- Set `template_folder="../../templates"` so you can place the HTML under `templates/demos/…`.
- You can also look at existing demos as reference, e.g.:
  - `pages/demos/kapitel2.py`
  - `pages/demos/dtft_impulses.py`
  - `pages/demos/z_trafo.py`
  - etc.

---

## 2. Create the HTML template

All demo templates live under `templates/demos/`.

Create the file:

```text
templates/demos/my_demo.html
```

Example template:

```html
{# templates/demos/my_demo.html #}
{% extends "base.html" %}

{% block content %}
  <div class="page-header" style="margin-bottom: 30px; text-align: center;">
    <h1>My Demo Title</h1>
    <p class="lead">
      Short explanation of what this demo shows and how it connects to the lecture.
    </p>
  </div>

  <!-- Controls -->
  <div style="max-width: 700px; margin: 0 auto; text-align: center;">
    <label for="param"><strong>Parameter:</strong></label>
    <input type="number" id="param" value="{{ defaults.param }}" step="0.1">
    <button id="update-btn" class="btn btn-primary">Update</button>
  </div>

  <!-- Plot -->
  <div style="max-width: 700px; margin: 30px auto;">
    <canvas id="my-demo-plot" height="300"
      style="width: 100%; border: 1px solid #ccd0d5; border-radius: 4px;"></canvas>
  </div>

  <!-- Chart.js (or reuse whatever plotting library is used in other demos) -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const COMPUTE_URL = "{{ url_for('my_demo.compute') }}";
    let chart = null;

    async function fetchData(param) {
      const response = await fetch(COMPUTE_URL, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({param})
      });
      if (!response.ok) {
        throw new Error("Server returned " + response.status);
      }
      return await response.json();
    }

    async function updatePlot() {
      const param = parseFloat(document.getElementById("param").value);
      try {
        const data = await fetchData(param);
        const ctx = document.getElementById("my-demo-plot").getContext("2d");

        if (chart) chart.destroy();
        chart = new Chart(ctx, {
          type: "line",
          data: {
            labels: data.t,
            datasets: [{
              label: data.label || "Signal",
              data: data.y,
              borderWidth: 2,
              fill: false
            }]
          },
          options: {
            responsive: true,
            animation: false
          }
        });
      } catch (err) {
        console.error(err);
        alert("Error: " + err.message);
      }
    }

    document.getElementById("update-btn").addEventListener("click", updatePlot);
    window.addEventListener("load", updatePlot);
  </script>
{% endblock %}
```

**Important:**

- The call `url_for('my_demo.compute')` must match your blueprint name (`"my_demo"`) and the endpoint name (`compute`).
- Follow the basic layout pattern used in other demo templates (header, controls, plot area).
- If you want to match the design of audio/image demos, check existing templates in `templates/demos/` for patterns (audio players, images, etc.).

---

## 3. Register the blueprint in `main.py`

Open `main.py`. Near the bottom you’ll see the **Demos section** where existing blueprints are registered:

```python
# Demos section
# SiSy2 lecture
app.register_blueprint(demos_menu_bp,     url_prefix="/demos")
app.register_blueprint(demos_kapitel2_bp, url_prefix="/demos/kapitel2")
app.register_blueprint(demos_kapitel4_bp, url_prefix="/demos/kapitel4")
# ...
# SiSy1 lecture
app.register_blueprint(demos_exponential_bp, url_prefix="/demos/exponential")
app.register_blueprint(demos_convolution_bp, url_prefix="/demos/convolution")
# ...
```

1. Add your import at the top of `main.py` next to the other demo imports:

   ```python
   from pages.demos.my_demo import my_demo_bp
   ```

2. Register your blueprint:

   ```python
   app.register_blueprint(my_demo_bp, url_prefix="/demos/my-demo")
   ```

> **Convention:**  
> The `url_prefix` should usually be `/demos/<slug>`, where `<slug>` is the same slug you will use in the DEMOS metadata (next step).

After this, your demo is reachable at:

```text
http://127.0.0.1:5000/demos/my-demo
```

(even before it appears in the nice overview menu).

---

## 4. Add metadata so it appears in the `/demos` menu

The pretty overview page `/demos` is driven by the `DEMOS` dictionary in `pages/demos/data.py`.

That file looks roughly like this (excerpt):

```python
# pages/demos/data.py

DEMOS = {
    "Signals and Systems I": {
        "Lecture": [
            {
                "slug": "exponential",
                "title": "Chapter 2-1",
                "title_desc": "Exponential Function",
                "desc": "Visualize x(t)=|X̂| e^{σt} e^{j(ωt+φ)}; full or up to t.",
                "endpoint": "demos_exponential.page",
            },
            {
                "slug": "convolution",
                "title": "Chapter 2-2",
                "title_desc": "Convolution",
                "desc": "Visualize the convolution of two signals.",
                "endpoint": "demos_convolution.page",
            },
            # ...
        ],
        "Tutorial": [],
    },
    "Signals and Systems II": {
        "Lecture": [
            {
                "slug": "kapitel2",
                "title": "Chapter 2",
                "title_desc": "Echo (Convolution)",
                "desc": "Audio input + echo impulse response (A, τ), output via convolution.",
                "endpoint": "demos_kapitel2.page",
            },
            # ...
        ],
        "Tutorial": [
            {
                "slug": "dtft_impulses",
                "title": "DTFT",
                "title_desc": "Discrete Cosine",
                "desc": "Two-window demo: time signal and impulse spectrum at ±ω₀.",
                "endpoint": "dtft_impulses.page",
            },
            # ...
        ],
    },
}
```

Each demo entry is a dictionary with:

- `slug`  
  The short URL segment after `/demos/`.  
  It should match your `url_prefix` in `main.py`.  
  Example: slug `"my-demo"` ↔ `url_prefix="/demos/my-demo"`.

- `title`  
  Usually the chapter number or short label (e.g. `"Chapter 2"`).

- `title_desc`  
  A descriptive subtitle (e.g. `"Echo (Convolution)"`).

- `desc`  
  Longer description of what the demo does (for documentation / future use).

- `endpoint`  
  The Flask endpoint string: `<blueprint_name>.<view_function_name>`.  
  If your blueprint is `my_demo_bp = Blueprint("my_demo", ...)` and your main view is `def page():`, then the endpoint is `"my_demo.page"`.

To add your demo, choose the appropriate course and section (`"Signals and Systems I"` or `"Signals and Systems II"`, `"Lecture"` or `"Tutorial"`) or create your own Section (follow the already created ones). After that, append an entry, for example:

```python
DEMOS["Signals and Systems I"]["Lecture"].append({
    "slug": "my-demo",
    "title": "Chapter X",
    "title_desc": "My custom demo",
    "desc": "Short explanation of what this demo shows and which parameters can be changed.",
    "endpoint": "my_demo.page",
})
```

### How `title` and `title_desc` are used in the menu

In `templates/demos/menu.html`, the menu **swaps** these two fields depending on whether this is a **Lecture** or **Tutorial** demo:

```jinja2
{% set swap_titles = section == 'Lecture' %}
{% for d in demo_list %}
  <h4>
    {% if swap_titles %}
      <strong>{{ d.title_desc }}</strong>    {# Lecture: description is big #}
    {% else %}
      <strong>{{ d.title }}</strong>         {# Tutorial: title is big #}
    {% endif %}
  </h4>
  <p class="demo-subtitle">
    {% if swap_titles %}
      {{ d.title }}                         {# Lecture: chapter as subtitle #}
    {% else %}
      {{ d.title_desc }}                    {# Tutorial: description as subtitle #}
    {% endif %}
  </p>
{% endfor %}
```

So:

- For **Lecture** demos:
  - `title_desc` is the **main heading** (e.g. “Exponential Function”),
  - `title` is the smaller subtitle (e.g. “Chapter 2-1”).

- For **Tutorial** demos:
  - `title` is the **main heading**,
  - `title_desc` is the subtitle.

Pick your texts accordingly.

Once you add your entry, your demo will appear in the `/demos` overview page with the correct title and a “Launch demo” button that uses `endpoint` to build the URL.

---

## 5. Static assets (audio, images, etc.)

If your demo uses audio files or images:

- Place demo-specific assets under `static/demos/`, for example:
  - `static/demos/audio/…`
  - `static/demos/images/…`

In templates, use:

```jinja2
<audio controls src="{{ url_for('static', filename='demos/audio/myfile.wav') }}"></audio>
<img src="{{ url_for('static', filename='demos/images/myimage.png') }}" alt="Description">
```

In Python code, you can derive paths using `current_app.static_folder`. See `pages/demos/kapitel2.py` for a robust pattern:

- `_audio_path` helpers that combine `current_app.static_folder` with a relative path
- utility functions from:
  - `utils.img` (e.g. `fig_to_base64` to embed Matplotlib figures)
  - `utils.audio` (e.g. `wav_data_url` for audio playback in the browser)

Re-use these utilities whenever possible instead of reinventing the wheel.

---

## 6. Testing your demo

Before opening a pull request, please test your demo thoroughly.

1. Start the app:
   ```bash
   python main.py
   ```
2. Visit your demo directly:
   ```text
   http://127.0.0.1:5000/demos/my-demo
   ```
3. Visit the overview page:
   ```text
   http://127.0.0.1:5000/demos
   ```
   Check that:
   - your demo appears under the correct lecture and section,
   - the button correctly launches your demo,
   - the UI responds and updates without JavaScript errors,
   - if you use audio or images, they load correctly.

If you added additional Python dependencies, mention them in your PR and update `requirements.txt` if needed.

---

## 7. Opening a pull request

When you’re happy with your demo:

- Commit the following:
  - your new blueprint file under `pages/demos/`
  - your template under `templates/demos/`
  - changes to `main.py` (blueprint import + registration)
  - changes to `pages/demos/data.py` (new entry in `DEMOS`)
  - any new static assets under `static/demos/...`

- Open a pull request and briefly describe:
  - the goal of the demo (which concept / lecture it supports),
  - how to use it (which parameters can be adjusted),
  - any special dependencies or data files,
  - whether it is intended for **Signals and Systems I** or **II** or something new - and whether it belongs to **Lecture** or **Tutorial** section.

---

## 8. Quick reference (TL;DR)

1. **Create Python blueprint** under `pages/demos/my_demo.py` with:
   - `my_demo_bp = Blueprint("my_demo", ...)`
   - `page()` view for the HTML
   - `compute()` endpoint for Ajax

2. **Create HTML template** under `templates/demos/my_demo.html`:
   - extend `base.html`
   - add controls and plots
   - call `url_for('my_demo.compute')` from JavaScript

3. **Register blueprint** in `main.py`:
   - import `my_demo_bp`
   - `app.register_blueprint(my_demo_bp, url_prefix="/demos/my-demo")`

4. **Add metadata** in `pages/demos/data.py`:
   - append an entry with `slug`, `title`, `title_desc`, `desc`, `endpoint`

5. **Test**:
   - open `/demos/my-demo`
   - open `/demos` and launch it via the menu

6. **Open PR** with a short description of the new demo.

If you follow this guide, your demo will integrate cleanly into the existing lecture structure and demo menu.
