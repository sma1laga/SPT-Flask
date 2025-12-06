(function () {
  const config = window.LLOYD_MAX_DEMO;
  if (!config) return;

  const slider = document.getElementById('iteration-slider');
  const sliderLabel = document.getElementById('iteration-count');
  const plotEl = document.getElementById('lloyd-plot');
  const levelButtons = document.querySelectorAll('.hero-actions .level-toggle button');

  const uniformImgEl = document.getElementById('uniform-image');
  const lloydImgEl = document.getElementById('lloyd-image');
  const uniformSNR = document.getElementById('uniform-snr');
  const uniformPSNR = document.getElementById('uniform-psnr');
  const lloydSNR = document.getElementById('lloyd-snr');
  const lloydPSNR = document.getElementById('lloyd-psnr');
  const uniformCurveEl = document.getElementById('uniform-curve');
  const lloydCurveEl = document.getElementById('lloyd-curve');
  const uniformLevelLabel = document.getElementById('uniform-level-label');
  const lloydLevelLabel = document.getElementById('lloyd-level-label');
  const uniformLevelButtons = document.querySelectorAll('.quant-header .level-toggle button');

  let histogram = [];
  let history = [];
  let grayPixels = null;
  let imageSize = { width: 0, height: 0 };
  let currentLevels = config.levels || 16;

  function clamp255(v) {
    return Math.max(0, Math.min(255, v));
  }

  function formatDb(value) {
    if (!Number.isFinite(value)) return '∞ dB';
    return `${value.toFixed(2)} dB`;
  }

  function loadGrayImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const { data } = ctx.getImageData(0, 0, img.width, img.height);
        const gray = new Float32Array(img.width * img.height);
        for (let i = 0; i < gray.length; i++) {
          const idx = i * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
        resolve({ gray, width: img.width, height: img.height });
      };
      img.onerror = () => reject(new Error('Could not load Lenna image.'));
      img.src = src;
    });
  }

  function buildHistogram(gray) {
    const hist = new Array(256).fill(0);
    for (let i = 0; i < gray.length; i++) {
      const idx = Math.round(clamp255(gray[i]));
      hist[idx] += 1;
    }
    return hist;
  }

  function computeReconLevels(hist, thresholds, levels) {
    const recon = [];
    for (let i = 0; i < levels; i++) {
      const start = Math.max(0, Math.floor(thresholds[i]));
      const end = Math.min(255, Math.ceil(thresholds[i + 1]) - 1);
      let weightSum = 0;
      let count = 0;
      for (let x = start; x <= end; x++) {
        const freq = hist[x] ?? 0;
        weightSum += freq * x;
        count += freq;
      }
      if (count === 0) {
        recon.push((thresholds[i] + thresholds[i + 1]) / 2);
      } else {
        recon.push(weightSum / count);
      }
    }
    return recon;
  }

  function nextThresholds(recon) {
    const thresholds = [0];
    for (let i = 0; i < recon.length - 1; i++) {
      thresholds.push((recon[i] + recon[i + 1]) / 2);
    }
    thresholds.push(255);
    return thresholds;
  }

  function buildHistory(hist, levels, maxIterations) {
    const initStep = 256 / levels;
    let thresholds = Array.from({ length: levels + 1 }, (_, i) =>
      i === levels ? 255 : i * initStep
    );
    const states = [];
    for (let iter = 0; iter <= maxIterations; iter++) {
      const recon = computeReconLevels(hist, thresholds, levels);
      states.push({ iteration: iter, thresholds: [...thresholds], recon });
      thresholds = nextThresholds(recon);
    }
    return states;
  }

  function syncLevelButtons(level) {
    [...levelButtons, ...uniformLevelButtons].forEach((btn) => {
      const btnLevel = Number(btn.dataset.level);
      btn.classList.toggle('active', btnLevel === level);
    });
  }

  function updateLevelLabels(level) {
    const labelText = `using ${level} levels`;
    uniformLevelLabel.textContent = labelText;
    lloydLevelLabel.textContent = labelText;
  }

  function renderIteration(state) {
    const xVals = [...Array(256).keys()];
    const maxY = Math.max(...histogram, 1);

    const verticals = state.thresholds
      .slice(1, -1)
      .map((t) => ({
        type: 'line',
        x0: t,
        x1: t,
        y0: 0,
        y1: maxY * 1.05,
        line: { color: 'rgba(46, 173, 78, 0.85)', dash: 'dot', width: 2 },
      }));

    const traces = [
      {
        x: xVals,
        y: histogram,
        mode: 'lines',
        name: 'Histogram',
        line: { color: '#4f70c3', width: 2 },
        hovertemplate: 'Grey level %{x}<br>Frequency %{y}<extra></extra>',
      },
      {
        x: state.recon,
        y: state.recon.map(() => -maxY * 0.04),
        mode: 'markers',
        name: 'Reconstruction level',
        marker: { color: '#d1443d', symbol: 'triangle-down', size: 12 },
        hovertemplate: 'Reconstruction %{x:.1f}<extra></extra>',
      },
    ];

    const layout = {
      title: `Iteration of Lloyd-Max quantization (iteration ${state.iteration})`,
      margin: { l: 50, r: 20, t: 40, b: 50 },
      xaxis: { title: 'Grey level', range: [0, 255] },
      yaxis: { title: 'Frequency of occurrence', range: [-maxY * 0.1, maxY * 1.1] },
      showlegend: false,
      shapes: verticals,
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
    };

    Plotly.react(plotEl, traces, layout, { displaylogo: false, responsive: true });
  }

  function quantize(values, thresholds, recon) {
    const out = new Float32Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      let idx = recon.length - 1;
      for (let k = 0; k < recon.length; k++) {
        if (v < thresholds[k + 1] || k === recon.length - 1) {
          idx = k;
          break;
        }
      }
      out[i] = recon[idx];
    }
    return out;
  }

  function toPng(values, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);
    for (let i = 0; i < values.length; i++) {
      const v = clamp255(values[i]);
      const idx = i * 4;
      imgData.data[idx] = v;
      imgData.data[idx + 1] = v;
      imgData.data[idx + 2] = v;
      imgData.data[idx + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL('image/png');
  }

  function computeMetrics(reference, quantized) {
    let signalSum = 0;
    let noiseSum = 0;
    for (let i = 0; i < reference.length; i++) {
      const ref = reference[i];
      const diff = ref - quantized[i];
      signalSum += ref * ref;
      noiseSum += diff * diff;
    }
    const signalP = signalSum / reference.length;
    const noiseP = noiseSum / reference.length;
    const snr = noiseP === 0 ? Infinity : 10 * Math.log10(signalP / noiseP);
    const psnr = noiseP === 0 ? Infinity : 10 * Math.log10((255 * 255) / noiseP);
    return { snr, psnr };
  }

  function buildCurveData(thresholds, recon) {
    const xs = [];
    const ys = [];
    for (let i = 0; i < recon.length; i++) {
      xs.push(thresholds[i]);
      xs.push(thresholds[i + 1]);
      ys.push(recon[i]);
      ys.push(recon[i]);
    }
    return { x: xs, y: ys };
  }

  function renderCurve(el, data) {
    Plotly.react(
      el,
      [
        {
          x: data.x,
          y: data.y,
          mode: 'lines',
          line: { color: '#4f70c3', width: 2, shape: 'hv' },
          hovertemplate: 'Input %{x:.0f} → %{y:.0f}<extra></extra>',
        },
      ],
      {
        margin: { l: 35, r: 10, t: 10, b: 35 },
        xaxis: { title: 'Input grey level', range: [0, 255] },
        yaxis: { title: 'Output level', range: [0, 255] },
        showlegend: false,
        plot_bgcolor: 'transparent',
        paper_bgcolor: 'transparent',
      },
      { displaylogo: false, responsive: true }
    );
  }

  function renderQuantization() {
    const levels = currentLevels;
    const step = 256 / levels;
    const uniformThresholds = Array.from({ length: levels + 1 }, (_, i) =>
      i === levels ? 255 : i * step
    );
    const uniformRecon = Array.from({ length: levels }, (_, i) => i * step + step / 2);

    const finalState = history[history.length - 1];
    const lloydThresholds = finalState.thresholds;
    const lloydRecon = finalState.recon;

    const uniformQuant = quantize(grayPixels, uniformThresholds, uniformRecon);
    const lloydQuant = quantize(grayPixels, lloydThresholds, lloydRecon);

    const uniformMetrics = computeMetrics(grayPixels, uniformQuant);
    const lloydMetrics = computeMetrics(grayPixels, lloydQuant);

    uniformSNR.textContent = formatDb(uniformMetrics.snr);
    uniformPSNR.textContent = formatDb(uniformMetrics.psnr);
    lloydSNR.textContent = formatDb(lloydMetrics.snr);
    lloydPSNR.textContent = formatDb(lloydMetrics.psnr);

    uniformImgEl.src = toPng(uniformQuant, imageSize.width, imageSize.height);
    lloydImgEl.src = toPng(lloydQuant, imageSize.width, imageSize.height);

    renderCurve(uniformCurveEl, buildCurveData(uniformThresholds, uniformRecon));
    renderCurve(lloydCurveEl, buildCurveData(lloydThresholds, lloydRecon));
  }

  function onSliderChange() {
    const idx = Number(slider.value);
    sliderLabel.textContent = idx;
    const state = history[Math.min(idx, history.length - 1)];
    renderIteration(state);
  }

  function setLevels(level) {
    if (level === currentLevels) return;
    currentLevels = level;
    syncLevelButtons(level);
    updateLevelLabels(level);
    history = buildHistory(histogram, currentLevels, config.maxIterations);
    slider.max = config.maxIterations;
    slider.value = Math.min(Number(slider.value), config.maxIterations);
    onSliderChange();
    renderQuantization();
  }

  async function init() {
    try {
      const image = await loadGrayImage(config.imageSrc);
      grayPixels = image.gray;
      imageSize = { width: image.width, height: image.height };
      histogram = buildHistogram(grayPixels);
      history = buildHistory(histogram, currentLevels, config.maxIterations);
      slider.max = config.maxIterations;
      slider.value = Math.min(1, config.maxIterations);
      updateLevelLabels(currentLevels);
      syncLevelButtons(currentLevels);
      onSliderChange();
      slider.addEventListener('input', onSliderChange);
      levelButtons.forEach((btn) =>
        btn.addEventListener('click', () => setLevels(Number(btn.dataset.level)))
      );
      uniformLevelButtons.forEach((btn) =>
        btn.addEventListener('click', () => setLevels(Number(btn.dataset.level)))
      );
      renderQuantization();
    } catch (err) {
      console.error(err);
    }
  }

  init();
})();