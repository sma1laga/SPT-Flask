(() => {

  const bodeData = window.bodeData || null;
  const pzData = window.pzData || null;
  const nyquistData = window.nyquistData || null;
  const bodeMeta = window.bodeMeta || {};


  const basePlotConfig = { responsive: true, displaylogo: false };
  let showCornerFrequencyMarkers = true;
  let currentPlotMode = 'exact';


  const isFiniteNumber = value => typeof value === 'number' && Number.isFinite(value);

  function formatNumber(value, digits = 3) {
    if (!isFiniteNumber(value)) return '—';
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) {
      return Number(value).toExponential(2);
    }
    const fixed = Number(value).toFixed(digits);
    if (fixed.includes('.')) {
      let trimmed = fixed.replace(/0+$/, '');
      if (trimmed.endsWith('.')) {
        trimmed = trimmed.slice(0, -1);
      }
      return trimmed;
    }
    return fixed;
  }

  function formatWithUnit(value, unit, digits = 3) {
    const formatted = formatNumber(value, digits);
    return formatted === '—' ? '—' : `${formatted} ${unit}`;
  }
  function formatFrequency(value) {
    if (!isFiniteNumber(value)) return '—';
    return `${formatNumber(value, 3)} rad/s`;
  }


  function formatMargin(value, freq, valueUnit, freqUnit) {
    const main = formatWithUnit(value, valueUnit, 2);
    const freqText = formatWithUnit(freq, freqUnit, 3);
    if (main === '—') return '—';
    return freqText === '—' ? main : `${main} @ ${freqText}`;
  }

  function renderMetrics(data) {
    const metricsBox = document.getElementById('bodeMetrics');
    if (!metricsBox) return;

    const gmText = formatMargin(data.gain_margin_db, data.gain_cross_freq, 'dB', 'rad/s');
    const pmText = formatMargin(data.phase_margin_deg, data.phase_cross_freq, '°', 'rad/s');
    const bwText = formatWithUnit(data.bandwidth, 'rad/s', 3);

    metricsBox.innerHTML = `
      <span><strong>Gain margin</strong>${gmText}</span>
      <span><strong>Phase margin</strong>${pmText}</span>
      <span><strong>Bandwidth</strong>${bwText}</span>
    `;
  }

  const CROSSOVER_MARGIN_DECADES = 1;
  const MAX_CROSSOVER_SPREAD_DECADES = 8;

  function getFrequencyRangeFromData(data) {
    if (!data || !Array.isArray(data.omega)) return null;
    const valid = data.omega
      .map(value => Number(value))
      .filter(value => Number.isFinite(value) && value > 0);
    if (!valid.length) return null;
    return {
      min: Math.min(...valid),
      max: Math.max(...valid)
    };
  }
  function isCrossoverWithinRange(freq, range) {
    if (!isFiniteNumber(freq) || freq <= 0) return false;
    if (!range || !(range.min > 0 && range.max > 0)) return true;
    const minLimit = range.min / Math.pow(10, CROSSOVER_MARGIN_DECADES);
    const maxLimit = range.max * Math.pow(10, CROSSOVER_MARGIN_DECADES);
    return freq >= minLimit && freq <= maxLimit;
  }

  function areCrossoversTooFar(freqA, freqB) {
    if (!isFiniteNumber(freqA) || !isFiniteNumber(freqB) || freqA <= 0 || freqB <= 0) {
      return false;
    }
    const distance = Math.abs(Math.log10(freqA) - Math.log10(freqB));
    return distance > MAX_CROSSOVER_SPREAD_DECADES;
  }

  function getDistanceFromRangeMidpoint(freq, range) {
    if (!isFiniteNumber(freq) || freq <= 0) return Number.POSITIVE_INFINITY;
    if (!range || !(range.min > 0 && range.max > 0)) {
      return Math.abs(Math.log10(freq));
    }
    const midLog = (Math.log10(range.min) + Math.log10(range.max)) / 2;
    return Math.abs(Math.log10(freq) - midLog);
  }

  function buildCrossingLines(data) {
    const shapes = [];
    const freqRange = getFrequencyRangeFromData(data);
    let showPhase = isCrossoverWithinRange(data.phase_cross_freq, freqRange);
    let showGain = isCrossoverWithinRange(data.gain_cross_freq, freqRange);

    if (showPhase && showGain && areCrossoversTooFar(data.phase_cross_freq, data.gain_cross_freq)) {
      const phaseDistance = getDistanceFromRangeMidpoint(data.phase_cross_freq, freqRange);
      const gainDistance = getDistanceFromRangeMidpoint(data.gain_cross_freq, freqRange);
      if (phaseDistance <= gainDistance) {
        showGain = false;
      } else {
        showPhase = false;
      }
    }

    if (showPhase) {
      shapes.push({
        type: 'line',
        x0: data.phase_cross_freq,
        x1: data.phase_cross_freq,
        y0: 0,
        y1: 1,
        xref: 'x',
        yref: 'paper',
        line: { color: '#ef4444', dash: 'dash', width: 2 }
      });
    }
    if (showGain) {
      shapes.push({
        type: 'line',
        x0: data.gain_cross_freq,
        x1: data.gain_cross_freq,
        y0: 0,
        y1: 1,
        xref: 'x',
        yref: 'paper',
        line: { color: '#f97316', dash: 'dash', width: 2 }
      });
    }
    return shapes;
  }
  function getCornerFrequencies(data) {
    if (!data) return [];
    const freqs = Array.isArray(data.corner_frequencies) ? data.corner_frequencies : [];
    return freqs
      .map(value => Number(value))
      .filter(freq => Number.isFinite(freq) && freq > 0);
  }

  function hasCornerFrequencies(data) {
    return getCornerFrequencies(data).length > 0;
  }

  function buildCornerFrequencyShapes(data) {
    if (!showCornerFrequencyMarkers) return [];
    return getCornerFrequencies(data).map(freq => ({
      type: 'line',
      x0: freq,
      x1: freq,
      y0: 0,
      y1: 1,
      xref: 'x',
      yref: 'paper',
      line: { color: '#a855f7', dash: 'dot', width: 1.5 }
    }));
  }

  function buildBodeShapes(data) {
    const shapes = buildCrossingLines(data);
    return shapes.concat(buildCornerFrequencyShapes(data));
  }
  function getMagnitudeSeries(data) {
    const exact = Array.isArray(data.magnitude_db) ? data.magnitude_db : [];
    const straight = Array.isArray(data.magnitude_straight_db) ? data.magnitude_straight_db : [];
    if (currentPlotMode === 'straight' && straight.length === exact.length && straight.length > 0) {
      return {
        values: straight,
        name: 'Straight-line magnitude approximation',
        line: { color: '#2563eb', width: 3, dash: 'dash' }
      };
    }
    return {
      values: exact,
      name: 'Exact magnitude (dB)',
      line: { color: '#2563eb', width: 3 }
    };
  }
  function getPhaseSeries(data) {
    const exact = Array.isArray(data.phase_deg) ? data.phase_deg : [];
    const straight = Array.isArray(data.phase_straight_deg) ? data.phase_straight_deg : [];
    if (currentPlotMode === 'straight' && straight.length === exact.length && straight.length > 0) {
      return {
        values: straight,
        name: 'Straight-line phase approximation',
        line: { color: '#16a34a', width: 3, dash: 'dash' }
      };
    }
    return {
      values: exact,
      name: 'Exact phase (°)',
      line: { color: '#16a34a', width: 3 }
    };
  }

  function renderBodeMagnitude(data) {
    if (typeof window.Plotly === 'undefined') return;
    const el = document.getElementById('bodeMagnitudePlot');
    if (!el) return;

    const freq = Array.isArray(data.omega) ? data.omega : [];
    const magnitudeSeries = getMagnitudeSeries(data);

    const layout = {
      margin: { l: 70, r: 20, t: 10, b: 40 },
      hovermode: 'x unified',
      shapes: buildBodeShapes(data),
      xaxis: {
        type: 'log',
        title: 'Frequency (rad/s)',
        showgrid: true,
        gridcolor: '#e5e7eb',
        showexponent: 'all',
        exponentformat: 'power'
      },
      yaxis: {
        title: 'Magnitude (dB)',
        showgrid: true,
        gridcolor: '#e5e7eb'
      },
      showlegend: false
    };

    Plotly.react(
      el,
      [
        {
          x: freq,
          y: magnitudeSeries.values,
          type: 'scatter',
          mode: 'lines',
          name: magnitudeSeries.name,
          line: magnitudeSeries.line
        }
      ],
      layout,
      basePlotConfig
    );
  }

  function renderBodePhase(data) {
    if (typeof window.Plotly === 'undefined') return;
    const el = document.getElementById('bodePhasePlot');
    if (!el) return;

    const freq = Array.isArray(data.omega) ? data.omega : [];
    const phaseSeries = getPhaseSeries(data);

    const layout = {
      margin: { l: 70, r: 20, t: 10, b: 40 },
      hovermode: 'x unified',
      shapes: buildBodeShapes(data),
      xaxis: {
        type: 'log',
        title: 'Frequency (rad/s)',
        showgrid: true,
        gridcolor: '#e5e7eb',
        showexponent: 'all',
        exponentformat: 'power'
      },
      yaxis: {
        title: 'Phase (°)',
        showgrid: true,
        gridcolor: '#e5e7eb'
      },
      showlegend: false
    };

    Plotly.react(
      el,
      [
        {
          x: freq,
          y: phaseSeries.values,
          type: 'scatter',
          mode: 'lines',
          name: phaseSeries.name,
          line: phaseSeries.line
        }
      ],
      layout,
      basePlotConfig
    );
  }

  function renderBodePlot(data) {
    renderBodeMagnitude(data);
    renderBodePhase(data);
  }

  function getTransferFunctionExportText() {
    const numerator = typeof bodeMeta.numerator === 'string' ? bodeMeta.numerator.trim() : '';
    const denominator = typeof bodeMeta.denominator === 'string' ? bodeMeta.denominator.trim() : '';
    if (numerator && denominator) {
      return `H(s) = ${numerator} / ${denominator}`;
    }
    if (typeof bodeMeta.functionLatex === 'string' && bodeMeta.functionLatex.trim()) {
      return bodeMeta.functionLatex.replace(/\\/g, '');
    }
    return 'H(s)';
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = src;
    });
  }

  function cloneForExport(value) {
    return JSON.parse(JSON.stringify(value || {}));
  }

  function createExportLayout(sourceLayout, titleText) {
    const layout = cloneForExport(sourceLayout);
    const xaxis = layout.xaxis || {};
    const yaxis = layout.yaxis || {};
    return {
      ...layout,
      title: {
        text: titleText,
        x: 0.02,
        xanchor: 'left',
        font: { size: 24, color: '#111827' }
      },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      margin: { l: 110, r: 40, t: 84, b: 80 },
      font: { family: 'Arial, sans-serif', size: 18, color: '#111827' },
      xaxis: {
        ...xaxis,
        title: { text: xaxis.title?.text || xaxis.title || 'Frequency (rad/s)', font: { size: 20 } },
        tickfont: { size: 16 },
        gridcolor: '#d1d5db',
        zerolinecolor: '#9ca3af'
      },
      yaxis: {
        ...yaxis,
        title: { text: yaxis.title?.text || yaxis.title || '', font: { size: 20 } },
        tickfont: { size: 16 },
        gridcolor: '#d1d5db',
        zerolinecolor: '#9ca3af'
      }
    };
  }

  async function createPlotImage(plotId, titleText) {
    const plot = document.getElementById(plotId);
    if (!plot || typeof Plotly === 'undefined') {
      throw new Error(`Plot ${plotId} is not available for export.`);
    }
    const exportLayout = createExportLayout(plot.layout || {}, titleText);
    const exportConfig = {
      ...basePlotConfig,
      responsive: false
    };
    const data = Array.isArray(plot.data) ? plot.data.map(trace => cloneForExport(trace)) : [];
    const container = document.createElement('div');
    container.style.position = 'fixed';
    container.style.left = '-99999px';
    container.style.top = '0';
    container.style.width = '1600px';
    container.style.height = '520px';
    document.body.appendChild(container);

    try {
      await Plotly.newPlot(container, data, exportLayout, exportConfig);
      const dataUrl = await Plotly.toImage(container, {
        format: 'png',
        width: 1600,
        height: 520,
        scale: 2
      });
      return await loadImage(dataUrl);
    } finally {
      if (typeof Plotly.purge === 'function') {
        Plotly.purge(container);
      }
      container.remove();
    }
  }

  async function exportBodeComposite() {
    const [magnitudeImage, phaseImage] = await Promise.all([
      createPlotImage('bodeMagnitudePlot', 'Magnitude Plot'),
      createPlotImage('bodePhasePlot', 'Phase Plot')
    ]);

    const canvas = document.createElement('canvas');
    canvas.width = 1800;
    canvas.height = 1500;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Unable to prepare PNG export canvas.');
    }

    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#111827';
    ctx.font = 'bold 42px Arial, sans-serif';
    ctx.fillText('Bode Diagram', 90, 90);

    ctx.font = '26px Arial, sans-serif';
    ctx.fillStyle = '#374151';
    ctx.fillText(getTransferFunctionExportText(), 90, 138);

    const modeLabel = currentPlotMode === 'straight' ? 'Straight-line approximation' : 'Exact response';
    const markersLabel = showCornerFrequencyMarkers ? 'Corner frequencies shown' : 'Corner frequencies hidden';
    ctx.font = '22px Arial, sans-serif';
    ctx.fillStyle = '#4b5563';
    ctx.fillText(`${modeLabel} • ${markersLabel}`, 90, 182);

    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#dbeafe';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(70, 220, 1660, 560, 22);
    ctx.fill();
    ctx.stroke();
    ctx.drawImage(magnitudeImage, 100, 242, 1600, 520);

    ctx.beginPath();
    ctx.roundRect(70, 830, 1660, 560, 22);
    ctx.fill();
    ctx.stroke();
    ctx.drawImage(phaseImage, 100, 852, 1600, 520);

    const dataUrl = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = 'bode_plot.png';
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  function setupBodeDownload() {
    const button = document.getElementById('bodeDownloadButton');
    if (!button) return;
    button.addEventListener('click', async () => {
      button.disabled = true;
      const originalLabel = button.textContent;
      button.textContent = 'Preparing PNG...';
      try {
        await exportBodeComposite();
      } catch (error) {
        console.error('Falling back to server-side Bode PNG export:', error);
        const fallbackUrl = button.dataset.fallbackUrl;
        if (fallbackUrl) {
          window.location.href = fallbackUrl;
        }
      } finally {
        button.disabled = false;
        button.textContent = originalLabel;
      }
    });
  }

  function setupCornerFrequencyToggle(data) {
    const toggle = document.getElementById('cornerFrequencyToggle');
    if (!toggle) return;
    const label = toggle.closest('.bode-toggle');
    const hasCorners = hasCornerFrequencies(data);
    if (!hasCorners) {
      toggle.checked = false;
      toggle.disabled = true;
      if (label) label.classList.add('bode-toggle--disabled');
      showCornerFrequencyMarkers = false;
      return;
    }
    showCornerFrequencyMarkers = toggle.checked;
    toggle.addEventListener('change', () => {
      showCornerFrequencyMarkers = toggle.checked;
      renderBodePlot(data);
    });
  }

  function setupMagnitudeModeToggle(data) {
    const select = document.getElementById('bodeMagnitudeMode');
    if (!select) return;
    currentPlotMode = select.value === 'straight' ? 'straight' : 'exact';
    select.addEventListener('change', () => {
      currentPlotMode = select.value === 'straight' ? 'straight' : 'exact';
      renderBodePlot(data);
    });
  }


  function renderPoleZeroPlot(data) {
    if (typeof window.Plotly === 'undefined') return;
    const el = document.getElementById('pzPlot');
    if (!el) return;

    const zeros = Array.isArray(data.zeros) ? data.zeros : [];
    const poles = Array.isArray(data.poles) ? data.poles : [];

    const zerosPoints = zeros.map(z => ({ x: Number(z.re) || 0, y: Number(z.im) || 0 }));
    const polesPoints = poles.map(p => ({ x: Number(p.re) || 0, y: Number(p.im) || 0 }));

    const realVals = zerosPoints.map(z => z.x).concat(polesPoints.map(p => p.x));
    const imagVals = zerosPoints.map(z => z.y).concat(polesPoints.map(p => p.y));
    const maxRe = realVals.length ? Math.max(...realVals.map(v => Math.abs(v))) : 1;
    const maxIm = imagVals.length ? Math.max(...imagVals.map(v => Math.abs(v))) : 1;
    const xRange = maxRe === 0 ? 1 : maxRe * 1.2;
    const yRange = maxIm === 0 ? 1 : maxIm * 1.2;

    const traces = [
      {
        x: zerosPoints.map(z => z.x),
        y: zerosPoints.map(z => z.y),
        type: 'scatter',
        mode: zerosPoints.length ? 'markers' : 'text',
        name: 'Zeros',
        marker: {
          symbol: 'circle-open',
          size: 14,
          color: '#0ea5e9',
          line: { width: 2 }
        },
        text: zerosPoints.length ? undefined : ['No zeros'],
        textposition: 'top center'
      },
      {
        x: polesPoints.map(p => p.x),
        y: polesPoints.map(p => p.y),
        type: 'scatter',
        mode: polesPoints.length ? 'markers' : 'text',
        name: 'Poles',
        marker: {
          symbol: 'x',
          size: 14,
          color: '#ef4444',
          line: { width: 2 }
        },
        text: polesPoints.length ? undefined : ['No poles'],
        textposition: 'bottom center'
      }
    ];

    const layout = {
      margin: { l: 60, r: 20, t: 20, b: 40 },
      xaxis: {
        title: 'Real',
        range: [-xRange, xRange],
        zeroline: false,
        showgrid: true,
        gridcolor: '#e5e7eb'
      },
      yaxis: {
        title: 'Imaginary',
        range: [-yRange, yRange],
        zeroline: false,
        showgrid: true,
        gridcolor: '#e5e7eb'
      },
      shapes: [
        { type: 'line', x0: -xRange, x1: xRange, y0: 0, y1: 0, line: { color: '#9ca3af', width: 1 } },
        { type: 'line', x0: 0, x1: 0, y0: -yRange, y1: yRange, line: { color: '#9ca3af', width: 1 } }
      ],
      legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.react(el, traces, layout, basePlotConfig);
  }

  function renderNyquistPlot(data) {
    if (typeof window.Plotly === 'undefined' || !data) return;
    const el = document.getElementById('nyquistPlot');
    if (!el) return;

    const sanitize = arr =>
      Array.isArray(arr)
        ? arr.map(v => {
            if (typeof v === 'number') return v;
            if (v === null || typeof v === 'undefined') return Number.NaN;
            const numeric = Number(v);
            return Number.isFinite(numeric) ? numeric : Number.NaN;
          })
        : [];
    const positive = data.positive || {};
    const negative = data.negative || {};

    const posReal = sanitize(positive.real);
    const posImag = sanitize(positive.imag);
    const posFreq = sanitize(positive.frequencies);
    const negReal = sanitize(negative.real);
    const negImag = sanitize(negative.imag);
    const negFreq = sanitize(negative.frequencies);

    const freqToHover = value => formatFrequency(value);

    const positiveTrace = {
      x: posReal,
      y: posImag,
      type: 'scatter',
      mode: 'lines',
      name: 'ω ≥ 0',
      line: { color: '#2563eb', width: 3 },
      hovertemplate: 'Re: %{x:.4f}<br>Im: %{y:.4f}<br>ω: %{customdata}<extra></extra>',
      customdata: posFreq.map(freqToHover)
    };

    const traces = [positiveTrace];

    if (negReal.length) {
      traces.push({
        x: negReal,
        y: negImag,
        type: 'scatter',
        mode: 'lines',
        name: 'ω ≤ 0',
        line: { color: '#7c3aed', width: 2, dash: 'dot' },
        hovertemplate: 'Re: %{x:.4f}<br>Im: %{y:.4f}<br>ω: %{customdata}<extra></extra>',
        customdata: negFreq.map(freqToHover)
      });
    }

    const critical = data.critical_point || { real: -1, imag: 0 };
    traces.push({
      x: [Number(critical.real) || -1],
      y: [Number(critical.imag) || 0],
      type: 'scatter',
      mode: 'markers',
      name: 'Critical point',
      marker: { color: '#ef4444', size: 10, symbol: 'x' },
      hovertemplate: 'Critical point (-1 + j0)<extra></extra>'
    });

    const low = data.low_freq || null;
    if (low && isFiniteNumber(low.real) && isFiniteNumber(low.imag)) {
      traces.push({
        x: [low.real],
        y: [low.imag],
        type: 'scatter',
        mode: 'markers+text',
        name: 'ω → 0',
        marker: { color: '#10b981', size: 10, symbol: 'circle' },
        text: ['ω → 0'],
        textposition: 'top right',
        hovertemplate: `Re: %{x:.4f}<br>Im: %{y:.4f}<br>ω: ${freqToHover(low.frequency)}<extra></extra>`
      });
    }

    const high = data.high_freq || null;
    if (high && isFiniteNumber(high.real) && isFiniteNumber(high.imag)) {
      traces.push({
        x: [high.real],
        y: [high.imag],
        type: 'scatter',
        mode: 'markers+text',
        name: 'ω → ∞',
        marker: { color: '#fbbf24', size: 10, symbol: 'square' },
        text: ['ω → ∞'],
        textposition: 'bottom left',
        hovertemplate: `Re: %{x:.4f}<br>Im: %{y:.4f}<br>ω: ${freqToHover(high.frequency)}<extra></extra>`
      });
    }

    const allReal = posReal.concat(negReal);
    const allImag = posImag.concat(negImag);
    const finiteReal = allReal.filter(isFiniteNumber);
    const finiteImag = allImag.filter(isFiniteNumber);
    const maxReal = finiteReal.length ? Math.max(...finiteReal.map(Math.abs)) : 0;
    const maxImag = finiteImag.length ? Math.max(...finiteImag.map(Math.abs)) : 0;
    const extent = Math.max(1, maxReal, maxImag);
    const pad = extent * 0.15;
    const limit = extent + pad;
    const circleRadius = Math.min(limit * 0.12, 1.5);

    const layout = {
      margin: { l: 60, r: 30, t: 30, b: 60 },
      hovermode: 'closest',
      showlegend: false,
      xaxis: {
        title: 'Re{L(jω)}',
        showgrid: true,
        gridcolor: '#e5e7eb',
        zeroline: false,
        range: [-limit, limit]
      },
      yaxis: {
        title: 'Im{L(jω)}',
        showgrid: true,
        gridcolor: '#e5e7eb',
        zeroline: false,
        scaleanchor: 'x',
        scaleratio: 1,
        range: [-limit, limit]
      },
      shapes: [
        { type: 'line', x0: -limit, x1: limit, y0: 0, y1: 0, line: { color: '#9ca3af', width: 1 } },
        { type: 'line', x0: 0, x1: 0, y0: -limit, y1: limit, line: { color: '#9ca3af', width: 1 } },
        {
          type: 'circle',
          xref: 'x',
          yref: 'y',
          x0: -1 - circleRadius,
          x1: -1 + circleRadius,
          y0: -circleRadius,
          y1: circleRadius,
          line: { color: 'rgba(239,68,68,0.45)', dash: 'dot', width: 1 }
        }
      ],
      annotations: [
        {
          x: -1,
          y: 0,
          text: '-1',
          showarrow: false,
          font: { size: 12, color: '#ef4444' },
          xanchor: 'left',
          yanchor: 'top'
        }
      ]
    };

    Plotly.react(el, traces, layout, basePlotConfig);
  }


  document.addEventListener('DOMContentLoaded', () => {
    const bootstrap = () => {
      if (bodeData) {
        setupMagnitudeModeToggle(bodeData);
        setupCornerFrequencyToggle(bodeData);
        renderBodePlot(bodeData);
        renderMetrics(bodeData);
        setupBodeDownload();
      }
      if (pzData) {
        renderPoleZeroPlot(pzData);
      }
      if (nyquistData) {
        renderNyquistPlot(nyquistData);
      }
    };

    if (typeof window.ensurePlotlyLoaded === 'function') {
      window.ensurePlotlyLoaded()
        .then(bootstrap)
        .catch((error) => {
          console.error('Unable to load Plotly for Bode page:', error);
        });
      return;
    }
    bootstrap();
  });
})();
