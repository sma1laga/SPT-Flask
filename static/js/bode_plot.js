(() => {

  const bodeData = window.bodeData || null;
  const pzData = window.pzData || null;
  const nyquistData = window.nyquistData || null;
  const bodeMeta = window.bodeMeta || {};


  const basePlotConfig = { responsive: true, displaylogo: false };
  let showCornerFrequencyMarkers = true;
  // 'exact' | 'straight' | 'both'
  let currentPlotMode = 'exact';

  const COLORS = {
    magnitude: '#2563eb',
    magnitudeAsymptote: '#93c5fd',
    phase: '#16a34a',
    phaseAsymptote: '#86efac',
    phaseCrossover: '#ef4444',
    gainCrossover: '#f97316',
    corner: '#a855f7',
    grid: '#e5e7eb',
    zeroline: '#9ca3af'
  };

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

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderMetrics(data) {
    const metricsBox = document.getElementById('bodeMetrics');
    if (!metricsBox) return;

    const unavailableHint = data.margins_available === false
      ? 'Not available for transfer functions with complex coefficients'
      : 'Could not be determined';

    let gmText;
    let gmHint = '';
    if (data.gain_margin_infinite) {
      gmText = '∞ dB';
      gmHint = 'The phase never crosses −180°, so the gain can be raised without limit.';
    } else if (data.gain_margin_zero) {
      gmText = isFiniteNumber(data.phase_crossover_freq)
        ? `−∞ dB @ ${formatWithUnit(data.phase_crossover_freq, 'rad/s', 3)}`
        : '−∞ dB';
      gmHint = 'The phase passes −180° at an undamped pole where |H| → ∞; the unity-feedback loop is unstable for any positive gain.';
    } else if (isFiniteNumber(data.gain_margin_db) && data.phase_crossover_at_dc) {
      gmText = `${formatWithUnit(data.gain_margin_db, 'dB', 2)} @ ω → 0`;
      gmHint = 'The phase is ±180° at ω → 0 (negative low-frequency gain), so the gain margin is read at DC.';
    } else if (isFiniteNumber(data.gain_margin_db)) {
      gmText = formatMargin(data.gain_margin_db, data.phase_crossover_freq, 'dB', 'rad/s');
      gmHint = 'Measured at the phase crossover frequency (phase = −180°).';
    } else {
      gmText = '—';
      gmHint = unavailableHint;
    }

    let pmText;
    let pmHint = '';
    if (data.phase_margin_infinite) {
      pmText = '∞ °';
      pmHint = 'The magnitude never crosses 0 dB.';
    } else if (isFiniteNumber(data.phase_margin_deg)) {
      pmText = formatMargin(data.phase_margin_deg, data.gain_crossover_freq, '°', 'rad/s');
      pmHint = 'Measured at the gain crossover frequency (|H| = 0 dB).';
    } else {
      pmText = '—';
      pmHint = unavailableHint;
    }

    let bwText;
    let bwHint = '';
    switch (data.bandwidth_status) {
      case 'value':
        bwText = formatWithUnit(data.bandwidth, 'rad/s', 3);
        bwHint = 'Frequency where |H| falls 3 dB below its DC value.';
        break;
      case 'infinite':
        bwText = '∞';
        bwHint = '|H| never falls 3 dB below its DC value in the examined range.';
        break;
      case 'undefined':
        bwText = '—';
        bwHint = 'Not defined: no finite, non-zero DC gain (pole or zero at the origin).';
        break;
      default:
        bwText = '—';
        bwHint = unavailableHint;
    }

    const dcText = isFiniteNumber(data.phase_dc_deg) ? `${formatNumber(data.phase_dc_deg, 1)} °` : '—';

    metricsBox.innerHTML = `
      <span title="${escapeHtml(gmHint)}"><strong>Gain margin</strong>${gmText}</span>
      <span title="${escapeHtml(pmHint)}"><strong>Phase margin</strong>${pmText}</span>
      <span title="${escapeHtml(bwHint)}"><strong>Bandwidth</strong>${bwText}</span>
      <span title="Phase of H(jω) as ω → 0 (the start of the asymptotic phase plot)"><strong>Low-freq. phase</strong>${dcText}</span>
    `;
  }

  const CROSSOVER_MARGIN_DECADES = 1;

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

  function verticalLine(x, color) {
    return {
      type: 'line',
      x0: x,
      x1: x,
      y0: 0,
      y1: 1,
      xref: 'x',
      yref: 'paper',
      line: { color, dash: 'dash', width: 2 }
    };
  }

  // Red dashed line: phase crossover (phase = −180°, where the gain margin is read).
  // Orange dashed line: gain crossover (|H| = 0 dB, where the phase margin is read).
  function buildCrossingLines(data) {
    const shapes = [];
    const freqRange = getFrequencyRangeFromData(data);
    if (isCrossoverWithinRange(data.phase_crossover_freq, freqRange)) {
      shapes.push(verticalLine(data.phase_crossover_freq, COLORS.phaseCrossover));
    }
    if (isCrossoverWithinRange(data.gain_crossover_freq, freqRange)) {
      shapes.push(verticalLine(data.gain_crossover_freq, COLORS.gainCrossover));
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
      line: { color: COLORS.corner, dash: 'dot', width: 1.5 }
    }));
  }

  function buildBodeShapes(data) {
    const shapes = buildCrossingLines(data);
    return shapes.concat(buildCornerFrequencyShapes(data));
  }

  function buildSeries(data, exactKey, straightKey, labels, colors) {
    const exact = Array.isArray(data[exactKey]) ? data[exactKey] : [];
    const straight = Array.isArray(data[straightKey]) ? data[straightKey] : [];
    const haveStraight = straight.length === exact.length && straight.length > 0;
    const exactTrace = {
      values: exact,
      name: labels.exact,
      line: { color: colors.exact, width: 3 }
    };
    const straightTrace = {
      values: straight,
      name: labels.straight,
      line: { color: colors.straight, width: 2.5, dash: 'dash' }
    };
    if (currentPlotMode === 'straight' && haveStraight) {
      return [{ ...straightTrace, line: { color: colors.exact, width: 3, dash: 'dash' } }];
    }
    if (currentPlotMode === 'both' && haveStraight) {
      return [exactTrace, straightTrace];
    }
    return [exactTrace];
  }

  function getMagnitudeSeries(data) {
    return buildSeries(
      data,
      'magnitude_db',
      'magnitude_straight_db',
      { exact: 'Exact magnitude (dB)', straight: 'Straight-line asymptotes' },
      { exact: COLORS.magnitude, straight: COLORS.magnitudeAsymptote }
    );
  }
  function getPhaseSeries(data) {
    return buildSeries(
      data,
      'phase_deg',
      'phase_straight_deg',
      { exact: 'Exact phase (°)', straight: 'Straight-line approximation' },
      { exact: COLORS.phase, straight: COLORS.phaseAsymptote }
    );
  }

  function seriesToTraces(freq, series) {
    return series.map(entry => ({
      x: freq,
      y: entry.values,
      type: 'scatter',
      mode: 'lines',
      name: entry.name,
      line: entry.line,
      connectgaps: false
    }));
  }

  function makeXAxis() {
    return {
      type: 'log',
      title: { text: 'Frequency (rad/s)' },
      showgrid: true,
      gridcolor: COLORS.grid,
      showexponent: 'all',
      exponentformat: 'power'
    };
  }

  function phaseSpan(series) {
    let min = Infinity;
    let max = -Infinity;
    series.forEach(entry => {
      entry.values.forEach(value => {
        if (isFiniteNumber(value)) {
          if (value < min) min = value;
          if (value > max) max = value;
        }
      });
    });
    return Number.isFinite(min) && Number.isFinite(max) ? max - min : 0;
  }

  function renderBodeMagnitude(data) {
    if (typeof window.Plotly === 'undefined') return;
    const el = document.getElementById('bodeMagnitudePlot');
    if (!el) return;

    const freq = Array.isArray(data.omega) ? data.omega : [];
    const series = getMagnitudeSeries(data);

    const layout = {
      margin: { l: 70, r: 20, t: 10, b: 40 },
      hovermode: 'x unified',
      shapes: buildBodeShapes(data),
      xaxis: makeXAxis(),
      yaxis: {
        title: { text: 'Magnitude (dB)' },
        showgrid: true,
        gridcolor: COLORS.grid,
        zeroline: true,
        zerolinecolor: COLORS.zeroline,
        zerolinewidth: 1.5
      },
      showlegend: series.length > 1,
      legend: { orientation: 'h', x: 0, y: 1.12 }
    };

    Plotly.react(el, seriesToTraces(freq, series), layout, basePlotConfig);
  }

  function renderBodePhase(data) {
    if (typeof window.Plotly === 'undefined') return;
    const el = document.getElementById('bodePhasePlot');
    if (!el) return;

    const freq = Array.isArray(data.omega) ? data.omega : [];
    const series = getPhaseSeries(data);
    const shapes = buildBodeShapes(data);

    // Horizontal reference at the −180° level (on the plotted branch) when a phase crossover exists.
    const freqRange = getFrequencyRangeFromData(data);
    if (isFiniteNumber(data.phase_crossover_level_deg) && isCrossoverWithinRange(data.phase_crossover_freq, freqRange)) {
      shapes.push({
        type: 'line',
        x0: 0,
        x1: 1,
        y0: data.phase_crossover_level_deg,
        y1: data.phase_crossover_level_deg,
        xref: 'paper',
        yref: 'y',
        line: { color: COLORS.phaseCrossover, dash: 'dot', width: 1 }
      });
    }

    const span = phaseSpan(series);
    const yaxis = {
      title: { text: 'Phase (°)' },
      showgrid: true,
      gridcolor: COLORS.grid,
      zeroline: true,
      zerolinecolor: COLORS.zeroline
    };
    if (span > 0 && span <= 400) {
      yaxis.dtick = 45;
    } else if (span > 400 && span <= 900) {
      yaxis.dtick = 90;
    }

    const layout = {
      margin: { l: 70, r: 20, t: 10, b: 40 },
      hovermode: 'x unified',
      shapes,
      xaxis: makeXAxis(),
      yaxis,
      showlegend: series.length > 1,
      legend: { orientation: 'h', x: 0, y: 1.12 }
    };

    Plotly.react(el, seriesToTraces(freq, series), layout, basePlotConfig);
  }

  function renderBodePlot(data) {
    renderBodeMagnitude(data);
    renderBodePhase(data);
  }

  function getTransferFunctionExportText() {
    if (typeof bodeMeta.functionText === 'string' && bodeMeta.functionText.trim()) {
      return bodeMeta.functionText.trim();
    }
    const numerator = typeof bodeMeta.numerator === 'string' ? bodeMeta.numerator.trim() : '';
    const denominator = typeof bodeMeta.denominator === 'string' ? bodeMeta.denominator.trim() : '';
    if (numerator && denominator) {
      return `H(s) = (${numerator}) / (${denominator})`;
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

  function axisTitleText(axis, fallback) {
    if (!axis) return fallback;
    if (axis.title && typeof axis.title === 'object' && typeof axis.title.text === 'string') {
      return axis.title.text;
    }
    if (typeof axis.title === 'string') return axis.title;
    return fallback;
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
      // Right-aligned so the legend (only shown in "both" mode) never meets the left-aligned title.
      legend: { orientation: 'h', x: 1, xanchor: 'right', y: 1.02, yanchor: 'bottom', font: { size: 16 } },
      font: { family: 'Arial, sans-serif', size: 18, color: '#111827' },
      xaxis: {
        ...xaxis,
        title: { text: axisTitleText(xaxis, 'Frequency (rad/s)'), font: { size: 20 } },
        tickfont: { size: 16 },
        gridcolor: '#d1d5db',
        zerolinecolor: '#9ca3af'
      },
      yaxis: {
        ...yaxis,
        title: { text: axisTitleText(yaxis, ''), font: { size: 20 } },
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

  function roundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(x, y, width, height, radius);
    } else {
      ctx.rect(x, y, width, height);
    }
    ctx.fill();
    ctx.stroke();
  }

  function modeLabel() {
    if (currentPlotMode === 'straight') return 'Straight-line approximation';
    if (currentPlotMode === 'both') return 'Exact response with straight-line asymptotes';
    return 'Exact response';
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

    const markersLabel = showCornerFrequencyMarkers ? 'Corner frequencies shown' : 'Corner frequencies hidden';
    ctx.font = '22px Arial, sans-serif';
    ctx.fillStyle = '#4b5563';
    ctx.fillText(`${modeLabel()} • ${markersLabel}`, 90, 182);

    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#dbeafe';
    ctx.lineWidth = 2;
    roundedRect(ctx, 70, 220, 1660, 560, 22);
    ctx.drawImage(magnitudeImage, 100, 242, 1600, 520);

    roundedRect(ctx, 70, 830, 1660, 560, 22);
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

  function readPlotMode(select) {
    return ['exact', 'straight', 'both'].includes(select.value) ? select.value : 'exact';
  }

  function setupPlotModeToggle(data) {
    const select = document.getElementById('bodePlotMode');
    if (!select) return;
    currentPlotMode = readPlotMode(select);
    select.addEventListener('change', () => {
      currentPlotMode = readPlotMode(select);
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

    const traces = [];
    if (zerosPoints.length) {
      traces.push({
        x: zerosPoints.map(z => z.x),
        y: zerosPoints.map(z => z.y),
        type: 'scatter',
        mode: 'markers',
        name: 'Zeros',
        marker: {
          symbol: 'circle-open',
          size: 14,
          color: '#0ea5e9',
          line: { width: 2 }
        },
        hovertemplate: 'Zero<br>Re: %{x:.4g}<br>Im: %{y:.4g}<extra></extra>'
      });
    }
    if (polesPoints.length) {
      traces.push({
        x: polesPoints.map(p => p.x),
        y: polesPoints.map(p => p.y),
        type: 'scatter',
        mode: 'markers',
        name: 'Poles',
        marker: {
          symbol: 'x',
          size: 14,
          color: '#ef4444',
          line: { width: 2 }
        },
        hovertemplate: 'Pole<br>Re: %{x:.4g}<br>Im: %{y:.4g}<extra></extra>'
      });
    }

    const annotations = [];
    if (!zerosPoints.length) {
      annotations.push({
        xref: 'paper', yref: 'paper', x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top',
        text: 'No finite zeros', showarrow: false, font: { color: '#0ea5e9', size: 13 }
      });
    }
    if (!polesPoints.length) {
      annotations.push({
        xref: 'paper', yref: 'paper', x: 0.02, y: 0.02, xanchor: 'left', yanchor: 'bottom',
        text: 'No finite poles', showarrow: false, font: { color: '#ef4444', size: 13 }
      });
    }

    const layout = {
      margin: { l: 60, r: 20, t: 20, b: 40 },
      xaxis: {
        title: { text: 'Real' },
        range: [-xRange, xRange],
        zeroline: false,
        showgrid: true,
        gridcolor: COLORS.grid
      },
      yaxis: {
        title: { text: 'Imaginary' },
        range: [-yRange, yRange],
        zeroline: false,
        showgrid: true,
        gridcolor: COLORS.grid
      },
      shapes: [
        { type: 'line', x0: -xRange, x1: xRange, y0: 0, y1: 0, line: { color: COLORS.zeroline, width: 1 } },
        { type: 'line', x0: 0, x1: 0, y0: -yRange, y1: yRange, line: { color: COLORS.zeroline, width: 1 } }
      ],
      annotations,
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
      connectgaps: false,
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
        connectgaps: false,
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

    const crossings = Array.isArray(data.real_axis_crossings) ? data.real_axis_crossings : [];
    const crossingReals = crossings.map(c => Number(c.real)).filter(isFiniteNumber);
    if (crossingReals.length) {
      traces.push({
        x: crossingReals,
        y: crossingReals.map(() => 0),
        type: 'scatter',
        mode: 'markers',
        name: 'Real-axis crossing',
        marker: { color: '#f97316', size: 9, symbol: 'diamond' },
        hovertemplate: 'Real-axis crossing<br>Re: %{x:.4f}<extra></extra>'
      });
    }

    // Default view: focus on the region around the critical point.  For systems with poles
    // at the origin the locus runs off to infinity, which would otherwise dominate the
    // autoscale and shrink everything that matters to a dot.  Plotly's "Autoscale" button
    // still shows the whole locus.
    const allReal = posReal.concat(negReal);
    const allImag = posImag.concat(negImag);
    const fullExtent = Math.max(
      1,
      ...allReal.filter(isFiniteNumber).map(Math.abs),
      ...allImag.filter(isFiniteNumber).map(Math.abs)
    );
    const focusThreshold = Math.max(2, 2 * Math.max(0, ...crossingReals.map(Math.abs)));
    let coreExtent = 0;
    for (let i = 0; i < allReal.length; i += 1) {
      const re = allReal[i];
      const im = allImag[i];
      if (!isFiniteNumber(re) || !isFiniteNumber(im)) continue;
      if (Math.hypot(re, im) <= focusThreshold) {
        coreExtent = Math.max(coreExtent, Math.abs(re), Math.abs(im));
      }
    }
    const extent = coreExtent > 0 ? Math.max(1.2, coreExtent) : fullExtent;
    const limit = Math.min(fullExtent * 1.15, extent * 1.15 + 0.2);
    const circleRadius = Math.min(limit * 0.12, 1.5);

    const layout = {
      margin: { l: 60, r: 30, t: 30, b: 60 },
      hovermode: 'closest',
      showlegend: false,
      xaxis: {
        title: { text: 'Re{L(jω)}' },
        showgrid: true,
        gridcolor: COLORS.grid,
        zeroline: false,
        range: [-limit, limit]
      },
      yaxis: {
        title: { text: 'Im{L(jω)}' },
        showgrid: true,
        gridcolor: COLORS.grid,
        zeroline: false,
        scaleanchor: 'x',
        scaleratio: 1,
        range: [-limit, limit]
      },
      shapes: [
        { type: 'line', x0: -fullExtent * 1.15, x1: fullExtent * 1.15, y0: 0, y1: 0, line: { color: COLORS.zeroline, width: 1 } },
        { type: 'line', x0: 0, x1: 0, y0: -fullExtent * 1.15, y1: fullExtent * 1.15, line: { color: COLORS.zeroline, width: 1 } },
        {
          type: 'circle',
          xref: 'x',
          yref: 'y',
          x0: -1 - circleRadius,
          x1: -1 + circleRadius,
          y0: -circleRadius,
          y1: circleRadius,
          line: { color: 'rgba(239,68,68,0.45)', dash: 'dot', width: 1 }
        },
        {
          type: 'circle',
          xref: 'x',
          yref: 'y',
          x0: -1,
          x1: 1,
          y0: -1,
          y1: 1,
          line: { color: 'rgba(107,114,128,0.35)', dash: 'dot', width: 1 }
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
        setupPlotModeToggle(bodeData);
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
