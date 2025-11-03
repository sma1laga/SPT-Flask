(() => {
  const plotlyAvailable = typeof window !== 'undefined' && typeof window.Plotly !== 'undefined';
  const bodeData = window.bodeData || null;
  const pzData = window.pzData || null;
  const nyquistData = window.nyquistData || null;


  const basePlotConfig = { responsive: true, displaylogo: false };

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

  function buildCrossingLines(data) {

    const shapes = [];
    if (isFiniteNumber(data.phase_cross_freq)) {
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
    if (isFiniteNumber(data.gain_cross_freq)) {
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

  function renderBodeMagnitude(data) {
    if (!plotlyAvailable) return;
    const el = document.getElementById('bodeMagnitudePlot');
    if (!el) return;

    const freq = Array.isArray(data.omega) ? data.omega : [];
    const mag = Array.isArray(data.magnitude_db) ? data.magnitude_db : [];

    const layout = {
      margin: { l: 70, r: 20, t: 10, b: 40 },
      hovermode: 'x unified',
      shapes: buildCrossingLines(data),
      xaxis: {
        type: 'log',
        title: 'Frequency (rad/s)',
        showgrid: true,
        gridcolor: '#e5e7eb'
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
          y: mag,
          type: 'scatter',
          mode: 'lines',
          name: 'Magnitude (dB)'
        }
      ],
      layout,
      basePlotConfig
    );
  }

  function renderBodePhase(data) {
    if (!plotlyAvailable) return;
    const el = document.getElementById('bodePhasePlot');
    if (!el) return;

    const freq = Array.isArray(data.omega) ? data.omega : [];
    const phase = Array.isArray(data.phase_deg) ? data.phase_deg : [];

    const layout = {
      margin: { l: 70, r: 20, t: 10, b: 40 },
      hovermode: 'x unified',
      shapes: buildCrossingLines(data),
      xaxis: {
        type: 'log',
        title: 'Frequency (rad/s)',
        showgrid: true,
        gridcolor: '#e5e7eb'
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
          y: phase,
          type: 'scatter',
          mode: 'lines',
          name: 'Phase (°)'
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

  function renderPoleZeroPlot(data) {
    if (!plotlyAvailable) return;
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
    if (!plotlyAvailable || !data) return;
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
    if (bodeData) {
      renderBodePlot(bodeData);
      renderMetrics(bodeData);
    }
    if (pzData) {
      renderPoleZeroPlot(pzData);
    }
    if (nyquistData) {
      renderNyquistPlot(nyquistData);
    }
  });
})();
