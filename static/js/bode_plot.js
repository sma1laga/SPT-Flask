(() => {
  const plotlyAvailable = typeof window !== 'undefined' && typeof window.Plotly !== 'undefined';
  const bodeData = window.bodeData || null;
  const pzData = window.pzData || null;

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

  function renderBodePlot(data) {
    if (!plotlyAvailable) return;
    const el = document.getElementById('bodePlot');
    if (!el) return;

    const freq = Array.isArray(data.omega) ? data.omega : [];
    const mag = Array.isArray(data.magnitude_db) ? data.magnitude_db : [];
    const phase = Array.isArray(data.phase_deg) ? data.phase_deg : [];

    const traces = [
      {
        x: freq,
        y: mag,
        type: 'scatter',
        mode: 'lines',
        name: 'Magnitude (dB)',
        xaxis: 'x',
        yaxis: 'y'
      },
      {
        x: freq,
        y: phase,
        type: 'scatter',
        mode: 'lines',
        name: 'Phase (°)',
        xaxis: 'x2',
        yaxis: 'y2'
      }
    ];

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

    const layout = {
      grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
      margin: { l: 70, r: 20, t: 24, b: 40 },
      hovermode: 'x unified',
      shapes,
      xaxis: { type: 'log', title: 'Frequency (rad/s)', showgrid: true, gridcolor: '#e5e7eb' },
      yaxis: { title: 'Magnitude (dB)', showgrid: true, gridcolor: '#e5e7eb' },
      xaxis2: { type: 'log', title: 'Frequency (rad/s)', showgrid: true, gridcolor: '#e5e7eb' },
      yaxis2: { title: 'Phase (°)', showgrid: true, gridcolor: '#e5e7eb' },
      legend: { orientation: 'h', x: 0, y: -0.25 }
    };

    Plotly.react(el, traces, layout, basePlotConfig);
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

  document.addEventListener('DOMContentLoaded', () => {
    if (bodeData) {
      renderBodePlot(bodeData);
      renderMetrics(bodeData);
    }
    if (pzData) {
      renderPoleZeroPlot(pzData);
    }
  });
})();