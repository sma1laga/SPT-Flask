// static/js/digital_modulation.js

// ---------- small utils ----------
function debounce(fn, wait = 150) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), wait); };
}

function getEl(id) { return document.getElementById(id); }

function setSliderLabel(id) {
  const el = getEl(id);
  const lab = getEl(id + '_val');
  if (el && lab) lab.textContent = el.value;
}

function getDefaultLayout(title, height = 280) {
  return {
    margin: { t: 30, r: 10, l: 50, b: 40 },
    title,
    height,
    legend: { orientation: 'h' },
    xaxis: { automargin: true, title: 't [s]' },
    yaxis: { automargin: true, title: 'Signal' }
  };
}

function plotlySafeReact(divId, data, layout) {
  const div = getEl(divId);
  if (!div || typeof Plotly === 'undefined') return;
  Plotly.react(div, data, layout, { responsive: true, displayModeBar: false });
}

function plotlyResize(divId) {
  const div = getEl(divId);
  if (!div || typeof Plotly === 'undefined') return;
  try { Plotly.Plots.resize(div); } catch (_) {}
}

// ---------- core helpers ----------
function makeTime(tEnd = 1.0, fs = 1000) {
  const N = Math.max(8, Math.floor(fs * tEnd));
  const t = new Array(N);
  for (let i = 0; i < N; i++) t[i] = i / fs;
  return t;
}

function digitalModulate({ type, fc, br, dev }) {
  const fs = 1000;       // keep lightweight client side demo
  const tEnd = 1.0;
  const t = makeTime(tEnd, fs);
  const Tb = 1 / br;
  const bits = t.map(v => (v % Tb < Tb / 2 ? 1 : 0));

  let y;
  if (type === 'ASK') {
    y = t.map((v, i) => bits[i] * Math.cos(2 * Math.PI * fc * v));
  } else if (type === 'PSK') {
    // 0 -> phase 0, 1 -> phase pi - BPSK demo
    y = t.map((v, i) => Math.cos(2 * Math.PI * fc * v + Math.PI * (1 - bits[i])));
  } else if (type === 'FSK') {
    const f0 = fc - dev, f1 = fc + dev;
    y = t.map((v, i) => Math.cos(2 * Math.PI * (bits[i] ? f1 : f0) * v));
  } else {
    throw new Error('Unknown digital modulation type');
  }

  return { t, bits, modulated: y, fs };
}

function digitalDemodulate({ br }) {
  // echoes ideal bits as a clean reference
  const fs = 1000, tEnd = 1.0;
  const t = makeTime(tEnd, fs);
  const Tb = 1 / br;
  const bits = t.map(v => (v % Tb < Tb / 2 ? 1 : 0));
  return { t, modulated: [...bits], demodulated: bits, fs };
}

function estimateConstellation({ type, fc, br }) {
  if (type === 'FSK') return { I: [], Q: [], evm: null };

  const { t, modulated, fs } = digitalModulate({ type, fc, br, dev: 0 });
  const spb = Math.max(1, Math.floor(fs / br));
  const nSym = Math.floor(t.length / spb);
  const I = new Array(nSym), Q = new Array(nSym);

  for (let k = 0; k < nSym; k++) {
    let sI = 0, sQ = 0;
    const n0 = k * spb, n1 = (k + 1) * spb;
    for (let n = n0; n < n1; n++) {
      const c = Math.cos(2 * Math.PI * fc * t[n]);
      const s = -Math.sin(2 * Math.PI * fc * t[n]);
      sI += modulated[n] * c;
      sQ += modulated[n] * s;
    }
    I[k] = (2 / spb) * sI;
    Q[k] = (2 / spb) * sQ;
  }

  // crude EVM vs two ideal BPSK points (+/-1, 0)
  let err = 0;
  const ideals = [[1, 0], [-1, 0]];
  for (let k = 0; k < I.length; k++) {
    const d0 = Math.hypot(I[k] - 1, Q[k] - 0);
    const d1 = Math.hypot(I[k] + 1, Q[k] - 0);
    err += Math.min(d0, d1);
  }
  const evm = I.length ? err / I.length : null;
  return { I, Q, evm };
}

// ---------- plotting ----------
function plotDigital(demod = false) {
  const type = demod ? getEl('dig_demod').value : getEl('dig_type').value;

  const params = {
    type,
    fc: +getEl('dig_fc').value,
    br: +getEl('dig_br').value,
    dev: +getEl('dig_dev').value
  };

  const data = demod ? digitalDemodulate(params) : digitalModulate(params);

  if (!demod) {
    const traces = [
      { x: data.t, y: data.bits, name: 'Bits', yaxis: 'y2' },
      { x: data.t, y: data.modulated, name: `${type} Signal` }
    ];
    const layout = getDefaultLayout(`${type} Modulation`);
    layout.yaxis2 = { overlaying: 'y', side: 'right', showgrid: false, title: 'Bits', range: [-0.2, 1.2] };
    plotlySafeReact('dig_plot', traces, layout);
  } else {
    const traces = [
      { x: data.t, y: data.modulated,   name: 'Received' },
      { x: data.t, y: data.demodulated, name: 'Demodulated Bits', yaxis: 'y2' }
    ];
    const layout = getDefaultLayout(`${type} Demodulation`);
    layout.yaxis2 = { overlaying: 'y', side: 'right', showgrid: false, title: 'Bits', range: [-0.2, 1.2] };
    plotlySafeReact('dig_demod_plot', traces, layout);
  }
}

function drawConstellation() {
  const type = getEl('dig_type').value;
  const fc   = +getEl('dig_fc').value;
  const br   = +getEl('dig_br').value;

  const infoEl = getEl('dig_constellation');
  const plotId = 'dig_constellation_plot';

  const { I, Q, evm } = estimateConstellation({ type, fc, br });

  if (!I.length) {
    if (infoEl) infoEl.innerHTML = '<span class="muted">Constellation not applicable (FSK)</span>';
    const div = getEl(plotId);
    if (div) Plotly.purge(div);
    return;
  }

  if (infoEl) infoEl.innerHTML =
    `<strong>Constellation</strong> — per-symbol integrate/mix; EVM≈${evm?.toFixed(2) ?? 'n/a'}`;

  const traces = [
    { x: I, y: Q, mode: 'markers', name: 'Symbols' },
    { x: [1, -1], y: [0, 0], mode: 'markers', name: 'Ideal BPSK' }
  ];
  const layout = {
    ...getDefaultLayout('Constellation', 300),
    xaxis: { title: 'I', automargin: true },
    yaxis: { title: 'Q', automargin: true, scaleanchor: 'x' }
  };
  plotlySafeReact(plotId, traces, layout);
}

// ---------- UI wiring  ----------
function updateSliderValue(id) {
  setSliderLabel(id);
  plotDigital(false);
  plotDigital(true);
  drawConstellation();
}

function updateDigControls() {
  const type = getEl('dig_type').value;
  const devGroup = getEl('dev_group');
  if (devGroup) devGroup.style.display = (type === 'FSK' ? '' : 'none');
  plotDigital(false);
  drawConstellation();
}

// ---------- init ----------
document.addEventListener('DOMContentLoaded', () => {
  // slider labels and live update
  ['dig_fc','dig_br','dig_dev'].forEach(id => {
    const el = getEl(id);
    if (!el) return;
    setSliderLabel(id);
    el.addEventListener('input', () => updateSliderValue(id));
  });

  // scheme & demod selectorss
  const typeSel = getEl('dig_type');
  if (typeSel) typeSel.addEventListener('change', () => {
    updateDigControls();
    ['dig_fc','dig_br','dig_dev'].forEach(setSliderLabel);
  });

  const demodSel = getEl('dig_demod');
  if (demodSel) demodSel.addEventListener('change', () => plotDigital(true));

  // first draw
  updateDigControls();
  plotDigital(false);
  plotDigital(true);
  drawConstellation();

  // resize handling — good for stacked layout
  const onResize = debounce(() => {
    ['dig_plot','dig_demod_plot','dig_constellation_plot'].forEach(plotlyResize);
  }, 100);
  window.addEventListener('resize', onResize);
});
