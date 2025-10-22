// static/js/digital_modulation.js

window.DIG_URLS = window.DIG_URLS || {
  presets: '/digital_modulation/api/presets',
  modulate: '/digital_modulation/api/modulate',
  demod: '/digital_modulation/api/demodulate',
  mpam: '/digital_modulation/api/m_pam',
  passband: '/digital_modulation/api/passband'
};

const $ = (id) => document.getElementById(id);
const setLabel = (id) => {
  const el = $(id);
  const lab = $(id + '_val');
  if (el && lab) lab.innerText = el.value;
};

const PLOT_CONFIG = { responsive: true, displayModeBar: false };
const tracesNoHover = (traces = []) => traces.map((trace) => ({ ...trace, hoverinfo: 'skip' }));
const withNoHover = (layout = {}) => Object.assign({ hovermode: false }, layout);


async function fetchJSON(url, params = {}) {
  const q = new URLSearchParams(params).toString();
  const res = await fetch(q ? `${url}?${q}` : url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function updateDigControls() {
  const type = $('dig_mod_type').value;
  const levelsGroup = $('dig_pcm_levels_group');
  if (levelsGroup) levelsGroup.style.display = type === 'PCM' ? '' : 'none';

  [
    'dig_fs',
    'dig_t_end',
    'dig_prf',
    'dig_fm',
    'dig_levels',
    'dig_snr_db'
  ].forEach(setLabel);
}

function renderDigFacts(info) {
  if (!info) return;
  const parts = [];
  if (info.type) parts.push(`${info.type} @ fs=${info.fs} Hz`);
  if (info.prf != null) parts.push(`prf = ${info.prf} Hz`);
  if (info.fm != null) parts.push(`fm = ${info.fm} Hz`);
  if (info.levels != null) parts.push(`${info.levels} levels`);
  if (info.snr_db != null && info.snr_db !== 'inf') {
    const snr = Number(info.snr_db);
    if (Number.isFinite(snr)) parts.push(`SNR = ${snr.toFixed(1)} dB`);
  }
  const el = $('dig_facts');
  if (el) el.innerText = parts.join('  •  ');
}

function updatePamControls() {
  [
    'pam_rolloff',
    'pam_timing_offset',
    'pam_sps',
    'pam_span',
    'pam_snr',
    'pam_symbols'
  ].forEach(setLabel);

  const offsetEl = $('pam_timing_offset');
  if (offsetEl) {
    const lab = $('pam_timing_offset_val');
    if (lab) {
      const val = Number(offsetEl.value);
      lab.innerText = Number.isFinite(val) ? val.toFixed(2) : offsetEl.value;
    }
  }
}

function collectPamParams() {
  const orderEl = $('pam_m');
  const rolloffEl = $('pam_rolloff');
  const timingEl = $('pam_timing_offset');
  const spsEl = $('pam_sps');
  const spanEl = $('pam_span');
  const snrEl = $('pam_snr');
  const symEl = $('pam_symbols');
  if (!orderEl || !rolloffEl || !timingEl || !spsEl || !spanEl || !snrEl || !symEl) {
    return {};
  }
  const disableTx = $('pam_disable_tx')?.checked;
  const disableRx = $('pam_disable_rx')?.checked;
  return {
    M: +orderEl.value,
    rolloff: +rolloffEl.value,
    timing_offset: +timingEl.value,
    sps: +spsEl.value,
    span: +spanEl.value,
    snr_db: +snrEl.value,
    symbols: +symEl.value,
    tx_filter: disableTx ? 0 : 1,
    rx_filter: disableRx ? 0 : 1
  };
}

function formatBer(value, bits) {
  if (value == null) return '—';
  const v = Number(value);
  if (!Number.isFinite(v)) return '—';
  if (v <= 0) {
    if (bits) {
      const floor = 1 / Math.max(1, Number(bits));
      return `< ${floor.toExponential(1)}`;
    }
    return '0';
  }
  if (v < 1e-3) return v.toExponential(2);
  return v.toFixed(3);
}

function renderPamFacts(info, ber, challenge) {
  const el = $('pam_facts');
  if (!el) return;
  if (!info) {
    el.innerText = '';
    return;
  }
  const parts = [];
  if (info.M) parts.push(`${info.M}-PAM`);
  if (info.bits_per_symbol != null) parts.push(`${info.bits_per_symbol} bits/sym`);
  if (info.rolloff != null) parts.push(`α=${Number(info.rolloff).toFixed(2)}`);
  if (info.timing_offset != null) parts.push(`Δt=${Number(info.timing_offset).toFixed(2)}T`);
  if (info.sps != null) parts.push(`sps=${info.sps}`);
  if (info.span != null) parts.push(`span=${info.span}`);
  if (info.tx_rrc === false) parts.push('TX RRC off');
  if (info.rx_rrc === false) parts.push('RX MF off');
  const point = ber?.point;
  if (point) {
    if (point.snr_db != null) parts.push(`Eb/N₀=${Number(point.snr_db).toFixed(1)} dB`);
    if (point.measured != null) parts.push(`BER ≈ ${formatBer(point.measured, point.bits)}`);
    if (point.theory != null) parts.push(`theory ${formatBer(point.theory)}`);
  }
  el.innerText = parts.join('  •  ');

  const challengeEl = $('pam_challenge');
  if (!challengeEl) return;
  if (!challenge || challenge.target == null || challenge.opening == null) {
    challengeEl.innerText = '';
    return;
  }

  const target = Number(challenge.target);
  const opening = Number(challenge.opening);
  if (!Number.isFinite(target) || !Number.isFinite(opening)) {
    challengeEl.innerText = '';
    return;
  }

  const roll = info?.rolloff != null ? Number(info.rolloff).toFixed(2) : '—';
  const passed = Boolean(challenge.passed && opening >= target);
  const delta = opening - target;
  const sign = delta >= 0 ? '+' : '−';
  const absDelta = Math.abs(delta).toFixed(2);
  const badge = passed ? '✅' : '⚠️';
  challengeEl.innerText = `${badge} Micro-challenge: eye ≥ ${target.toFixed(2)} @ α=${roll}. Currently ${opening.toFixed(2)} (${sign}${absDelta}).`;
}


function updatePassbandControls() {
  [
    'pass_carrier',
    'pass_symbol_rate',
    'pass_offset',
    'pass_phase',
    'pass_sps'
  ].forEach(setLabel);
}

function collectPassbandParams() {
  const schemeEl = $('pass_scheme');
  const params = {
    scheme: schemeEl ? schemeEl.value : 'BPSK',
    carrier: +($('pass_carrier')?.value || 2000),
    symbol_rate: +($('pass_symbol_rate')?.value || 400),
    freq_offset: +($('pass_offset')?.value || 0),
    phase_deg: +($('pass_phase')?.value || 0),
    sps: +($('pass_sps')?.value || 16),
    costas: $('pass_costas')?.checked ? 1 : 0
  };
  return params;
}

function renderPassbandFacts(info) {
  const el = $('passband_info');
  if (!el) return;
  if (!info) {
    el.innerText = '';
    return;
  }
  const parts = [];
  if (info.scheme) parts.push(info.scheme);
  if (info.bits_per_symbol != null) parts.push(`${info.bits_per_symbol} bits/sym`);
  if (info.symbol_rate != null) parts.push(`${Number(info.symbol_rate).toFixed(0)} baud`);
  if (info.carrier_hz != null) parts.push(`fc=${Number(info.carrier_hz).toFixed(0)} Hz`);
  if (info.freq_offset_hz != null && Number(info.freq_offset_hz) !== 0) {
    parts.push(`Δf=${Number(info.freq_offset_hz).toFixed(1)} Hz`);
  }
  if (info.phase_offset_deg != null && Number(info.phase_offset_deg) !== 0) {
    parts.push(`Δφ=${Number(info.phase_offset_deg).toFixed(1)}°`);
  }
  if (info.costas_enabled === false) {
    parts.push('Costas loop off');
  } else if (info.costas_enabled) {
    parts.push('Costas loop on');
    if (info.estimated_freq_hz != null) {
      parts.push(`est Δf≈${Number(info.estimated_freq_hz).toFixed(2)} Hz`);
    }
    if (info.estimated_phase_deg != null) {
      parts.push(`est Δφ≈${Number(info.estimated_phase_deg).toFixed(1)}°`);
    }
  }
  el.innerText = parts.join('  •  ');
}

async function plotPassband() {
  const params = collectPassbandParams();
  try {
    const data = await fetchJSON(DIG_URLS.passband, params);
    const t = (data.time || []).map(Number);
    const tx = (data.tx_passband || []).map(Number);
    const rx = (data.rx_passband || []).map(Number);

    const waveTraces = [];
    if (t.length && tx.length) {
      waveTraces.push({ x: t, y: tx, name: 'TX passband', line: { color: '#1d4ed8' } });
    }
    if (t.length && rx.length) {
      waveTraces.push({ x: t, y: rx, name: 'RX (offset)', line: { color: '#dc2626' } });
    }
    if (waveTraces.length) {
      Plotly.newPlot('passband_wave_plot', tracesNoHover(waveTraces), withNoHover({
        margin: { t: 30 },
        legend: { orientation: 'h' },
        xaxis: { title: 'Time [s]' },
        yaxis: { title: 'Amplitude' }
      }), PLOT_CONFIG);
    } else {
      Plotly.purge('passband_wave_plot');
      showError('passband_wave_plot', 'No waveform data');
    }

    const base = data.baseband || {};
    const baseTime = (base.time || []).map(Number);
    const baseTraces = [];
    if (baseTime.length && Array.isArray(base.i_tx) && base.i_tx.length) {
      baseTraces.push({ x: baseTime, y: base.i_tx.map(Number), name: 'I (tx)', line: { color: '#111827' } });
    }
    if (baseTime.length && Array.isArray(base.q_tx) && base.q_tx.length) {
      baseTraces.push({ x: baseTime, y: base.q_tx.map(Number), name: 'Q (tx)', line: { color: '#6b7280', dash: 'dash' } });
    }
    if (baseTime.length && Array.isArray(base.i_rx) && base.i_rx.length) {
      baseTraces.push({ x: baseTime, y: base.i_rx.map(Number), name: 'I (rx)', line: { color: '#2563eb' } });
    }
    if (baseTime.length && Array.isArray(base.q_rx) && base.q_rx.length) {
      baseTraces.push({ x: baseTime, y: base.q_rx.map(Number), name: 'Q (rx)', line: { color: '#ec4899' } });
    }
    if (baseTime.length && Array.isArray(base.i_costas) && base.i_costas.length) {
      baseTraces.push({ x: baseTime, y: base.i_costas.map(Number), name: 'I (Costas)', line: { color: '#16a34a' } });
    }
    if (baseTime.length && Array.isArray(base.q_costas) && base.q_costas.length) {
      baseTraces.push({ x: baseTime, y: base.q_costas.map(Number), name: 'Q (Costas)', line: { color: '#f97316' } });
    }
    if (baseTraces.length) {
      Plotly.newPlot('passband_iq_plot', tracesNoHover(baseTraces), withNoHover({
        margin: { t: 30 },
        legend: { orientation: 'h' },
        xaxis: { title: 'Time [s]' },
        yaxis: { title: 'I/Q amplitude' }
      }), PLOT_CONFIG);
    } else {
      Plotly.purge('passband_iq_plot');
      showError('passband_iq_plot', 'No baseband data');
    }

    const cons = data.constellation || {};
    const idealX = (cons.ideal_i || []).map(Number);
    const idealY = (cons.ideal_q || []).map(Number);
    const rxX = (cons.rx_i || []).map(Number);
    const rxY = (cons.rx_q || []).map(Number);
    const corrX = (cons.corrected_i || []).map(Number);
    const corrY = (cons.corrected_q || []).map(Number);

    const constTraces = [];
    if (idealX.length) {
      constTraces.push({
        x: idealX,
        y: idealY,
        mode: 'markers',
        name: 'Ideal',
        marker: { color: '#9ca3af', size: 6, symbol: 'circle-open' }
      });
    }
    if (rxX.length) {
      constTraces.push({
        x: rxX,
        y: rxY,
        mode: 'markers',
        name: 'RX',
        marker: { color: '#ef4444', size: 7, opacity: 0.7 }
      });
    }
    if (corrX.length) {
      constTraces.push({
        x: corrX,
        y: corrY,
        mode: 'markers',
        name: cons.costas ? 'Costas corrected' : 'RX (no Costas)',
        marker: { color: '#10b981', size: 7, opacity: 0.75 }
      });
    }
    if (constTraces.length) {
      Plotly.newPlot('passband_constellation_plot', tracesNoHover(constTraces), withNoHover({
        margin: { t: 30 },
        legend: { orientation: 'h' },
        xaxis: { title: 'In-phase', zeroline: true, scaleanchor: 'y', scaleratio: 1 },
        yaxis: { title: 'Quadrature', zeroline: true }
      }), PLOT_CONFIG);
    } else {
      Plotly.purge('passband_constellation_plot');
      showError('passband_constellation_plot', 'No constellation data');
    }

    renderPassbandFacts(data.info);
  } catch (err) {
    const msg = `Passband error: ${err.message || err}`;
    ['passband_wave_plot', 'passband_iq_plot', 'passband_constellation_plot'].forEach((id) => {
      Plotly.purge(id);
      showError(id, msg);
    });
  }
}


function collectModParams(){
  const type = $('dig_mod_type').value;
  const params = {
    type,
    fs: +$('dig_fs').value,
    t_end: +$('dig_t_end').value,
    snr_db: $('dig_snr_toggle').checked ? +$('dig_snr_db').value : Infinity,
    prf: +$('dig_prf').value,
    fm: +$('dig_fm').value
  };
  if (type === 'PCM') params.levels = +$('dig_levels').value;
  return params;
}

function collectDemodParams(){
  const type = $('dig_demod_type').value;

  const params = {
    type,
    fs: +$('dig_fs').value,
    t_end: +$('dig_t_end').value,
    prf: +$('dig_prf').value,
    fm: +$('dig_fm').value
  };
  if (type === 'PCM') params.levels = +$('dig_levels').value;
  return params;
}

function showError(targetId, err) {
  const el = $(targetId);
  if (el) el.innerHTML = `<div class="muted">${err}</div>`;
}

async function plotPam() {
  const params = collectPamParams();
  if (params.M == null) return;
  try {
    const data = await fetchJSON(DIG_URLS.mpam, params);

    const constellation = data.constellation || {};
    const samples = (constellation.samples || []).map(Number);
    const jitter = samples.map((_, idx) => ((idx % 5) - 2) * 0.02);
    const ideal = (constellation.ideal || []).map(Number);

    const traces = [];
    if (ideal.length) {
      traces.push({
        x: ideal,
        y: ideal.map(() => 0),
        mode: 'markers',
        name: 'Ideal levels',
        marker: { size: 10, symbol: 'line-ns-open', color: '#111827' }
      });
    }
    if (samples.length) {
      traces.push({
        x: samples,
        y: jitter,
        mode: 'markers',
        name: 'Matched samples',
        marker: { size: 6, color: '#2563eb', opacity: 0.7 }
      });
    }
    Plotly.newPlot('pam_constellation_plot', tracesNoHover(traces), withNoHover({
      margin: { t: 30 },
      xaxis: { title: 'Amplitude', zeroline: false },
      yaxis: { showticklabels: false, showgrid: false, zeroline: false, range: [-0.12, 0.12] },
      legend: { orientation: 'h' }
    }), PLOT_CONFIG);

    const eye = data.eye || {};
    const eyeTime = Array.isArray(eye.time) ? eye.time.map(Number) : [];
    const eyeTraces = Array.isArray(eye.traces) ? eye.traces.map((row) => row.map(Number)) : [];
    const meanTrace = Array.isArray(eye.mean) ? eye.mean.map(Number) : null;
    const cursorTime = typeof eye.cursor_time === 'number' ? Number(eye.cursor_time) : null;
    const eyeOpening = typeof eye.opening === 'number' ? Number(eye.opening) : null;
    const eyeHigh = typeof eye.high === 'number' ? Number(eye.high) : null;
    const eyeLow = typeof eye.low === 'number' ? Number(eye.low) : null;
    let yMin = Number.isFinite(Number(eye.y_min)) ? Number(eye.y_min) : null;
    let yMax = Number.isFinite(Number(eye.y_max)) ? Number(eye.y_max) : null;

    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
      let minVal = Infinity;
      let maxVal = -Infinity;
      eyeTraces.forEach((row) => {
        row.forEach((val) => {
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        });
      });
      if (minVal !== Infinity && maxVal !== -Infinity) {
        yMin = minVal;
        yMax = maxVal;
      }
    }
    if (!Number.isFinite(yMin)) yMin = -1;
    if (!Number.isFinite(yMax)) yMax = 1;
    if (eyeTime.length && eyeTraces.length) {
      const eyePlot = eyeTraces.map((row) => ({
        x: eyeTime,
        y: row,
        mode: 'lines',
        line: { color: 'rgba(59, 130, 246, 0.25)', width: 1 },
        hoverinfo: 'skip',
        showlegend: false
      }));
      const avg = meanTrace && meanTrace.length === eyeTime.length
        ? meanTrace
        : eyeTime.map((_, idx) => eyeTraces.reduce((acc, row) => acc + row[idx], 0) / eyeTraces.length);
        eyePlot.push({
        x: eyeTime,
        y: avg,
        mode: 'lines',
        name: 'Mean',
        line: { color: '#2563eb', width: 2 }
      });
      if (cursorTime != null && eyeHigh != null && eyeLow != null) {
        eyePlot.push({
          x: [cursorTime, cursorTime],
          y: [eyeLow, eyeHigh],
          mode: 'lines',
          name: 'Eye opening',
          line: { color: '#f97316', width: 3 },
          hoverinfo: 'skip'
        });
      }
      const layout = {
        margin: { t: 30 },
        xaxis: { title: 'Time [symbols]' },
        yaxis: { title: 'Amplitude' },
        legend: { orientation: 'h' }
      };
      if (cursorTime != null) {
        layout.shapes = [
          {
            type: 'line',
            x0: cursorTime,
            x1: cursorTime,
            y0: yMin,
            y1: yMax,
            line: { color: '#f97316', dash: 'dot', width: 1.5 }
          }
        ];
      }
      if (eyeOpening != null) {
        layout.annotations = [
          {
            x: cursorTime != null ? cursorTime : eyeTime[Math.floor(eyeTime.length / 2)] || 0,
            y: eyeHigh != null ? eyeHigh : yMax,
            xref: 'x',
            yref: 'y',
            text: `opening≈${eyeOpening.toFixed(2)}`,
            showarrow: false,
            font: { size: 11, color: '#f97316' },
            xanchor: 'left',
            yanchor: 'bottom'
          }
        ];
      }
      Plotly.newPlot('pam_eye_plot', tracesNoHover(eyePlot), withNoHover(layout), PLOT_CONFIG);
    } else {
      Plotly.purge('pam_eye_plot');
      showError('pam_eye_plot', 'Eye diagram unavailable — increase symbols or enable filtering.');
    }

    const ber = data.ber || {};
    const snrGrid = (ber.curve_snr || []).map(Number);
    const theoryCurve = (ber.curve_theory || []).map(Number);
    const simCurve = (ber.curve_sim || []).map(Number);
    const point = ber.point || {};
    const berTraces = [];
    if (snrGrid.length && theoryCurve.length) {
      berTraces.push({
        x: snrGrid,
        y: theoryCurve,
        mode: 'lines',
        name: 'Theory',
        line: { color: '#1f77b4', width: 2 }
      });
    }
    if (snrGrid.length && simCurve.length) {
      berTraces.push({
        x: snrGrid,
        y: simCurve,
        mode: 'markers+lines',
        name: 'Simulation',
        marker: { color: '#d97706', size: 7 },
        line: { color: '#d97706', width: 1 }
      });
    }
    if (point.snr_db != null && point.measured_plot != null) {
      berTraces.push({
        x: [Number(point.snr_db)],
        y: [Number(point.measured_plot)],
        mode: 'markers',
        name: 'Current',
        marker: { color: '#22c55e', size: 10 }
      });
    }
    Plotly.newPlot('pam_ber_plot', tracesNoHover(berTraces), withNoHover({
      margin: { t: 30 },
      xaxis: { title: 'Eb/N₀ [dB]' },
      yaxis: { title: 'Bit error rate', type: 'log', rangemode: 'tozero' },
      legend: { orientation: 'h' }
    }), PLOT_CONFIG);

    renderPamFacts(data.info, ber, data.challenge);
  } catch (err) {
    const msg = `M-PAM error: ${err.message || err}`;
    ['pam_constellation_plot', 'pam_eye_plot', 'pam_ber_plot'].forEach((id) => {
      Plotly.purge(id);
      showError(id, msg);
    });
  }
}


async function plotDigMod() {
  const params = collectModParams();
  try {
    const data = await fetchJSON(DIG_URLS.modulate, params);

    Plotly.newPlot('dig_mod_plot', tracesNoHover([
      { x: data.t, y: data.message, name: 'Message' },
      ...(data.carrier?.length ? [{ x: data.t, y: data.carrier, name: 'Carrier' }] : []),
      { x: data.t, y: data.modulated, name: `${params.type} Signal` }
    ]), withNoHover({
      margin: { t: 30 },
      title: `${params.type} — Modulation`,
      legend: { orientation: 'h' },
      xaxis: { title: 'Time [s]' },
      yaxis: { title: 'Amplitude' }
    }), PLOT_CONFIG);

    if ($('dig_show_spectrum').checked) {
      Plotly.newPlot('dig_mod_spec', tracesNoHover([
        { x: data.f, y: data.P_db, mode: 'lines', name: 'PSD (modulated)' }
      ]), withNoHover({
        margin: { t: 30 },
        title: 'Spectrum (Hann + rFFT)',
        xaxis: { title: 'Frequency [Hz]' },
        yaxis: { title: 'Power [dB]' }
      }), PLOT_CONFIG);
    } else {
      Plotly.purge('dig_mod_spec');
      showError('dig_mod_spec', 'Spectrum hidden');
    }

    renderDigFacts(data.info);
  } catch (err) {
    const msg = `Modulation error: ${err.message || err}`;
    showError('dig_mod_plot', msg);
    showError('dig_mod_spec', msg);
  }

}

async function plotDigDemod(){
  const params = collectDemodParams();
  try {
    const data = await fetchJSON(DIG_URLS.demod, params);
    const tDem = (data.t.length === data.demodulated.length) ? data.t : data.t.slice(1);

    Plotly.newPlot('dig_demod_plot', tracesNoHover([
      { x: data.t, y: data.modulated, name: 'Received' },
      { x: tDem, y: data.demodulated, name: 'Demodulated' }
    ]), withNoHover({
      margin: { t: 30 },
      title: `${params.type} — Demodulation`,
      legend: { orientation: 'h' },
      xaxis: { title: 'Time [s]' },
      yaxis: { title: 'Amplitude' }
    }), PLOT_CONFIG);

    if ($('dig_show_spectrum').checked) {
      Plotly.newPlot('dig_demod_spec', tracesNoHover([
        { x: data.f, y: data.P_db, mode: 'lines', name: 'PSD (demodulated)' }
      ]), withNoHover({
        margin: { t: 30 },
        title: 'Demod Spectrum',
        xaxis: { title: 'Frequency [Hz]' },
        yaxis: { title: 'Power [dB]' }
      }), PLOT_CONFIG);
    } else {
      Plotly.purge('dig_demod_spec');
      showError('dig_demod_spec', '');
    }
  } catch (err) {
    const msg = `Demodulation error: ${err.message || err}`;
    showError('dig_demod_plot', msg);
    showError('dig_demod_spec', msg);
  }
}

async function loadDigPresets() {
  try {
    const list = await fetchJSON(DIG_URLS.presets, {});
    const sel = $('dig_preset');
    if (!sel) return;
    sel.innerHTML = '<option value="">— choose preset —</option>';
    list.forEach((name) => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    });
  } catch (_) {
    // ignore silently; UI stays usable without presets
  }
}

async function applyDigPreset() {
  const sel = $('dig_preset');
  if (!sel || !sel.value) return;
  try {
    const data = await fetchJSON(DIG_URLS.modulate, { preset: sel.value });
    const info = data.info || {};
    if (info.type) {
      $('dig_mod_type').value = info.type;
      $('dig_demod_type').value = info.type;
    }
    if (info.fs != null) $('dig_fs').value = info.fs;
    if (info.duration_s != null) $('dig_t_end').value = info.duration_s;
    if (info.prf != null) $('dig_prf').value = info.prf;
    if (info.fm != null) $('dig_fm').value = info.fm;
    if (info.levels != null) $('dig_levels').value = info.levels;
    updateDigControls();
    await plotDigMod();
    await plotDigDemod();
  } catch (err) {
    showError('dig_mod_plot', `Preset error: ${err.message || err}`);
  }

}

function attachInputHandlers(ids, handler) {
  ids.forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener('input', handler);
  });
}

// ---------- init ----------
document.addEventListener('DOMContentLoaded', () => {
  ['dig_mod_type', 'dig_demod_type'].forEach((id) => {
    const el = $(id);
    if (el) {
      el.addEventListener('change', () => {
        updateDigControls();
        plotDigMod();
        plotDigDemod();
      });
    }
  });

  attachInputHandlers([
    'dig_fs',
    'dig_t_end',
    'dig_prf',
    'dig_fm',
    'dig_levels',
    'dig_snr_db'
  ], () => {
    updateDigControls();
    plotDigMod();
    plotDigDemod();
  });

  const pamModSel = $('pam_m');
  if (pamModSel) {
    pamModSel.addEventListener('change', () => {
      updatePamControls();
      plotPam();
    });
  }

  attachInputHandlers([
    'pam_rolloff',
    'pam_timing_offset',
    'pam_sps',
    'pam_span',
    'pam_snr',
    'pam_symbols'
  ], () => {
    updatePamControls();
    plotPam();
  });

  ['pam_disable_tx', 'pam_disable_rx'].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener('change', plotPam);
  });

  const passScheme = $('pass_scheme');
  if (passScheme) {
    passScheme.addEventListener('change', () => {
      plotPassband();
    });
  }

  attachInputHandlers([
    'pass_carrier',
    'pass_symbol_rate',
    'pass_offset',
    'pass_phase',
    'pass_sps'
  ], () => {
    updatePassbandControls();
    plotPassband();
  });

  const passCostas = $('pass_costas');
  if (passCostas) passCostas.addEventListener('change', plotPassband);


  const snrToggle = $('dig_snr_toggle');
  if (snrToggle) snrToggle.addEventListener('change', plotDigMod);

  const specToggle = $('dig_show_spectrum');
  if (specToggle) specToggle.addEventListener('change', () => {
    plotDigMod();
    plotDigDemod();
  });

  const presetSel = $('dig_preset');
  if (presetSel) presetSel.addEventListener('change', applyDigPreset);

  loadDigPresets();
  updateDigControls();
  updatePamControls();
  updatePassbandControls();
  plotDigMod();
  plotDigDemod();
  plotPam();
  plotPassband();


  window.addEventListener('resize', () => {
    [
      'dig_mod_plot',
      'dig_demod_plot',
      'dig_mod_spec',
      'dig_demod_spec',
      'pam_constellation_plot',
      'pam_eye_plot',
      'pam_ber_plot',
      'passband_wave_plot',
      'passband_iq_plot',
      'passband_constellation_plot'
    ].forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      try {
        Plotly.Plots.resize(el);
      } catch (_) {
        /* ignore */
      }
    });
  });
});

