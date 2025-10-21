// static/js/digital_modulation.js

window.DIG_URLS = window.DIG_URLS || {
  presets: '/digital_modulation/api/presets',
  modulate: '/digital_modulation/api/modulate',
  demod: '/digital_modulation/api/demodulate',
  mpam: '/digital_modulation/api/m_pam'
};

const $ = (id) => document.getElementById(id);
const setLabel = (id) => {
  const el = $(id);
  const lab = $(id + '_val');
  if (el && lab) lab.innerText = el.value;
};

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
    'pam_sps',
    'pam_span',
    'pam_snr',
    'pam_symbols'
  ].forEach(setLabel);
}

function collectPamParams() {
  const orderEl = $('pam_m');
  const rolloffEl = $('pam_rolloff');
  const spsEl = $('pam_sps');
  const spanEl = $('pam_span');
  const snrEl = $('pam_snr');
  const symEl = $('pam_symbols');
  if (!orderEl || !rolloffEl || !spsEl || !spanEl || !snrEl || !symEl) {
    return {};
  }
  const disableTx = $('pam_disable_tx')?.checked;
  const disableRx = $('pam_disable_rx')?.checked;
  return {
    M: +orderEl.value,
    rolloff: +rolloffEl.value,
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

function renderPamFacts(info, ber) {
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
    Plotly.newPlot('pam_constellation_plot', traces, {
      margin: { t: 30 },
      xaxis: { title: 'Amplitude', zeroline: false },
      yaxis: { showticklabels: false, showgrid: false, zeroline: false, range: [-0.12, 0.12] },
      legend: { orientation: 'h' }
    }, { responsive: true });

    const eye = data.eye || {};
    const eyeTime = Array.isArray(eye.time) ? eye.time.map(Number) : [];
    const eyeTraces = Array.isArray(eye.traces) ? eye.traces.map((row) => row.map(Number)) : [];
    if (eyeTime.length && eyeTraces.length) {
      const eyePlot = eyeTraces.map((row) => ({
        x: eyeTime,
        y: row,
        mode: 'lines',
        line: { color: 'rgba(59, 130, 246, 0.25)', width: 1 },
        hoverinfo: 'skip',
        showlegend: false
      }));
      const avg = eyeTime.map((_, idx) => eyeTraces.reduce((acc, row) => acc + row[idx], 0) / eyeTraces.length);
      eyePlot.push({
        x: eyeTime,
        y: avg,
        mode: 'lines',
        name: 'Mean',
        line: { color: '#2563eb', width: 2 }
      });
      Plotly.newPlot('pam_eye_plot', eyePlot, {
        margin: { t: 30 },
        xaxis: { title: 'Time [symbols]' },
        yaxis: { title: 'Amplitude' }
      }, { responsive: true });
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
    Plotly.newPlot('pam_ber_plot', berTraces, {
      margin: { t: 30 },
      xaxis: { title: 'Eb/N₀ [dB]' },
      yaxis: { title: 'Bit error rate', type: 'log', rangemode: 'tozero' },
      legend: { orientation: 'h' }
    }, { responsive: true });

    renderPamFacts(data.info, ber);
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

    Plotly.newPlot('dig_mod_plot', [
      { x: data.t, y: data.message, name: 'Message' },
      ...(data.carrier?.length ? [{ x: data.t, y: data.carrier, name: 'Carrier' }] : []),
      { x: data.t, y: data.modulated, name: `${params.type} Signal` }
    ], {
      margin: { t: 30 },
      title: `${params.type} — Modulation`,
      legend: { orientation: 'h' },
      xaxis: { title: 'Time [s]' },
      yaxis: { title: 'Amplitude' }
    }, { responsive: true });

    if ($('dig_show_spectrum').checked) {
      Plotly.newPlot('dig_mod_spec', [
        { x: data.f, y: data.P_db, mode: 'lines', name: 'PSD (modulated)' }
      ], {
        margin: { t: 30 },
        title: 'Spectrum (Hann + rFFT)',
        xaxis: { title: 'Frequency [Hz]' },
        yaxis: { title: 'Power [dB]' }
      }, { responsive: true });
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

    Plotly.newPlot('dig_demod_plot', [
      { x: data.t, y: data.modulated, name: 'Received' },
      { x: tDem, y: data.demodulated, name: 'Demodulated' }
    ], {
      margin: { t: 30 },
      title: `${params.type} — Demodulation`,
      legend: { orientation: 'h' },
      xaxis: { title: 'Time [s]' },
      yaxis: { title: 'Amplitude' }
    }, { responsive: true });

    if ($('dig_show_spectrum').checked) {
      Plotly.newPlot('dig_demod_spec', [
        { x: data.f, y: data.P_db, mode: 'lines', name: 'PSD (demodulated)' }
      ], {
        margin: { t: 30 },
        title: 'Demod Spectrum',
        xaxis: { title: 'Frequency [Hz]' },
        yaxis: { title: 'Power [dB]' }
      }, { responsive: true });
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
  plotDigMod();
  plotDigDemod();
  plotPam();


  window.addEventListener('resize', () => {
    ['dig_mod_plot', 'dig_demod_plot', 'dig_mod_spec', 'dig_demod_spec', 'pam_constellation_plot', 'pam_eye_plot', 'pam_ber_plot'].forEach((id) => {
      try { Plotly.Plots.resize(id); } catch (_) { /* ignore */ }
    });
  });
});

