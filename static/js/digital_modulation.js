// static/js/digital_modulation.js

window.DIG_URLS = window.DIG_URLS || {
  presets: '/digital_modulation/api/presets',
  modulate: '/digital_modulation/api/modulate',
  demod: '/digital_modulation/api/demodulate'
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
  plotDigMod();
  plotDigDemod();

  window.addEventListener('resize', () => {
    ['dig_mod_plot', 'dig_demod_plot', 'dig_mod_spec', 'dig_demod_spec'].forEach((id) => {
      try { Plotly.Plots.resize(id); } catch (_) { /* ignore */ }
    });
  });
});

