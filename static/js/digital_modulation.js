// static/js/digital_modulation.js

window.DIG_URLS = window.DIG_URLS || {
  presets: '/digital_modulation/api/presets',
  modulate: '/digital_modulation/api/modulate',
  demod: '/digital_modulation/api/demodulate'
};

function $(id){ return document.getElementById(id); }


function setLabel(id){
  const el = $(id), lab = $(id + '_val');
  if (el && lab) lab.innerText = el.value;
}

async function fetchJSON(url, params){
  const q = new URLSearchParams(params).toString();
  const res = await fetch(url + '?' + q);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function updateDigControls(){
  const type = $('dig_mod_type').value;
  const levelsGroup = $('dig_pcm_levels_group');
  if (levelsGroup) levelsGroup.style.display = (type === 'PCM') ? '' : 'none';

  [
    'dig_fs','dig_t_end','dig_prf','dig_fm','dig_levels','dig_snr_db'
  ].forEach(setLabel);
}

function renderDigFacts(info){
  if (!info) return;
  const parts = [];
  parts.push(`${info.type} @ fs=${info.fs} Hz`);
  if (info.prf) parts.push(`prf = ${info.prf} Hz`);
  if (info.fm) parts.push(`fm = ${info.fm} Hz`);
  if (info.levels) parts.push(`${info.levels} levels`);
  if (Number.isFinite(+info.snr_db)) parts.push(`SNR = ${(+info.snr_db).toFixed(1)} dB`);
  const el = $('dig_facts');
  if (el) el.innerText = parts.join('  •  ');
}

function collectModParams(){
  const type = $('dig_mod_type').value;
  const params = {
    type,
    fs: +$('dig_fs').value,
    t_end: +$('dig_t_end').value,
    snr_db: $('dig_snr_toggle').checked ? +$('dig_snr_db').value : Infinity
  };
  params.prf = +$('dig_prf').value;
  params.fm = +$('dig_fm').value;
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
async function plotDigMod(){
  const params = collectModParams();
  const data = await fetchJSON(DIG_URLS.modulate, params);

  Plotly.newPlot('dig_mod_plot', [
    { x: data.t, y: data.message, name: 'Message' },
    ...(data.carrier?.length ? [{ x: data.t, y: data.carrier, name: 'Carrier' }] : []),
    { x: data.t, y: data.modulated, name: `${params.type} Signal` }
  ], {
    margin: { t: 30 },
    title: `${params.type} — Modulation`,
    legend: { orientation: 'h' }
  }, {responsive: true});

  if ($('dig_show_spectrum').checked) {
    Plotly.newPlot('dig_mod_spec', [
      { x: data.f, y: data.P_db, mode: 'lines', name: 'PSD (modulated)' }
    ], {
      margin: { t: 30 },
      title: 'Spectrum (Hann + rFFT)',
      xaxis: { title: 'Frequency [Hz]' },
      yaxis: { title: 'Power [dB]' }
    }, {responsive: true});

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
    Plotly.purge('dig_mod_spec');
    const div = $('dig_mod_spec');
    if (div) div.innerHTML = '<div class="muted">Spectrum hidden</div>';
  }


  renderDigFacts(data.info);
}

async function plotDigDemod(){
  const params = collectDemodParams();
  const data = await fetchJSON(DIG_URLS.demod, params);
  const tDem = (data.t.length === data.demodulated.length) ? data.t : data.t.slice(1);

  Plotly.newPlot('dig_demod_plot', [
    { x: data.t, y: data.modulated, name: 'Received' },
    { x: tDem, y: data.demodulated, name: 'Demodulated' }
  ], {
    margin: { t: 30 },
    title: `${params.type} — Demodulation`,
    legend: { orientation: 'h' }
  }, {responsive: true});

  if ($('dig_show_spectrum').checked) {
    Plotly.newPlot('dig_demod_spec', [
      { x: data.f, y: data.P_db, mode: 'lines', name: 'PSD (demodulated)' }
    ], {
      margin: { t: 30 },
      title: 'Demod Spectrum',
      xaxis: { title: 'Frequency [Hz]' },
      yaxis: { title: 'Power [dB]' }
    }, {responsive: true});
  } else {
    Plotly.purge('dig_demod_spec');
    const div = $('dig_demod_spec');
    if (div) div.innerHTML = '';
  }
}

async function applyDigPreset(){
  const p = $('dig_preset').value;
  if (!p) return;
  const data = await fetchJSON(DIG_URLS.modulate, { preset: p });
  if (data.info){
    const I = data.info;
    $('dig_mod_type').value = I.type;
    $('dig_demod_type').value = I.type;
    if (I.fs) $('dig_fs').value = I.fs;
    if (I.duration_s) $('dig_t_end').value = I.duration_s;
    if (I.prf) $('dig_prf').value = I.prf;
    if (I.fm) $('dig_fm').value = I.fm;
    if (I.levels) $('dig_levels').value = I.levels;
    updateDigControls();
  }
  plotDigMod();
  plotDigDemod();
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
  ['dig_mod_type','dig_demod_type'].forEach(id => {
    const el = $(id); if (el) el.addEventListener('change', () => {
      updateDigControls();
      plotDigMod();
      plotDigDemod();
    });
  });

  $('dig_snr_toggle').addEventListener('change', () => {
    plotDigMod();
  });

  $('dig_show_spectrum').addEventListener('change', () => {
    plotDigMod();
    plotDigDemod();
  });

  loadDigPresets();
  $('dig_preset').addEventListener('change', applyDigPreset);
  });

  const demodSel = getEl('dig_demod');
  if (demodSel) demodSel.addEventListener('change', () => plotDigital(true));

  // first draw
  updateDigControls();
  plotDigMod();
  plotDigDemod();

  window.addEventListener('resize', () => {
    ['dig_mod_plot','dig_demod_plot','dig_mod_spec','dig_demod_spec'].forEach(id => {
      try { Plotly.Plots.resize(id); } catch (_) {}
    });
  });
});

