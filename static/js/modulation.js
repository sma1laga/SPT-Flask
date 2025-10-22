// static/js/modulation.js
window.MOD_URLS = window.MOD_URLS || {
  presets: '/modulation/api/presets',
  modulate: '/modulation/api/modulate',
  demod: '/modulation/api/demodulate'
};

function $(id){ return document.getElementById(id); }
function setLabel(id){ const el=$(id), lab=$(id+'_val'); if(el&&lab) lab.innerText = el.value; }

async function fetchJSON(url, params) {
  const q = new URLSearchParams(params).toString();
  const res = await fetch(url + '?' + q);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function updateModControls() {
  const type = $('mod_type').value;
  ['am_controls','fm_controls','qam_controls'].forEach(i=>{
    const e = $(i); if (e) e.style.display = 'none';
  });

  if (type === 'AM') $('am_controls').style.display = '';
  else if (type === 'FM' || type === 'PM') {
    $('fm_controls').style.display = '';
    $('fm_header').innerText = `${type} Modulation`;
    $('beta_label').innerText = (type==='FM') ? 'β (mod index):' : 'Phase index m:';

  }
  else if (type === 'QAM') {
    $('qam_controls').style.display = '';
  }

  [
    'am_fc','am_fm','am_m',
    'fm_fc','fm_fm','fm_beta',
    'qam_fc','qam_fm','qam_i_amp','qam_q_amp','qam_phase_error',
    'fs','t_end','snr_db'
  ].forEach(setLabel);
}

function renderFacts(info){
  if (!info) return;
  const parts = [];
  if (info.type==='AM'){
    parts.push(`m = ${(+info.m).toFixed(2)}` + (info.overmod ? ' (overmod ⚠️)' : ''));
    parts.push(`fc = ${info.fc} Hz, fm = ${info.fm} Hz`);
    if (info.carrier_mode){
      parts.push(info.carrier_mode === 'without' ? 'mode: without carrier' : 'mode: with carrier');
    }
  } else if (info.type==='FM'){
    parts.push(`β = ${(+info.beta).toFixed(2)}, fc = ${info.fc} Hz, fm = ${info.fm} Hz`);
    parts.push(`Carson BW ≈ ${info.carson_bw_hz.toFixed(1)} Hz`);
  } else if (info.type==='PM'){
    parts.push(`phase index m = ${(+info.phase_index).toFixed(2)}`);
    parts.push(`fc = ${info.fc} Hz, fm = ${info.fm} Hz`);
  } else if (info.type==='QAM') {
    parts.push(`fc = ${info.fc} Hz, fm = ${info.fm} Hz`);
    if (info.i_amp != null) parts.push(`Iₘax = ${(+info.i_amp).toFixed(2)}`);
    if (info.q_amp != null) parts.push(`Qₘax = ${(+info.q_amp).toFixed(2)}`);
  }
  if (Number.isFinite(+info.snr_db)) parts.push(`SNR = ${(+info.snr_db).toFixed(1)} dB`);
  $('facts').innerText = parts.join('  •  ');
}

function renderSummary(info){
  const wrap = $('snr_summary');
  if (!wrap) return;
  const rows = info?.snr_summary;
  if (!rows || !rows.length){
    wrap.innerHTML = '<div>No AM SNR summary available for this mode.</div>';
    return;
  }

  const body = rows.map((row) => {
    const classes = row.active ? 'active' : '';
    const syncText = row.requires_sync ? '<div class="muted">needs synchronous demod</div>' : '';
    return `<tr class="${classes}">
      <th scope="row">${row.scheme}${syncText}</th>
      <td>${row.snr_formula}</td>
      <td>${row.nf_formula}<div class="muted">NF=${row.nf_value} (${row.nf_db})</div></td>
      <td>${row.eta_formula}<div class="muted">η=${row.eta_value}</div></td>
      <td><div>SNR<sub>out</sub>: ${row.snr_out_db}</div><div class="muted">factor ×${row.snr_factor}</div></td>
    </tr>`;
  }).join('');

  wrap.classList.remove('muted');
  wrap.innerHTML = `<table>
    <thead>
      <tr>
        <th>Scheme</th>
        <th>SNR formula</th>
        <th>NF</th>
        <th>Efficiency</th>
        <th>Current values</th>
      </tr>
    </thead>
    <tbody>${body}</tbody>
  </table>`;

  if (window.MathJax?.typesetPromise){
    MathJax.typesetPromise([wrap]);
  }
}
let modRequestId = 0;
let demodRequestId = 0;

async function plotMod() {
  const type = $('mod_type').value;
  const requestId = ++modRequestId;


  const params = {
    type,
    fs: +$('fs').value,
    t_end: +$('t_end').value,
    snr_db: $('snr_toggle').checked ? +$('snr_db').value : Infinity
  };

  if (type === 'AM') {
    params.fc = +$('am_fc').value;
    params.fm = +$('am_fm').value;
    params.m  = +$('am_m').value;
    params.carrier_mode = $('am_carrier').value;
  } else if (type==='FM' || type==='PM') {
    params.fc   = +$('fm_fc').value;
    params.fm   = +$('fm_fm').value;
    params.beta = +$('fm_beta').value;
    if (type==='PM'){ params.m = params.beta; delete params.beta; }
  } else if (type === 'QAM') {
    params.fc = +$('qam_fc').value;
    params.fm = +$('qam_fm').value;
    params.i_amp = +$('qam_i_amp').value;
    params.q_amp = +$('qam_q_amp').value;
  }

  const data = await fetchJSON(MOD_URLS.modulate, params);

  if (requestId !== modRequestId) {
    return;
  }

  const plots = $('plots_grid');
  if (plots) {
    const demodType = $('demod_type')?.value;
    plots.classList.toggle('is-wide', type === 'QAM' || demodType === 'QAM');
  }

  let traces = [];
  if (type === 'QAM') {
    traces = [
      { x: data.t, y: data.modulated, name: 'QAM passband' }
    ];
    if (Array.isArray(data.message_i)) {
      traces.push({ x: data.t, y: data.message_i, name: 'I(t) (baseband)', line: { dash: 'dot' } });
    }
    if (Array.isArray(data.message_q)) {
      traces.push({ x: data.t, y: data.message_q, name: 'Q(t) (baseband)', line: { dash: 'dot' } });
    }
    window.__QAM_LAST_MOD = data;
  } else {
    traces = [
      { x: data.t, y: data.message,   name: 'Message' },
      ...(data.carrier?.length ? [{ x: data.t, y: data.carrier, name: 'Carrier' }] : []),
      { x: data.t, y: data.modulated, name: `${type} Signal` }
    ];
    window.__QAM_LAST_MOD = null;
  }

  Plotly.newPlot('mod_plot', traces, {
    margin: { t: 30 },
    title: type === 'QAM' ? 'QAM — Passband view' : `${type} — Modulation`,
    legend: { orientation: 'h' },
    xaxis: { title: 'Time [s]' },
    yaxis: { title: 'Amplitude' }
  }, {responsive: true});

  if ($('show_spectrum').checked) {
    Plotly.newPlot('spec_plot', [
      { x: data.f, y: data.P_db, type: 'scatter', mode: 'lines', name: 'PSD (modulated)' }
    ], {
      margin: { t: 30 },
      title: 'Spectrum (Hann + rFFT)',
      xaxis: { title: 'Frequency [Hz]' },
      yaxis: { title: 'Power [dB]' }
    }, {responsive: true});
  } else {
    Plotly.purge('spec_plot');
    $('spec_plot').innerHTML = '<div class="muted">Spectrum hidden</div>';
  }

  renderFacts(data.info);
  renderSummary(data.info);
}

async function plotDemod() {
  const type = $('demod_type').value;
  const requestId = ++demodRequestId;

  const params = {
    type,
    fs: +$('fs').value,
    t_end: +$('t_end').value
  };

  if (type === 'AM') {
    params.fc = +$('am_fc').value; params.fm = +$('am_fm').value; params.m = +$('am_m').value;
    params.carrier_mode = $('am_carrier').value;
  } else if (type==='FM' || type==='PM') {
    params.fc = +$('fm_fc').value; params.fm = +$('fm_fm').value;
    if (type==='FM') params.beta = +$('fm_beta').value; else params.m = +$('fm_beta').value;
  } else if (type === 'QAM') {
    params.fc = +$('qam_fc').value;
    params.fm = +$('qam_fm').value;
    params.i_amp = +$('qam_i_amp').value;
    params.q_amp = +$('qam_q_amp').value;
    params.phase_error_deg = +$('qam_phase_error').value;
    params.snr_db = $('snr_toggle').checked ? +$('snr_db').value : Infinity;
  }

  const data = await fetchJSON(MOD_URLS.demod, params);

  if (requestId !== demodRequestId) {
    return;
  }

  const plots = $('plots_grid');
  if (plots) {
    const modType = $('mod_type')?.value;
    plots.classList.toggle('is-wide', type === 'QAM' || modType === 'QAM');
  }

  if (type === 'QAM') {
    const showMixers = $('qam_show_mixers')?.checked;
    const phaseErr = Number.isFinite(Number(data.phase_error_deg)) ? Number(data.phase_error_deg) : Number(params.phase_error_deg || 0);
    const phaseLabel = Number.isFinite(phaseErr) ? `${phaseErr.toFixed(0)}°` : `${phaseErr}`;

    const basebandTraces = [
      { x: data.t, y: data.baseband_i, name: 'I(t)', line: { color: '#2563eb' } },
      { x: data.t, y: data.baseband_q, name: 'Q(t)', line: { color: '#f97316' } }
    ];
    if (showMixers && Array.isArray(data.mix_i) && Array.isArray(data.mix_q)) {
      basebandTraces.push({ x: data.t, y: data.mix_i, name: 'Mixer ×cos', line: { dash: 'dot', color: '#1f2937' }, opacity: 0.6 });
      basebandTraces.push({ x: data.t, y: data.mix_q, name: 'Mixer ×sin', line: { dash: 'dot', color: '#4b5563' }, opacity: 0.6 });
    }

    const magPhaseTraces = [
      { x: data.t, y: data.magnitude, name: '|s_bb(t)|', xaxis: 'x2', yaxis: 'y2', line: { color: '#111827' } },
      { x: data.t, y: data.phase_deg, name: '∠s_bb(t) [deg]', xaxis: 'x2', yaxis: 'y3', line: { color: '#10b981' } }
    ];

    const demodLayout = {
      margin: { t: 30 },
      title: `QAM — ECB view (phase error ${phaseLabel})`,
      legend: { orientation: 'h' },
      grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
      xaxis: { title: 'Time [s]' },
      yaxis: { title: 'Amplitude' },
      xaxis2: { title: 'Time [s]' },
      yaxis2: { title: '|s_bb(t)|' },
      yaxis3: { title: 'Phase [deg]', overlaying: 'y2', side: 'right' }
    };

    Plotly.newPlot('demod_plot', [...basebandTraces, ...magPhaseTraces], demodLayout, { responsive: true });

    const passbandTraces = [
      { x: data.t, y: data.modulated, name: 'QAM passband' }
    ];
    if (window.__QAM_LAST_MOD?.message_i) {
      passbandTraces.push({ x: data.t, y: window.__QAM_LAST_MOD.message_i, name: 'I(t) (baseband)', line: { dash: 'dot' } });
    }
    if (window.__QAM_LAST_MOD?.message_q) {
      passbandTraces.push({ x: data.t, y: window.__QAM_LAST_MOD.message_q, name: 'Q(t) (baseband)', line: { dash: 'dot' } });
    }
    if (showMixers && Array.isArray(data.lo_cos) && Array.isArray(data.lo_sin)) {
      passbandTraces.push({ x: data.t, y: data.lo_cos, name: 'LO cos', line: { dash: 'dash', color: '#9ca3af' } });
      passbandTraces.push({ x: data.t, y: data.lo_sin, name: 'LO sin', line: { dash: 'dash', color: '#6b7280' } });
    }
    Plotly.newPlot('mod_plot', passbandTraces, {
      margin: { t: 30 },
      title: `QAM — Passband view (phase error ${phaseLabel})`,
      legend: { orientation: 'h' },
      xaxis: { title: 'Time [s]' },
      yaxis: { title: 'Amplitude' }
    }, { responsive: true });

    if ($('show_spectrum').checked) {
      Plotly.newPlot('spec_demod_plot', [
        { x: data.f, y: data.P_db, type: 'scatter', mode: 'lines', name: 'PSD (I baseband)' }
      ], {
        margin: { t: 30 },
        title: 'Demod Spectrum (I-channel)',
        xaxis: { title: 'Frequency [Hz]' },
        yaxis: { title: 'Power [dB]' }
      }, { responsive: true });
    } else {
      Plotly.purge('spec_demod_plot');
      $('spec_demod_plot').innerHTML = '';
    }

    return;
  }
  const tDem = (data.t.length === data.demodulated.length) ? data.t : data.t.slice(1);

  Plotly.newPlot('demod_plot', [
    { x: data.t,   y: data.modulated,   name: 'Received' },
    { x: tDem,     y: data.demodulated, name: 'Demodulated' }
  ], {
    margin: { t: 30 },
    title: `${type} — Demodulation`,
    legend: { orientation: 'h' },
    xaxis: { title: 'Time [s]' },
    yaxis: { title: 'Amplitude' }
  }, {responsive: true});

  if ($('show_spectrum').checked) {
    Plotly.newPlot('spec_demod_plot', [
      { x: data.f, y: data.P_db, type: 'scatter', mode: 'lines', name: 'PSD (demodulated)' }
    ], {
      margin: { t: 30 },
      title: 'Demod Spectrum',
      xaxis: { title: 'Frequency [Hz]' },
      yaxis: { title: 'Power [dB]' }
    }, {responsive: true});
  } else {
    Plotly.purge('spec_demod_plot');
    $('spec_demod_plot').innerHTML = '';
  }
}

async function loadPresets(){
  try {
    const list = await fetchJSON(MOD_URLS.presets, {});
    const sel = $('preset');
    sel.innerHTML = '<option value="">— choose preset —</option>';
    list.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p; opt.textContent = p;
      sel.appendChild(opt);
    });
  } catch(e) {/* ok */}
}

async function applyPreset(){
  const p = $('preset').value;
  if (!p) return;
  const data = await fetchJSON(MOD_URLS.modulate, { preset: p });
  if (data.info){
    const I = data.info;
    $('mod_type').value = I.type;
    $('demod_type').value = I.type;
    if (I.fs)   $('fs').value = I.fs;
    if (I.duration_s) $('t_end').value = I.duration_s;
    if (I.fc)   { $('am_fc').value = I.fc; $('fm_fc').value = I.fc; }
    if (I.fm)   { $('am_fm').value = I.fm; $('fm_fm').value = I.fm; }
    if (I.m!=null)    $('am_m').value = I.m;
    if (I.phase_index!=null) $('fm_beta').value = I.phase_index;
    if (I.beta!=null) $('fm_beta').value = I.beta;
    if (I.carrier_mode){ $('am_carrier').value = I.carrier_mode; }
    updateModControls();
  }
  plotMod(); plotDemod();
}

document.addEventListener('DOMContentLoaded', () => {
  $('mod_type').addEventListener('change', ()=>{ updateModControls(); plotMod(); });
  $('demod_type').addEventListener('change', plotDemod);

  [
    'am_fc','am_fm','am_m',
    'fm_fc','fm_fm','fm_beta',
    'qam_fc','qam_fm','qam_i_amp','qam_q_amp',
    'fs','t_end','snr_db'
  ].forEach(id=>{
    const el = $(id); if (el) {
      el.addEventListener('input', ()=>{ setLabel(id); plotMod(); plotDemod(); });
    }
  });

  const amCarrier = $('am_carrier');
  if (amCarrier){
    amCarrier.addEventListener('change', ()=>{ plotMod(); plotDemod(); });
  }

  const qamPhase = $('qam_phase_error');
  if (qamPhase) {
    qamPhase.addEventListener('input', ()=>{ setLabel('qam_phase_error'); plotDemod(); });
  }

  const qamMixers = $('qam_show_mixers');
  if (qamMixers) {
    qamMixers.addEventListener('change', ()=>{ plotDemod(); });
  }

  $('snr_toggle').addEventListener('change', ()=>{ plotMod(); plotDemod(); });
  $('show_spectrum').addEventListener('change', ()=>{ plotMod(); plotDemod(); });

  loadPresets();
  $('preset').addEventListener('change', applyPreset);

  updateModControls();
  plotMod();
  plotDemod();

  // ensure reflow in tight layouts
  window.addEventListener('resize', () => {
    Plotly.Plots.resize('mod_plot');
    Plotly.Plots.resize('demod_plot');
    Plotly.Plots.resize('spec_plot');
    Plotly.Plots.resize('spec_demod_plot');
  });
});
