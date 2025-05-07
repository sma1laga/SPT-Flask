// update slider label & replot
function updateSliderValue(id) {
    const el = document.getElementById(id);
    document.getElementById(id + '_val').innerText = el.value;
    plotMod();
    plotDemod();
  }
  
  function updateModControls() {
    const type = document.getElementById('mod_type').value;
    // hide all three
    ['am_controls','fm_controls','pulse_controls'].forEach(i=>{
      document.getElementById(i).style.display = 'none';
    });
  
    if (type === 'AM') {
      document.getElementById('am_controls').style.display = '';
    }
    else if (type === 'FM' || type === 'PM') {
      document.getElementById('fm_controls').style.display = '';
      document.getElementById('fm_header').innerText = `${type} Modulation`;
      document.getElementById('beta_label').innerText =
        type==='FM' ? 'Beta (Hz dev):' : 'Phase Index:';
    }
    else {
      // PAM, PWM, PPM, PCM
      document.getElementById('pulse_controls').style.display = '';
      document.getElementById('pulse_header').innerText = `${type} Modulation`;
      document.getElementById('pcm_levels_group').style.display =
        type==='PCM' ? '' : 'none';
    }
  
    // sync current slider labels
    ['am_fc','am_fm','am_m','fm_fc','fm_fm','fm_beta',
     'pm_prf','pm_fm','pm_levels']
      .forEach(id => {
        if (document.getElementById(id))
          document.getElementById(id + '_val').innerText =
            document.getElementById(id).value;
      });
  }
  
  async function fetchJSON(url, params) {
    const q = new URLSearchParams(params).toString();
    const res = await fetch(url + '?' + q);
    return res.json();
  }
  
  async function plotMod() {
    const type = document.getElementById('mod_type').value;
    let params = { type };
  
    if (type === 'AM') {
      params.fc = +document.getElementById('am_fc').value;
      params.fm = +document.getElementById('am_fm').value;
      params.m  = +document.getElementById('am_m').value;
    }
    else if (type==='FM' || type==='PM') {
      params.fc   = +document.getElementById('fm_fc').value;
      params.fm   = +document.getElementById('fm_fm').value;
      params.beta = +document.getElementById('fm_beta').value;
    }
    else {
      // pulse
      params.prf = +document.getElementById('pm_prf').value;
      params.fm  = +document.getElementById('pm_fm').value;
      if (type==='PCM')
        params.levels = +document.getElementById('pm_levels').value;
    }
  
    const data   = await fetchJSON('/modulation/api/modulate', params);
    let traces;
  
    if (['AM','FM','PM'].includes(type)) {
      traces = [
        { x: data.t, y: data.message,   name: 'Message' },
        ...(data.carrier.length
          ? [{ x: data.t, y: data.carrier, name: 'Carrier' }]
          : []),
        { x: data.t, y: data.modulated, name: `${type} Signal` }
      ];
    } else {
      traces = [
        { x: data.t, y: data.message,   name: 'Message' },
        { x: data.t, y: data.modulated, name: `${type} Signal` }
      ];
    }
  
    Plotly.newPlot('mod_plot', traces, {
      margin: { t: 30 },
      title: `${type} Modulation`
    });
  }
  
  async function plotDemod() {
    const type = document.getElementById('demod_type').value;
    let params = { type };
  
    if (type === 'AM') {
      params.fc = +document.getElementById('am_fc').value;
      params.fm = +document.getElementById('am_fm').value;
      params.m  = +document.getElementById('am_m').value;
    }
    else if (type==='FM' || type==='PM') {
      params.fc   = +document.getElementById('fm_fc').value;
      params.fm   = +document.getElementById('fm_fm').value;
      params.beta = +document.getElementById('fm_beta').value;
    }
    else {
      // pulse
      params.prf = +document.getElementById('pm_prf').value;
      params.fm  = +document.getElementById('pm_fm').value;
      if (type==='PCM')
        params.levels = +document.getElementById('pm_levels').value;
    }
  
    const data = await fetchJSON('/modulation/api/demodulate', params);
    const tDem = data.t.length === data.demodulated.length
                 ? data.t
                 : data.t.slice(1);
  
    Plotly.newPlot('demod_plot', [
      { x: data.t,   y: data.modulated,   name: 'Received Signal' },
      { x: tDem,     y: data.demodulated, name: 'Demodulated'    }
    ], {
      margin: { t: 30 },
      title: `${type} Demodulation`
    });
  }
  
  // on‐load wiring
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('demod_type').addEventListener('change', plotDemod);
    updateModControls();
    plotMod();
    plotDemod();
  });
  