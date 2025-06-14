// static/js/digital_modulation.js

// reflect slider label & live‐update both plots
function updateSliderValue(id) {
    const el = document.getElementById(id);
    document.getElementById(id + '_val').innerText = el.value;
    plotDigital();       // update modulation plot
    plotDigital(true);   // update demodulation plot
  }
  
  // show/hide deviation control (FSK only) & live‐plot on scheme change
  function updateDigControls() {
    const type = document.getElementById('dig_type').value;
    document.getElementById('dev_group').style.display =
      (type === 'FSK' ? '' : 'none');
    plotDigital();     // replot modulation
  }
  
// simple time vector helper used for local computations
function makeTime(tEnd = 1.0, fs = 1000) {
  const N = Math.floor(fs * tEnd);
  return Array.from({ length: N }, (_, i) => i / fs);
}

// client-side modulation implementation mirroring the former API
function digitalModulate({ type, fc, br, dev }) {
  const t = makeTime();
  const bitPeriod = 1 / br;
  const bits = t.map(v => (v % bitPeriod < bitPeriod / 2 ? 1 : 0));
  let modulated;

  if (type === 'ASK') {
    modulated = t.map((v, i) => bits[i] * Math.cos(2 * Math.PI * fc * v));
  } else if (type === 'PSK') {
    modulated = t.map((v, i) =>
      Math.cos(2 * Math.PI * fc * v + Math.PI * (1 - bits[i])));
  } else if (type === 'FSK') {
    const f0 = fc - dev;
    const f1 = fc + dev;
    modulated = t.map((v, i) =>
      Math.cos(2 * Math.PI * (f0 + (f1 - f0) * bits[i]) * v));
  } else {
    throw new Error('Unknown modulation type');
  }
  
  return { t, bits, modulated };
}

// very basic demodulation example – echoes the bits like the server did
function digitalDemodulate({ br }) {
  const t = makeTime();
  const bitPeriod = 1 / br;
  const bits = t.map(v => (v % bitPeriod < bitPeriod / 2 ? 1 : 0));
  return { t, modulated: [...bits], demodulated: bits };
}
  
  // draw either modulation or demodulation
  function plotDigital(demod=false) {
    const type = demod
      ? document.getElementById('dig_demod').value
      : document.getElementById('dig_type').value;
  
    const params = {
      type,
      fc:  +document.getElementById('dig_fc').value,
      br:  +document.getElementById('dig_br').value,
      dev: +document.getElementById('dig_dev').value
    };

    const data = demod
      ? digitalDemodulate(params)
      : digitalModulate(params);
  
    if (!demod) {
      Plotly.newPlot('dig_plot', [
        { x: data.t, y: data.bits,      name: 'Bits',  yaxis: 'y2' },
        { x: data.t, y: data.modulated, name: `${type} Signal` }
      ], {
        margin: { t: 30 },
        title:  `${type} Modulation`,
        yaxis:  { title: 'Signal' },
        yaxis2: {
          overlaying: 'y', side: 'right',
          showgrid: false,
          title: 'Bits',
          range: [-0.2,1.2]
        }
      });
    } else {
      Plotly.newPlot('dig_demod_plot', [
        { x: data.t,        y: data.modulated,   name: 'Received' },
        { x: data.t,        y: data.demodulated, name: 'Demodulated Bits', yaxis: 'y2' }
      ], {
        margin: { t: 30 },
        title:  `${type} Demodulation`,
        yaxis:  { title: 'Signal' },
        yaxis2: {
          overlaying: 'y', side: 'right',
          showgrid: false,
          title: 'Bits',
          range: [-0.2,1.2]
        }
      });
    }
  }
  
  // wire up live‐update events on load
  document.addEventListener('DOMContentLoaded', () => {
    // sliders
    ['dig_fc','dig_br','dig_dev'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('input', () => updateSliderValue(id));
      }
    });
  
    // modulation scheme dropdown
    document.getElementById('dig_type')
      .addEventListener('change', () => {
        updateDigControls();
        // sync labels & plots
        ['dig_fc','dig_br','dig_dev'].forEach(updateSliderValue);
      });
  
    // demodulation dropdown
    document.getElementById('dig_demod')
      .addEventListener('change', () => plotDigital(true));
  
    // initial setup
    updateDigControls();
    ['dig_fc','dig_br','dig_dev'].forEach(updateSliderValue);
    plotDigital();
    plotDigital(true);
  });
  