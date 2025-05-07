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
  
  // helper to fetch JSON from Flask API
  async function fetchJSON(url, params) {
    const q = new URLSearchParams(params).toString();
    const res = await fetch(url + '?' + q);
    return await res.json();
  }
  
  // draw either modulation or demodulation
  async function plotDigital(demod=false) {
    const type = demod
      ? document.getElementById('dig_demod').value
      : document.getElementById('dig_type').value;
  
    const params = {
      type,
      fc:  +document.getElementById('dig_fc').value,
      br:  +document.getElementById('dig_br').value,
      dev: +document.getElementById('dig_dev').value
    };
  
    const url = demod
      ? '/digital_modulation/api/digital_demodulate'
      : '/digital_modulation/api/digital_modulate';
  
    const data = await fetchJSON(url, params);
  
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
  