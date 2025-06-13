// Client-side speech filter processing
const ctx = new (window.AudioContext || window.webkitAudioContext)();
let originalSignal = null;
let sampleRate = 44100;

const fft = (re, im) => {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const step = -2 * Math.PI / len;
    for (let i = 0; i < n; i += len) {
      for (let j = 0; j < len / 2; j++) {
        const idx = i + j;
        const idx2 = idx + len / 2;
        const cos = Math.cos(step * j);
        const sin = Math.sin(step * j);
        const tre = re[idx2] * cos - im[idx2] * sin;
        const tim = re[idx2] * sin + im[idx2] * cos;
        re[idx2] = re[idx] - tre;
        im[idx2] = im[idx] - tim;
        re[idx] += tre;
        im[idx] += tim;
      }
    }
  }
};

const ifft = (re, im) => {
  for (let i = 0; i < re.length; i++) im[i] = -im[i];
  fft(re, im);
  for (let i = 0; i < re.length; i++) {
    re[i] /= re.length;
    im[i] = -im[i] / re.length;
  }
};

function quadraticRoots(a, b, c) {
  if (Math.abs(a) < 1e-12) return [{ re: -c / b, im: 0 }];
  const disc = b * b - 4 * a * c;
  if (disc >= 0) {
    const r = Math.sqrt(disc);
    return [
      { re: (-b + r) / (2 * a), im: 0 },
      { re: (-b - r) / (2 * a), im: 0 },
    ];
  }
  const r = Math.sqrt(-disc);
  return [
    { re: -b / (2 * a), im: r / (2 * a) },
    { re: -b / (2 * a), im: -r / (2 * a) },
  ];
}

function biquadCoeffs(type, f0, Q, fs) {
  const w0 = 2 * Math.PI * f0 / fs;
  const cos = Math.cos(w0);
  const sin = Math.sin(w0);
  const alpha = sin / (2 * Q);
  let b0, b1, b2, a0, a1, a2;
  if (type === 'lowpass') {
    b0 = (1 - cos) / 2; b1 = 1 - cos; b2 = (1 - cos) / 2;
  } else if (type === 'highpass') {
    b0 = (1 + cos) / 2; b1 = -(1 + cos); b2 = (1 + cos) / 2;
  } else if (type === 'bandpass') {
    b0 = sin / 2; b1 = 0; b2 = -sin / 2;
  } else { // notch
    b0 = 1; b1 = -2 * cos; b2 = 1;
  }
  a0 = 1 + alpha;
  a1 = -2 * cos;
  a2 = 1 - alpha;
  return {
    b: [b0 / a0, b1 / a0, b2 / a0],
    a: [1, a1 / a0, a2 / a0]
  };
}

async function loadAudio() {
  const choice = document.querySelector('input[name="audio_choice"]:checked').value;
  let arrayBuffer;
  if (choice === 'default') {
    const resp = await fetch('/static/audio/example.wav');
    arrayBuffer = await resp.arrayBuffer();
  } else {
    const file = document.getElementById('audio_file').files[0];
    if (!file) throw new Error('Please select an audio file.');
    arrayBuffer = await file.arrayBuffer();
  }
  const decoded = await ctx.decodeAudioData(arrayBuffer);
  sampleRate = decoded.sampleRate;
  const ch0 = decoded.numberOfChannels > 1
    ? decoded.getChannelData(0).map((v, i) => (v + decoded.getChannelData(1)[i]) / 2)
    : decoded.getChannelData(0);
  originalSignal = new Float32Array(ch0);
  if (originalSignal.length / sampleRate > 10) throw new Error('Audio longer than 10 seconds.');
  return originalSignal;
}

function getParams() {
  if (document.getElementById('use_standard_filter').checked) {
    const preset = document.getElementById('standard_filter').value;
    if (preset === 'lowpass_std') return { filter: 'lowpass', order: 3, cutoff: '1000' };
    if (preset === 'bandpass_std') return { filter: 'bandpass', order: 4, cutoff: '300,3400' };
    if (preset === 'highpass_std') return { filter: 'highpass', order: 2, cutoff: '80' };
    if (preset === 'telephone_filter') return { filter: 'bandpass', order: 4, cutoff: '300,3400' };
    if (preset === 'podcast_filter') return { filter: 'lowpass', order: 3, cutoff: '3000' };
    return { filter: 'bandstop', order: 2, cutoff: '50,150' };
  }
  return {
    filter: document.getElementById('filter_type').value,
    order: parseInt(document.getElementById('order').value, 10),
    cutoff: document.getElementById('cutoff').value
  };
}

async function applyFilter(signal, params) {
  const cuts = params.cutoff.split(',').map(Number);
  const stages = Math.ceil(params.order / 2);
  const offCtx = new OfflineAudioContext(1, signal.length, sampleRate);
  const src = offCtx.createBufferSource();
  const buf = offCtx.createBuffer(1, signal.length, sampleRate);
  buf.copyToChannel(signal, 0);
  src.buffer = buf;
  let node = src;
  const zeros = [], poles = [];
  for (let i = 0; i < stages; i++) {
    const bq = offCtx.createBiquadFilter();
    if (params.filter === 'bandstop') bq.type = 'notch';
    else bq.type = params.filter;
    if (params.filter === 'bandpass' || params.filter === 'bandstop') {
      const low = cuts[0], high = cuts[1];
      const fc = Math.sqrt(low * high);
      bq.frequency.value = fc;
      bq.Q.value = fc / (high - low);
    } else {
      bq.frequency.value = cuts[0];
      bq.Q.value = Math.SQRT1_2;
    }
    const coeffs = biquadCoeffs(bq.type === 'notch' ? 'notch' : bq.type, bq.frequency.value, bq.Q.value, sampleRate);
    zeros.push(...quadraticRoots(coeffs.b[0], coeffs.b[1], coeffs.b[2]));
    poles.push(...quadraticRoots(1, coeffs.a[1], coeffs.a[2]));
    node.connect(bq);
    node = bq;
  }
  node.connect(offCtx.destination);
  src.start();
  const rendered = await offCtx.startRendering();
  return { data: rendered.getChannelData(0), zeros, poles };
}

function stft(sig, nfft = 256, hop = 128) {
  const window = new Float32Array(nfft);
  for (let i = 0; i < nfft; i++) window[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / nfft);
  const frames = Math.floor((sig.length - nfft) / hop) + 1;
  const spec = Array.from({ length: nfft / 2 }, () => new Array(frames).fill(0));
  const times = [];
  const freqs = [];
  for (let k = 0; k < nfft / 2; k++) freqs.push(k * sampleRate / nfft);
  const re = new Float32Array(nfft);
  const im = new Float32Array(nfft);
  for (let i = 0; i < frames; i++) {
    times.push(i * hop / sampleRate);
    for (let j = 0; j < nfft; j++) {
      re[j] = sig[i * hop + j] * window[j];
      im[j] = 0;
    }
    fft(re, im);
    for (let k = 0; k < nfft / 2; k++) {
      spec[k][i] = 20 * Math.log10(Math.hypot(re[k], im[k]) + 1e-6);
    }
  }
  return { times, freqs, spec };
}

function plotResults(orig, filtered, fs, zeros, poles) {
  const dur = orig.length / fs;
  const t = Array.from({ length: orig.length }, (_, i) => i / fs);
  Plotly.newPlot('wavePlot', [
    { x: t, y: orig, name: 'Original', mode: 'lines' },
    { x: t, y: filtered, name: 'Filtered', mode: 'lines', line: { color: 'orange' } }
  ], { height: 300, margin: { t: 30 }, legend: { orientation: 'h' } });

  const os = stft(orig);
  const fspec = stft(filtered);
  Plotly.newPlot('specPlot', [
    { x: os.times, y: os.freqs, z: os.spec, type: 'heatmap', colorscale: 'Viridis' },
    { x: fspec.times, y: fspec.freqs, z: fspec.spec, type: 'heatmap', colorscale: 'Viridis', xaxis: 'x2', yaxis: 'y2' }
  ], {
    grid: { rows: 2, columns: 1, pattern: 'independent' },
    height: 400,
    margin: { t: 30 }
  });

  const zr = zeros.map(z => z.re); const zi = zeros.map(z => z.im);
  const pr = poles.map(p => p.re); const pi = poles.map(p => p.im);
  const theta = Array.from({ length: 200 }, (_, i) => i * 2 * Math.PI / 199);
  Plotly.newPlot('pzPlot', [
    { x: zr, y: zi, mode: 'markers', name: 'Zeros', marker: { color: 'blue' } },
    { x: pr, y: pi, mode: 'markers', name: 'Poles', marker: { symbol: 'x', color: 'red' } },
    { x: theta.map(c => Math.cos(c)), y: theta.map(s => Math.sin(s)), mode: 'lines', line: { dash: 'dot', color: 'black' }, showlegend: false }
  ], {
    xaxis: { title: 'Real', range: [-1.5, 1.5] },
    yaxis: { title: 'Imag', range: [-1.5, 1.5] },
    height: 350,
    margin: { t: 30 }
  });
}

function exportWav(samples, fs) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
  write(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, fs, true);
  view.setUint32(28, fs * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 32768 : s * 32767, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

function setOutput(buf, fs) {
  const blob = exportWav(buf, fs);
  const url = URL.createObjectURL(blob);
  document.getElementById('liveAudio').src = url;
  const dl = document.getElementById('downloadLink');
  dl.href = url;
  document.getElementById('vizSection').style.display = 'block';
  document.getElementById('audioSection').style.display = 'block';
  document.getElementById('pzSection').style.display = 'block';
}

function showError(msg) {
  document.getElementById('errorMsg').textContent = msg || '';
}

document.addEventListener('DOMContentLoaded', () => {
  const radios = document.getElementsByName('audio_choice');
  const uploadDiv = document.getElementById('uploadDiv');
  const fileIn = document.getElementById('audio_file');
  radios.forEach(r => r.addEventListener('change', () => {
    if (r.value === 'upload' && r.checked) { uploadDiv.style.display = 'block'; fileIn.required = true; }
    else if (r.value === 'default' && r.checked) { uploadDiv.style.display = 'none'; fileIn.required = false; }
  }));
  const useStd = document.getElementById('use_standard_filter');
  useStd.addEventListener('change', () => {
    document.getElementById('standard_filter_div').style.display = useStd.checked ? 'block' : 'none';
    document.getElementById('custom_filter_div').style.display = useStd.checked ? 'none' : 'block';
  });

  document.getElementById('filterForm').addEventListener('submit', async ev => {
    ev.preventDefault();
    try {
      const signal = await loadAudio();
      const params = getParams();
      const result = await applyFilter(signal, params);
      plotResults(signal, result.data, sampleRate, result.zeros, result.poles);
      setOutput(result.data, sampleRate);
      showError('');
    } catch (err) {
      showError(err.message);
    }
  });

  document.getElementById('reapplyBtn').addEventListener('click', async () => {
    if (!originalSignal) return;
    try {
      const params = getParams();
      const result = await applyFilter(originalSignal, params);
      plotResults(originalSignal, result.data, sampleRate, result.zeros, result.poles);
      setOutput(result.data, sampleRate);
    } catch (err) {
      showError(err.message);
    }
  });
});