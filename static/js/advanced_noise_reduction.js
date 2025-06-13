// Client-side advanced noise reduction
const ctx = new (window.AudioContext || window.webkitAudioContext)();
let originalSignal = null;
let sampleRate = 44100;

const fft = (re, im) => {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const step = -2 * Math.PI / len;
    for (let i = 0; i < n; i += len) {
      for (let j = 0; j < len / 2; j++) {
        const k = i + j;
        const k2 = k + len / 2;
        const cos = Math.cos(step * j);
        const sin = Math.sin(step * j);
        const tre = re[k2] * cos - im[k2] * sin;
        const tim = re[k2] * sin + im[k2] * cos;
        re[k2] = re[k] - tre;
        im[k2] = im[k] - tim;
        re[k] += tre;
        im[k] += tim;
      }
    }
  }
};

const ifft = (re, im) => {
  for (let i = 0; i < re.length; i++) im[i] = -im[i];
  fft(re, im);
  for (let i = 0; i < re.length; i++) { re[i] /= re.length; im[i] = -im[i] / re.length; }
};

const nextPow2 = n => 1 << Math.ceil(Math.log2(n));

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

function wienerFilter(sig) {
  const N = nextPow2(sig.length);
  const re = new Float32Array(N); re.set(sig);
  const im = new Float32Array(N);
  fft(re, im);
  let maxMag = 0;
  for (let i = 0; i < N; i++) maxMag = Math.max(maxMag, Math.hypot(re[i], im[i]));
  const noise = 0.05 * maxMag;
  for (let i = 0; i < N; i++) {
    const mag2 = re[i] * re[i] + im[i] * im[i];
    const gain = mag2 / (mag2 + noise * noise);
    re[i] *= gain; im[i] *= gain;
  }
  ifft(re, im);
  return re.slice(0, sig.length);
}

function spectralSubtract(sig) {
  const N = nextPow2(sig.length);
  const re = new Float32Array(N); re.set(sig);
  const im = new Float32Array(N);
  fft(re, im);
  let maxMag = 0;
  for (let i = 0; i < N; i++) maxMag = Math.max(maxMag, Math.hypot(re[i], im[i]));
  const noise = 0.05 * maxMag;
  for (let i = 0; i < N; i++) {
    const mag = Math.hypot(re[i], im[i]);
    const ph = Math.atan2(im[i], re[i]);
    const newMag = Math.max(mag - noise, 0.1 * noise);
    re[i] = newMag * Math.cos(ph);
    im[i] = newMag * Math.sin(ph);
  }
  ifft(re, im);
  return re.slice(0, sig.length);
}

function exportWav(samples, fs) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
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

function plotTimeFreq(orig, den) {
  const t = Array.from({ length: orig.length }, (_, i) => i / sampleRate);
  Plotly.newPlot('timePlot', [
    { x: t, y: orig, name: 'Original', mode: 'lines' },
    { x: t, y: den, name: 'Denoised', mode: 'lines', line: { color: 'orange' } }
  ], { height: 300, margin: { t: 30 }, legend: { orientation: 'h' } });

  const N = nextPow2(orig.length);
  const r1 = new Float32Array(N); r1.set(orig);
  const i1 = new Float32Array(N);
  const r2 = new Float32Array(N); r2.set(den);
  const i2 = new Float32Array(N);
  fft(r1, i1); fft(r2, i2);
  const freqs = Array.from({ length: N / 2 }, (_, k) => k * sampleRate / N);
  const mag1 = freqs.map((_, k) => 20 * Math.log10(Math.hypot(r1[k], i1[k]) + 1e-6));
  const mag2 = freqs.map((_, k) => 20 * Math.log10(Math.hypot(r2[k], i2[k]) + 1e-6));
  Plotly.newPlot('freqPlot', [
    { x: freqs, y: mag1, mode: 'lines', name: 'Original' },
    { x: freqs, y: mag2, mode: 'lines', name: 'Denoised', line: { color: 'orange' } }
  ], { height: 300, margin: { t: 30 }, xaxis: { title: 'Hz' } });

  const gain = freqs.map((_, k) => mag2[k] - mag1[k]);
  Plotly.newPlot('gainPlot', [
    { x: freqs, y: gain, mode: 'lines' }
  ], { height: 250, margin: { t: 30 }, xaxis: { title: 'Hz' }, yaxis: { title: 'Gain (dB)' } });
}

function setOutput(data) {
  const blob = exportWav(data, sampleRate);
  const url = URL.createObjectURL(blob);
  document.getElementById('outputAudio').src = url;
  document.getElementById('downloadLink').href = url;
  document.getElementById('vizSection').style.display = 'block';
  document.getElementById('audioSection').style.display = 'block';
}

document.addEventListener('DOMContentLoaded', () => {
  const radios = document.getElementsByName('audio_choice');
  const uploadDiv = document.getElementById('uploadDiv');
  const fileIn = document.getElementById('audio_file');
  radios.forEach(r => r.addEventListener('change', () => {
    if (r.value === 'upload' && r.checked) { uploadDiv.style.display = 'block'; fileIn.required = true; }
    else if (r.value === 'default' && r.checked) { uploadDiv.style.display = 'none'; fileIn.required = false; }
  }));

  document.getElementById('nrForm').addEventListener('submit', async ev => {
    ev.preventDefault();
    const err = document.getElementById('errorMsg');
    try {
      const signal = await loadAudio();
      const method = document.getElementById('method').value;
      let den;
      if (method === 'wiener') den = wienerFilter(signal);
      else if (method === 'spectral') den = spectralSubtract(signal);
      else { err.textContent = 'Wavelet unavailable client-side'; return; }
      plotTimeFreq(signal, den);
      setOutput(den);
      err.textContent = '';
    } catch (e) {
      err.textContent = e.message;
    }
  });
});