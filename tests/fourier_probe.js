// Loads the browser-side Fourier module in a Node sandbox and prints a JSON
// report of a handful of known correspondences. Driven by tests/test_fourier_js.py;
// not a test on its own.
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const JS_DIR = path.join(__dirname, '..', 'static', 'js');

const sandbox = {window: {}, console};
vm.createContext(sandbox);
for (const file of ['fft.js', 'signal_functions.js', 'fourier_compute.js']) {
  vm.runInContext(fs.readFileSync(path.join(JS_DIR, file), 'utf8'), sandbox, {filename: file});
}
const compute = sandbox.window.compute_fourier;
const NEUTRAL = {shift: 0, scale: 1, modulate: 0};

function indexNearest(arr, value) {
  let best = Infinity;
  let idx = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - value);
    if (d < best) { best = d; idx = i; }
  }
  return idx;
}

const magAt = (r, w) => r.magnitude[indexNearest(r.omega, w)];
const phaseAt = (r, w) => r.phase[indexNearest(r.omega, w)];

function maxAbs(arr, limitOmega, omega) {
  let m = 0;
  for (let i = 0; i < arr.length; i++) {
    if (limitOmega !== undefined && Math.abs(omega[i]) > limitOmega) continue;
    const v = Math.abs(arr[i]);
    if (v > m) m = v;
  }
  return m;
}

function maxMagnitudeDifference(a, b, limit) {
  let d = 0;
  for (let i = 0; i < a.omega.length; i++) {
    if (Math.abs(a.omega[i]) > limit) continue;
    d = Math.max(d, Math.abs(a.magnitude[i] - b.magnitude[i]));
  }
  return d;
}

function sampleAt(r, tv) {
  const i = indexNearest(r.t, tv);
  return {t: r.t[i], re: r.y_real[i], im: r.y_imag[i]};
}

const report = {};

// --- the transform itself ---------------------------------------------------
const rect = compute('rect(t)', 0, NEUTRAL);
report.rect = {
  areaAtDc: magAt(rect, 0),                 // X(0) equals the area under x(t)
  atFirstZero: magAt(rect, 2 * Math.PI),    // si(w/2) vanishes at w = 2*pi
  windowStart: rect.t[0],
  windowEnd: rect.t[rect.t.length - 1],
  truncated: rect.truncated,
};
report.tri = {areaAtDc: magAt(compute('tri(t)', 0, NEUTRAL), 0)};
report.scaledTri = {areaAtDc: magAt(compute('2*tri(t/2)', 0, NEUTRAL), 0)};
report.delta = (() => {
  const r = compute('delta(t)', 0, NEUTRAL);
  return {atDc: magAt(r, 0), atW3: magAt(r, 3), atW10: magAt(r, 10)};
})();

// si() must be sin(t)/t: si(pi*t) is an ideal lowpass cutting off at |w| = pi
const si = compute('si(pi*t)', 0, NEUTRAL);
report.si = {inBand: magAt(si, 2), atCutoff: magAt(si, 3), aboveCutoff: magAt(si, 4), farAbove: magAt(si, 8)};

// shift theorem, straight from the expression
const shiftedByHand = compute('rect(t-2)', 0, NEUTRAL);
report.shiftByExpression = {omega: shiftedByHand.omega[indexNearest(shiftedByHand.omega, 1)],
                            phase: phaseAt(shiftedByHand, 1)};

// --- properties must equal the hand-written equivalent ----------------------
report.properties = {
  shift: maxMagnitudeDifference(compute('rect(t)', 0, {...NEUTRAL, shift: 2}), shiftedByHand, 5 * Math.PI),
  shiftPhase: phaseAt(compute('rect(t)', 0, {...NEUTRAL, shift: 2}), 1),
  scale: maxMagnitudeDifference(compute('rect(t)', 0, {...NEUTRAL, scale: 2}),
                                compute('rect(2*t)', 0, NEUTRAL), 5 * Math.PI),
  scaleDcValue: magAt(compute('rect(t)', 0, {...NEUTRAL, scale: 2}), 0),
  modulate: maxMagnitudeDifference(compute('rect(t)', 0, {...NEUTRAL, modulate: Math.PI}),
                                   compute('rect(t)*exp_iwt(t,pi)', 0, NEUTRAL), 5 * Math.PI),
  combined: maxMagnitudeDifference(compute('rect(t)', 0, {shift: 1.5, scale: 2, modulate: 0}),
                                   compute('rect(2*(t-1.5))', 0, NEUTRAL), 5 * Math.PI),
};

// hostile property values fall back to the neutral element
report.propertyFallbacks = [
  {scale: 0}, {scale: -2}, {scale: NaN}, {shift: NaN}, {modulate: NaN},
].map(p => magAt(compute('rect(t)', 0, {...NEUTRAL, ...p}), 0));
report.propertyFallbacks.push(magAt(compute('rect(t)', 0), 0));

// --- exp_iwt: real summands must not leak into the imaginary part -----------
const mixed = compute('rect(t)+exp_iwt(t)', 0, NEUTRAL);
const outside = sampleAt(mixed, 5);
report.expIwt = {t: outside.t, im: outside.im, expectedIm: Math.sin(outside.t)};

// forms the cos/sin split cannot represent must be refused, not mis-evaluated
report.expIwtGuards = {};
for (const expr of ['exp_iwt(t)', 'exp_iwt(t,2)', 'rect(t)*exp_iwt(t)', 'exp_iwt(t)+exp_iwt(t,-1)',
                    '2*exp_iwt(t)', 'exp_iwt(t)/2', 'rect(t)+exp_iwt(t)',
                    'exp_iwt(t)**2', 'exp_iwt(t)*exp_iwt(t)', '1/exp_iwt(t)', 'exp_iwt(2*(t+1))']) {
  report.expIwtGuards[expr] = compute(expr, 0, NEUTRAL).error ? 'rejected' : 'accepted';
}

// --- symmetry: real and even is real, real and odd is imaginary -------------
const even = compute('rect(t)', 0, NEUTRAL);
const odd = compute('sign(t)*tri(t)', 0, NEUTRAL);
report.symmetry = {
  evenMaxImag: maxAbs(even.spec_imag, 5 * Math.PI, even.omega),
  oddMaxReal: maxAbs(odd.spec_real, 5 * Math.PI, odd.omega),
  oddMaxImag: maxAbs(odd.spec_imag, 5 * Math.PI, odd.omega),
};

// --- windowing, errors and the phase gaps -----------------------------------
const stepWindow = compute('step(t)', 0, NEUTRAL);
report.windows = {
  step: [stepWindow.t[0], stepWindow.t[stepWindow.t.length - 1]],
  shiftedFar: (() => { const r = compute('rect(t-50)', 0, NEUTRAL); return [r.t[0], r.t[r.t.length - 1]]; })(),
};
report.truncation = {
  sine: compute('sin(pi*t)', 0, NEUTRAL).truncated,
  step: stepWindow.truncated,
  decaying: compute('exp(-t)*step(t)', 0, NEUTRAL).truncated,
  rect: rect.truncated,
};
report.errors = {
  nonFinite: compute('tri(t)/rect(t)', 0, NEUTRAL).error || null,
  syntax: compute('rect(t)+', 0, NEUTRAL).error || null,
  unknownName: compute('foo(t)', 0, NEUTRAL).error || null,
};
report.phaseGaps = {
  isPlainArray: Array.isArray(rect.phase),
  nullCount: rect.phase.filter(v => v === null).length,
  numericCount: rect.phase.filter(v => typeof v === 'number').length,
};

// every quick-button expression has to produce finite output
report.quickButtons = {};
for (const expr of ['rect(t)', 'tri(t)', 'sin(pi*t)', 'cos(pi*t)', 'step(t)', 'delta(t)', 'sign(t)',
                    'inv_t(t)', 'si(pi*t)', 'si(pi*t)**2', 'exp_iwt(t)', 'exp(t)']) {
  const r = compute(expr, 0, NEUTRAL);
  report.quickButtons[expr] = r.error
    ? 'error'
    : (r.magnitude.every(Number.isFinite) && r.y_real.every(Number.isFinite) ? 'finite' : 'non-finite');
}

process.stdout.write(JSON.stringify(report));
