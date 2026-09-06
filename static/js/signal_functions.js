// Shared continuous-time signal helpers for the browser-side modules.
//
// Single source of truth: fourier_compute.js and convolution_compute.js used to
// carry their own copies, which is how si() silently drifted to sin(pi*t)/(pi*t)
// in one file while the rest of the toolkit used sin(t)/t. The definitions mirror
// utils/math_utils.py and tests/test_signal_functions.py pins them against each
// other, so keep both sides in sync when adding a function.
(function(){
  'use strict';

  function rect(t){ return Math.abs(t) < 0.5 ? 1 : 0; }
  function tri(t){ t = Math.abs(t); return t <= 1 ? 1 - t : 0; }
  function step(t){ return t >= 0 ? 1 : 0; }
  function cos(t){ return Math.cos(t); }
  function sin(t){ return Math.sin(t); }
  function sign(t){ return Math.sign(t); }
  function delta(t){ const eps = 1e-3; return Math.exp(-t*t/eps)/Math.sqrt(Math.PI*eps); }
  function inv_t(t){ return t !== 0 ? 1/t : 0; }
  function si(t){ return t === 0 ? 1 : Math.sin(t)/t; }

  // Real part only. Modules that need Im{e^(j*w0*t)} substitute sin() for cos()
  // in the expression string instead of calling this.
  function exp_iwt(t, omega_0 = 1){ return Math.cos(omega_0*t); }

  const np = {exp: Math.exp, sin: Math.sin, cos: Math.cos, abs: Math.abs, pi: Math.PI};

  window.SPTSignals = {rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si, np};
})();
