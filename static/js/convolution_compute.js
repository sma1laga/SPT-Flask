(function(){
  'use strict';

  function linspace(start, end, n){
    const arr = new Float64Array(n);
    const step = (end - start)/(n-1);
    for(let i=0;i<n;i++) arr[i] = start + i*step;
    return arr;
  }

  function rect(t){ return Math.abs(t) < 0.5 ? 1 : 0; }
  function tri(t){ t = Math.abs(t); return t <= 1 ? 1 - t : 0; }
  function step(t){ return t >= 0 ? 1 : 0; }
  const cos = Math.cos;
  const sin = Math.sin;
  const sign = Math.sign;
  function delta(t){ const eps=1e-3; return Math.exp(-t*t/eps)/Math.sqrt(Math.PI*eps); }
  function exp_iwt(t, omega_0=1){ return Math.cos(omega_0*t); }
  function inv_t(t){ return t!==0 ? 1/t : 0; }
  function si(t){ return t===0 ? 1 : Math.sin(Math.PI*t)/(Math.PI*t); }

  const np = {exp: Math.exp, sin: Math.sin, cos: Math.cos, abs: Math.abs, pi: Math.PI};

  function makeEvaluator(expr){
    try {
      return new Function('t','rect','tri','step','cos','sin','sign','delta','exp_iwt','inv_t','si','exp','np',
        'return '+expr);
    } catch(e){
      return null;
    }
  }

  function evaluateArray(fn, tArr){
    const out = new Float64Array(tArr.length);
    for(let i=0;i<tArr.length;i++){
      out[i] = fn(tArr[i],rect,tri,step,cos,sin,sign,delta,exp_iwt,inv_t,si,Math.exp,np);
    }
    return out;
  }

  function convolve(x, h, dt){
    const n = x.length, m = h.length;
    const out = new Float64Array(n + m - 1);
    for(let i=0;i<n;i++){
      const xi = x[i];
      for(let j=0;j<m;j++){
        out[i+j] += xi * h[j] * dt;
      }
    }
    return out;
  }

  function compute_convolution(func1, func2){
    const N = 1024;
    const t = linspace(-10,10,N);
    const dt = t[1]-t[0];
    const f1 = makeEvaluator(func1||'0');
    const f2 = makeEvaluator(func2||'0');
    if(!f1 || !f2) return {error:'Error evaluating function'};
    let y1, y2;
    try { y1 = evaluateArray(f1, t); } catch(e){ return {error:'Error evaluating Function 1: '+e.message}; }
    try { y2 = evaluateArray(f2, t); } catch(e){ return {error:'Error evaluating Function 2: '+e.message}; }

    const y_conv_full = convolve(y1, y2, dt);
    const start = (y_conv_full.length - N) >> 1;
    const y_conv = y_conv_full.subarray(start, start + N);

    return {
      t: Array.from(t),
      y1: Array.from(y1),
      y2: Array.from(y2),
      y_conv: Array.from(y_conv)
    };
  }

  window.compute_convolution = compute_convolution;
})();