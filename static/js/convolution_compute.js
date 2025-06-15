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

    function maxAbs(arr){
    let m = 0;
    for(let i=0;i<arr.length;i++){ const v = Math.abs(arr[i]); if(v>m) m = v; }
    return m;
  }

  function activeLimits(arr, amp, axis){
    if(amp<=0) return null;
    const thr = 0.01*amp;
    let i0=-1, i1=-1;
    for(let i=0;i<arr.length;i++){ if(Math.abs(arr[i])>thr){ i0=i; break; } }
    for(let i=arr.length-1;i>=0;i--){ if(Math.abs(arr[i])>thr){ i1=i; break; } }
    if(i0>=0 && i1>=0) return [axis[i0], axis[i1]];
    return null;
  }

  function interpUniform(x0, step, yArr, xq, out){
    const n = yArr.length;
    for(let i=0;i<xq.length;i++){
      const idx = (xq[i]-x0)/step;
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      if(lo<0 || hi>=n){ out[i]=0; continue; }
      const w = idx-lo;
      out[i] = yArr[lo]*(1-w)+yArr[hi]*w;
    }
  }

  function compute_convolution(func1, func2){
    const N_SCAN = 8192;
    const tScan = linspace(-100,100,N_SCAN);
    const dtScan = tScan[1]-tScan[0];
    const f1 = makeEvaluator(func1||'0');
    const f2 = makeEvaluator(func2||'0');
    if(!f1 || !f2) return {error:'Error evaluating function'};
    let y1Scan, y2Scan;
    try { y1Scan = evaluateArray(f1, tScan); } catch(e){ return {error:'Error evaluating Function 1: '+e.message}; }
    try { y2Scan = evaluateArray(f2, tScan); } catch(e){ return {error:'Error evaluating Function 2: '+e.message}; }

    const amp1 = maxAbs(y1Scan);
    const amp2 = maxAbs(y2Scan);
    const r1 = activeLimits(y1Scan, amp1, tScan);
    const r2 = activeLimits(y2Scan, amp2, tScan);

    let tMin, tMax;
    if(r1 || r2){
      const t1Min = r1 ? r1[0] : 0;
      const t1Max = r1 ? r1[1] : 0;
      const t2Min = r2 ? r2[0] : 0;
      const t2Max = r2 ? r2[1] : 0;
      const convMin = t1Min + t2Min;
      const convMax = t1Max + t2Max;
      tMin = Math.min(t1Min, t2Min, convMin);
      tMax = Math.max(t1Max, t2Max, convMax);
    } else {
      tMin = -10; tMax = 10;
    }
    const margin = 2;
    tMin -= margin; tMax += margin;

    const N = 4096;
    const t = linspace(tMin, tMax, N);
    const dt = t[1]-t[0];
    let y1, y2;
    try { y1 = evaluateArray(f1, t); } catch(e){ return {error:'Error evaluating Function 1: '+e.message}; }
    try { y2 = evaluateArray(f2, t); } catch(e){ return {error:'Error evaluating Function 2: '+e.message}; }

    const yConvFull = convolve(y1Scan, y2Scan, dtScan);
    const convStart = 2*tScan[0];
    const y_conv = new Float64Array(N);
    interpUniform(convStart, dtScan, yConvFull, t, y_conv);

    return {
      t: Array.from(t),
      y1: Array.from(y1),
      y2: Array.from(y2),
      y_conv: Array.from(y_conv)
    };
  }

  window.compute_convolution = compute_convolution;
})();