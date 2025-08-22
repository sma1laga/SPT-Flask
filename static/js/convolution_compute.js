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
  function si(t){ return t===0 ? 1 : Math.sin(t)/t; }

  const np = {exp: Math.exp, sin: Math.sin, cos: Math.cos, abs: Math.abs, pi: Math.PI};

  function makeEvaluator(expr){
    // Allow direct usage of common math helpers like pi, e, abs, exp without np.
    // This avoids the previous requirement of prefixing with "np.".
    try {
      return new Function(
        't','rect','tri','step','cos','sin','sign','delta','exp_iwt','inv_t','si','exp','abs','pi','e','np',        'return '+expr);
    } catch(e){
      return null;
    }
  }

  function evaluateArray(fn, tArr, reverse=false){
    const out = new Float64Array(tArr.length);
    for(let i=0;i<tArr.length;i++){
      const tt = reverse ? -tArr[i] : tArr[i];
      out[i] = fn(tt,rect,tri,step,cos,sin,sign,delta,exp_iwt,inv_t,si,Math.exp,Math.abs,Math.PI,Math.E,np);
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

  function compute_convolution(func1, func2, reverseSecond=false){
    // reverseSecond enables time reversal of the second function without string manipulation.
    const N_SCAN = 8192;
    const tScan = linspace(-100,100,N_SCAN);
    const dtScan = tScan[1]-tScan[0];
    const f1 = makeEvaluator(func1||'0');
    const f2 = makeEvaluator(func2||'0');
    if(!f1 || !f2) return {error:'Error evaluating function'};
    let y1Scan, y2Scan;
    try { y1Scan = evaluateArray(f1, tScan); } catch(e){ return {error:'Error evaluating Function 1: '+e.message}; }
    try { y2Scan = evaluateArray(f2, tScan, reverseSecond); } catch(e){ return {error:'Error evaluating Function 2: '+e.message}; }

    const amp1 = maxAbs(y1Scan);
    const amp2 = maxAbs(y2Scan);
    let r1 = activeLimits(y1Scan, amp1, tScan);
    let r2 = activeLimits(y2Scan, amp2, tScan);

    const DISP_WIDTH = 20;
    const edgeMargin = 0.05*(tScan[tScan.length-1]-tScan[0]);

    function adjustRegion(r){
      if(!r) return [-10,10];
      const leftTouch = r[0] <= tScan[0] + edgeMargin;
      const rightTouch = r[1] >= tScan[tScan.length-1] - edgeMargin;
      if(leftTouch && rightTouch) return [-DISP_WIDTH/2, DISP_WIDTH/2];
      if(leftTouch) return [r[1]-DISP_WIDTH, r[1]];
      if(rightTouch) return [r[0], r[0]+DISP_WIDTH];
      return r;
    }

    r1 = adjustRegion(r1);
    r2 = adjustRegion(r2);

    const margin = 2;

    // Axis for x(t)
    let t1Min = r1[0];
    let t1Max = r1[1];
    t1Min -= margin; t1Max += margin;

    // Axis for h(t)
    let t2Min = r2[0];
    let t2Max = r2[1];
    t2Min -= margin; t2Max += margin;

    // Axis for convolution result
    let convMin = r1[0] + r2[0];
    let convMax = r1[1] + r2[1];
    convMin -= margin; convMax += margin;


    const N = 4096;
    const t1 = linspace(t1Min, t1Max, N);
    const t2 = linspace(t2Min, t2Max, N);
    const tConv = linspace(convMin, convMax, N);
    const dt1 = t1[1]-t1[0];
    const dt2 = t2[1]-t2[0];

    let y1, y2;
    try { y1 = evaluateArray(f1, t1); } catch(e){ return {error:'Error evaluating Function 1: '+e.message}; }
    try { y2 = evaluateArray(f2, t2, reverseSecond); } catch(e){ return {error:'Error evaluating Function 2: '+e.message}; }

    const yConvFull = convolve(y1Scan, y2Scan, dtScan);
    const convStart = 2*tScan[0];
    const y_conv = new Float64Array(tConv.length);
    interpUniform(convStart, dtScan, yConvFull, tConv, y_conv);

    return {
      t1: Array.from(t1),
      t2: Array.from(t2),
      t_conv: Array.from(tConv),
      y1: Array.from(y1),
      y2: Array.from(y2),
      y_conv: Array.from(y_conv)
    };
  }

  window.compute_convolution = compute_convolution;
})();