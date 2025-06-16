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
  // Complex exponential helper. The computation pipeline expects
  // separate real and imaginary parts, so we handle exp_iwt by
  // evaluating the real part (cos) and imaginary part (sin)
  // separately.  The actual combination happens in compute_fourier.
  function exp_iwt(t, omega_0=1){
    // Placeholder returning only the real part.  The evaluator will
    // replace calls to exp_iwt with either cos() or sin() to obtain
    // the respective real and imaginary components.
    return Math.cos(omega_0*t);
  }
  function inv_t(t){ return t!==0 ? 1/t : 0; }
  function si(t){ return t===0 ? 1 : Math.sin(Math.PI*t)/(Math.PI*t); }

  const np = {exp: Math.exp, sin: Math.sin, cos: Math.cos, abs: Math.abs, pi: Math.PI};

  function makeEvaluator(expr){
    try {
      return new Function('t','rect','tri','step','cos','sin','sign','delta','exp_iwt','inv_t','si','exp','np',
        'return '+expr);
    } catch(e) {
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

  function replaceExp(expr, funcName){
    return expr.replace(/exp_iwt\s*\(([^)]*)\)/g, funcName + '($1)');
  }


  function fftshift(re, im){
    const N = re.length;
    const half = N>>1;
    const sre = new Float64Array(N);
    const sim = new Float64Array(N);
    for(let i=0;i<N;i++){
      const j = (i + half) % N;
      sre[i] = re[j];
      sim[i] = im[j];
    }
    return {re:sre, im:sim};
  }

  function compute_fourier(funcStr, phaseDeg){
    phaseDeg = Number(phaseDeg)||0;
    const phaseRad = phaseDeg*Math.PI/180;
    const N_SCAN = 8192;
    const N = 4096;

    const tBroad = linspace(-100,100,N_SCAN);
    const fnReBroad = makeEvaluator(replaceExp(funcStr,'Math.cos'));
    const fnImBroad = makeEvaluator(replaceExp(funcStr,'Math.sin'));
    if(!fnReBroad || !fnImBroad) return {error:'Error evaluating function'};
    let yBroadRe, yBroadIm;
    try {
      yBroadRe = evaluateArray(fnReBroad, tBroad);
      yBroadIm = evaluateArray(fnImBroad, tBroad);
    } catch(e){
      return {error:'Error evaluating function: '+e.message};
    }

    let sumMag = 0, sumT = 0;
    for(let i=0;i<yBroadRe.length;i++){
      const mag = Math.hypot(yBroadRe[i], yBroadIm[i]);
      sumMag += mag;
      sumT += tBroad[i]*mag;
    }
    const center = sumMag===0 ? 0 : sumT/sumMag;

    const t = linspace(center-20, center+20, N);
    const fnRe = makeEvaluator(replaceExp(funcStr,'Math.cos'));
    const fnIm = makeEvaluator(replaceExp(funcStr,'Math.sin'));
    if(!fnRe || !fnIm) return {error:'Error evaluating function'};
    let yRe, yIm;
    try {
      yRe = evaluateArray(fnRe, t);
      yIm = evaluateArray(fnIm, t);
    } catch(e){
      return {error:'Error evaluating function: '+e.message};
    }
    const dt = t[1]-t[0];

    // Apply phase shift
    const y_re = new Float64Array(N);
    const y_im = new Float64Array(N);
    for(let i=0;i<N;i++){
      const re = yRe[i];
      const im = yIm[i];
      y_re[i] = re*Math.cos(phaseRad) - im*Math.sin(phaseRad);
      y_im[i] = re*Math.sin(phaseRad) + im*Math.cos(phaseRad);
    }

    const fft = new FFT(N);
    const input = fft.createComplexArray();
    const output = fft.createComplexArray();
    for(let i=0;i<N;i++){
      input[2*i] = y_re[i];
      input[2*i+1] = y_im[i];
    }
    fft.transform(output, input);
    for(let i=0;i<output.length;i++) output[i] *= dt;

    const out_re = new Float64Array(N);
    const out_im = new Float64Array(N);
    for(let i=0;i<N;i++){
      out_re[i] = output[2*i];
      out_im[i] = output[2*i+1];
    }
    const shifted = fftshift(out_re, out_im);

    const f = new Float64Array(N);
    const df = 1/(N*dt);
    for(let i=0;i<N;i++) f[i] = (i - N/2)*df;

    const magnitude = new Float64Array(N);
    const phase = new Float64Array(N);
    let maxMag=0;
    for(let i=0;i<N;i++){
      const mag = Math.hypot(shifted.re[i], shifted.im[i]);
      magnitude[i]=mag; if(mag>maxMag) maxMag=mag;
    }
    for(let i=0;i<N;i++){
      let ang = Math.atan2(shifted.im[i], shifted.re[i]);
      if(maxMag>0 && magnitude[i] < 0.01*maxMag) ang = 0;
      phase[i] = ang/Math.PI;
    }
    if(maxMag>0){
      for(let i=0;i<N;i++) magnitude[i]/=maxMag;
    }

    return {
      t: Array.from(t),
      y_real: Array.from(y_re),
      y_imag: Array.from(y_im),
      f: Array.from(f),
      magnitude: Array.from(magnitude),
      phase: Array.from(phase),
      transformation_label: `Phase Shift: ${phaseDeg.toFixed(2)}°`
    };
  }

  window.compute_fourier = compute_fourier;
})();