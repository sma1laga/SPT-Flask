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
  // Complex exponential helper
  function exp_iwt(t, omega_0=1){
    // Placeholder returning only Re part
    return Math.cos(omega_0*t);
  }
  function inv_t(t){ return t!==0 ? 1/t : 0; }
  function si(t){ return t===0 ? 1 : Math.sin(Math.PI*t)/(Math.PI*t); }

  const np = {exp: Math.exp, sin: Math.sin, cos: Math.cos, abs: Math.abs, pi: Math.PI};

  const evaluatorCache = new Map();
  let cachedFunc = null;
  let cachedBase = null;
  let cachedFFT = null;
  function makeEvaluator(expr){
    if (evaluatorCache.has(expr)) {
      return evaluatorCache.get(expr);
    }
    let fn = null;
    try {
      const factory = new Function(
        'rect','tri','step','cos','sin','sign','delta','exp_iwt','inv_t','si','exp','np','pi','Math',
        'return function(t){ return ' + expr + '; };'
      );
      fn = factory(rect,tri,step,cos,sin,sign,delta,exp_iwt,inv_t,si,Math.exp,np,Math.PI,Math);
    } catch(e) {
      fn = null;
    }
    evaluatorCache.set(expr, fn);
    return fn;
  }

  function evaluateArray(fn, tArr, out){
    const y = out || new Float64Array(tArr.length);
    for(let i=0;i<tArr.length;i++){
      y[i] = fn(tArr[i]);
    }
    return y;
  }

  function replaceExp(expr, funcName){
    return expr.replace(/exp_iwt\s*\(\s*([^,()]+?)\s*(?:,\s*([^)]+?)\s*)?\)/g,
      (m, tExpr, wExpr) => wExpr ? `${funcName}((${wExpr})*(${tExpr}))` : `${funcName}(${tExpr})`
    );
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

    if (!cachedFFT || cachedFFT.size !== N) {
      cachedFFT = {size: N, fft: new FFT(N)};
    }

    if (funcStr !== cachedFunc) {
      const hasExp = /exp_iwt\s*\(/.test(funcStr);
      const exprRe = hasExp ? replaceExp(funcStr,'Math.cos') : funcStr;
      const exprIm = hasExp ? replaceExp(funcStr,'Math.sin') : null;

      const tBroad = linspace(-100,100,N_SCAN);
      const fnReBroad = makeEvaluator(exprRe);
      if(!fnReBroad) return {error:'Error evaluating function'};

      let yBroadRe, yBroadIm;
      try {
        yBroadRe = evaluateArray(fnReBroad, tBroad);
        if (hasExp) {
          const fnImBroad = makeEvaluator(exprIm);
          if(!fnImBroad) return {error:'Error evaluating function'};
          yBroadIm = evaluateArray(fnImBroad, tBroad);
        }
      } catch(e){
        return {error:'Error evaluating function: '+e.message};
      }

      let sumMag = 0, sumT = 0;
      if (hasExp) {
        for(let i=0;i<yBroadRe.length;i++){
          const mag = Math.hypot(yBroadRe[i], yBroadIm[i]);
          sumMag += mag;
          sumT += tBroad[i]*mag;
        }
      } else {
        for(let i=0;i<yBroadRe.length;i++){
          const mag = Math.abs(yBroadRe[i]);
          sumMag += mag;
          sumT += tBroad[i]*mag;
        }
      }
      const center = sumMag===0 ? 0 : sumT/sumMag;

      const t = linspace(center-20, center+20, N);
      const fnRe = makeEvaluator(exprRe);
      if(!fnRe) return {error:'Error evaluating function'};

      let yRe, yIm;
      try {
        yRe = evaluateArray(fnRe, t);
        if (hasExp) {
          const fnIm = makeEvaluator(exprIm);
          if(!fnIm) return {error:'Error evaluating function'};
          yIm = evaluateArray(fnIm, t);
        } else {
          yIm = new Float64Array(N);
        }
      } catch(e){
        return {error:'Error evaluating function: '+e.message};
      }

      const dt = t[1]-t[0];

      const fft = cachedFFT.fft;
      const input = fft.createComplexArray();
      const output = fft.createComplexArray();
      for(let i=0;i<N;i++){
        input[2*i] = yRe[i];
        input[2*i+1] = yIm[i];
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
      const omega = new Float64Array(N);
      for(let i=0;i<N;i++) omega[i] = 2 * Math.PI * f[i];

      const magnitude = new Float64Array(N);
      const basePhase = new Float64Array(N);
      const lowMagnitude = new Uint8Array(N);
      let maxMag=0;
      for(let i=0;i<N;i++){
        const mag = Math.hypot(shifted.re[i], shifted.im[i]);
        magnitude[i]=mag; if(mag>maxMag) maxMag=mag;
      }
      const noiseThreshold = 0.02;
      const thresholdValue = maxMag * noiseThreshold;
      for(let i=0;i<N;i++){
        let ang = Math.atan2(shifted.im[i], shifted.re[i]);
        if(maxMag>0 && magnitude[i] < thresholdValue){
          ang = 0;
          lowMagnitude[i] = 1;
        }
        if(ang > Math.PI) ang -= 2 * Math.PI;
        if(ang < -Math.PI) ang += 2 * Math.PI;
        basePhase[i] = ang;
      }
      if(maxMag>0){
        for(let i=0;i<N;i++) magnitude[i]/=maxMag;
      }

      cachedFunc = funcStr;
      cachedBase = {
        t,
        yRe,
        yIm,
        omega,
        magnitude,
        basePhase,
        lowMagnitude
      };
    }

    if (!cachedBase) {
      return {error:'Error evaluating function'};
    }
    const {t, yRe, yIm, omega, magnitude, basePhase, lowMagnitude} = cachedBase;

    // Apply phase shift using cached base arrays
    // Apply phase shift
    const y_re = new Float64Array(N);
    const y_im = new Float64Array(N);
    const cosPhase = Math.cos(phaseRad);
    const sinPhase = Math.sin(phaseRad);
    for(let i=0;i<N;i++){
      const re = yRe[i];
      const im = yIm[i];
      y_re[i] = re*cosPhase - im*sinPhase;
      y_im[i] = re*sinPhase + im*cosPhase;
    }


    const phase = new Float64Array(N);
    for(let i=0;i<N;i++){
      if (lowMagnitude[i]) {
        phase[i] = 0;
        continue;
      }
      let ang = basePhase[i] + phaseRad;
      if(ang > Math.PI) ang -= 2 * Math.PI;
      if(ang < -Math.PI) ang += 2 * Math.PI;
      phase[i] = ang;
    }

    return {
      t,
      y_real: y_re,
      y_imag: y_im,
      omega,
      magnitude,
      phase,
      transformation_label: `Phase Shift = ${phaseDeg.toFixed(2)}°`
    };
  }

  window.compute_fourier = compute_fourier;
})();