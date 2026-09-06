(function(){
  'use strict';

  function linspace(start, end, n){
    const arr = new Float64Array(n);
    const step = (end - start)/(n-1);
    for(let i=0;i<n;i++) arr[i] = start + i*step;
    return arr;
  }

  const SPT = window.SPTSignals;
  if(!SPT) throw new Error('signal_functions.js must be loaded before this script');
  const {rect, tri, step, cos, sin, sign, delta, exp_iwt, inv_t, si, np} = SPT;

  const evaluatorCache = new Map();
  let cachedKey = null;
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

  // `map` turns the plot time into the argument x is evaluated at, which is how
  // the scaling and shifting properties are applied: u = a*(t - t0).
  function evaluateArray(fn, tArr, map){
    const y = new Float64Array(tArr.length);
    if(map){
      for(let i=0;i<tArr.length;i++) y[i] = fn(map(tArr[i]));
    } else {
      for(let i=0;i<tArr.length;i++) y[i] = fn(tArr[i]);
    }
    return y;
  }

  // exp_iwt(t) is expanded into a cos/sin pair, so an expression that is affine in
  // exp_iwt reads as A + B*cos(wt) for the real and A + B*sin(wt) for the imaginary
  // part. Replacing exp_iwt by 0 yields the exp_iwt-free part A, which has to be
  // subtracted so that purely real summands do not leak into Im{x(t)}.
  const EXP_IWT_CALL = /exp_iwt\s*\(\s*([^,()]+?)\s*(?:,\s*([^)]+?)\s*)?\)/g;

  function replaceExp(expr, funcName){
    return expr.replace(EXP_IWT_CALL,
      (m, tExpr, wExpr) => wExpr ? `${funcName}((${wExpr})*(${tExpr}))` : `${funcName}(${tExpr})`
    );
  }

  function stripExp(expr){
    return expr.replace(EXP_IWT_CALL, '(0)');
  }

  // The cos/sin split is only valid for expressions that are affine in exp_iwt.
  // Rather than trusting the source text, the expression is probed numerically:
  // replacing every exp_iwt call by a free variable c, an affine expression must
  // satisfy f(2c) - f(0) = 2*(f(c) - f(0)) at every sample point. This catches
  // powers, products and quotients of exp_iwt that would otherwise be evaluated
  // to a silently wrong imaginary part.
  function isAffineInExp(expr){
    let probe;
    try {
      probe = new Function(
        'rect','tri','step','cos','sin','sign','delta','exp_iwt','inv_t','si','exp','np','pi','Math',
        'return function(t, __c){ return ' + expr.replace(EXP_IWT_CALL, '(__c)') + '; };'
      )(rect,tri,step,cos,sin,sign,delta,exp_iwt,inv_t,si,Math.exp,np,Math.PI,Math);
    } catch(e) {
      return true; // a broken expression is reported by the normal error path
    }
    // Probed at c = 1, 2, 3 rather than at 0: a quotient such as 1/exp_iwt(t)
    // is infinite at c = 0 and would be skipped as undecidable.
    for (const tv of [-3.7, -1.2, -0.3, 0.4, 1.1, 2.6, 5.3]) {
      let f1, f2, f3;
      try { f1 = probe(tv, 1); f2 = probe(tv, 2); f3 = probe(tv, 3); } catch(e) { return true; }
      if (![f1, f2, f3].every(Number.isFinite)) continue;
      const lhs = f3 - f1;
      const rhs = 2*(f2 - f1);
      const scale = Math.max(1, Math.abs(lhs), Math.abs(rhs));
      if (Math.abs(lhs - rhs) > 1e-9*scale) return false;
    }
    return true;
  }

  function evaluateImag(fnIm, fnZero, tArr, map){
    const im = evaluateArray(fnIm, tArr, map);
    const base = evaluateArray(fnZero, tArr, map);
    for(let i=0;i<im.length;i++){
      if (Number.isFinite(base[i])) im[i] -= base[i];
    }
    return im;
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

  function normaliseProperty(value, fallback, positive){
    const v = Number(value);
    if(!Number.isFinite(v)) return fallback;
    if(positive && v <= 0) return fallback;
    return v;
  }

  // The Fourier properties are applied to the signal itself and the transform is
  // taken of the result, so every slider is exact rather than an approximation:
  //   y(t) = x(a*(t - t0)) * e^(j*w0*t)
  //   Y(jw) = (1/|a|) * X(j*(w - w0)/a) * e^(-j*(w - w0)*t0)
  function compute_fourier(funcStr, phaseDeg, properties){
    phaseDeg = Number(phaseDeg)||0;
    const phaseRad = phaseDeg*Math.PI/180;
    const N_SCAN = 8192;
    const N = 4096;
    const HALF_WINDOW = 20;

    const opts = properties || {};
    const shift = normaliseProperty(opts.shift, 0, false);
    const scale = normaliseProperty(opts.scale, 1, true);
    const modulate = normaliseProperty(opts.modulate, 0, false);

    const argMap = (scale === 1 && shift === 0) ? null : (tv => scale*(tv - shift));
    const cacheKey = `${funcStr}|${shift}|${scale}|${modulate}`;

    if (!cachedFFT || cachedFFT.size !== N) {
      cachedFFT = {size: N, fft: new FFT(N)};
    }

    if (cacheKey !== cachedKey) {
      const hasExp = /exp_iwt\s*\(/.test(funcStr);
      if (hasExp) {
        const calls = (funcStr.match(/exp_iwt\s*\(/g) || []).length;
        const parsed = (funcStr.match(EXP_IWT_CALL) || []).length;
        if (calls !== parsed) {
          return {error: 'exp_iwt(...) takes a simple argument such as exp_iwt(t-1) or exp_iwt(t, 2); brackets inside the argument are not supported.'};
        }
        if (!isAffineInExp(funcStr)) {
          return {error: 'Only expressions that are linear in exp_iwt(...) can be split into real and imaginary part, for example rect(t)*exp_iwt(t) or exp_iwt(t)+exp_iwt(t,-1). Powers, products or quotients of exp_iwt are not supported.'};
        }
      }
      const exprRe = hasExp ? replaceExp(funcStr,'Math.cos') : funcStr;
      const exprIm = hasExp ? replaceExp(funcStr,'Math.sin') : null;
      const exprZero = hasExp ? stripExp(funcStr) : null;

      const tBroad = linspace(-100,100,N_SCAN);
      const fnReBroad = makeEvaluator(exprRe);
      if(!fnReBroad) return {error:'Error evaluating function'};

      let yBroadRe, yBroadIm;
      try {
        yBroadRe = evaluateArray(fnReBroad, tBroad, argMap);
        if (hasExp) {
          const fnImBroad = makeEvaluator(exprIm);
          const fnZeroBroad = makeEvaluator(exprZero);
          if(!fnImBroad || !fnZeroBroad) return {error:'Error evaluating function'};
          yBroadIm = evaluateImag(fnImBroad, fnZeroBroad, tBroad, argMap);
        }
      } catch(e){
        return {error:'Error evaluating function: '+e.message};
      }

      let sumMag = 0, sumT = 0, sumNear = 0;
      for(let i=0;i<yBroadRe.length;i++){
        const mag = hasExp ? Math.hypot(yBroadRe[i], yBroadIm[i]) : Math.abs(yBroadRe[i]);
        if(!Number.isFinite(mag)) continue;
        sumMag += mag;
        sumT += tBroad[i]*mag;
        if(Math.abs(tBroad[i]) <= HALF_WINDOW) sumNear += mag;
      }

      // Recentring exists to catch signals that live outside the default window.
      // For unbounded signals such as step(t) the centroid is an artefact of the
      // scan range (step would move the view to t = 50 and hide its own edge), so
      // t = 0 is kept as long as the default window shows anything at all.
      let center = 0;
      if (sumMag > 0 && sumNear < 0.01*sumMag) {
        const centroid = sumT/sumMag;
        if (Number.isFinite(centroid)) center = centroid;
      }

      const t = linspace(center-HALF_WINDOW, center+HALF_WINDOW, N);
      const fnRe = makeEvaluator(exprRe);
      if(!fnRe) return {error:'Error evaluating function'};

      let yRe, yIm;
      try {
        yRe = evaluateArray(fnRe, t, argMap);
        if (hasExp) {
          const fnIm = makeEvaluator(exprIm);
          const fnZero = makeEvaluator(exprZero);
          if(!fnIm || !fnZero) return {error:'Error evaluating function'};
          yIm = evaluateImag(fnIm, fnZero, t, argMap);
        } else {
          yIm = new Float64Array(N);
        }
      } catch(e){
        return {error:'Error evaluating function: '+e.message};
      }

      // Modulation property: multiplying by e^(j*w0*t) shifts the spectrum by w0.
      if (modulate !== 0) {
        const mRe = new Float64Array(N);
        const mIm = new Float64Array(N);
        for(let i=0;i<N;i++){
          const c = Math.cos(modulate*t[i]);
          const sn = Math.sin(modulate*t[i]);
          mRe[i] = yRe[i]*c - yIm[i]*sn;
          mIm[i] = yRe[i]*sn + yIm[i]*c;
        }
        yRe = mRe;
        yIm = mIm;
      }

      for(let i=0;i<N;i++){
        if(!Number.isFinite(yRe[i]) || !Number.isFinite(yIm[i])){
          return {error:`Function is not finite at t = ${t[i].toFixed(3)} (division by zero or overflow).`};
        }
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

      // The FFT assumes samples start at t=0 (x[n] corresponds to t=n*dt)
      // Here we sample the continuous signal at t = t0 + n*dt with t0 = t[0]
      // Thereforee the CTFT sample must be multiplied by exp(-jωt0) to reference the correct time origin

      const t0 = t[0];
      if (Math.abs(t0) > 1e-12) {
        for (let i = 0; i < N; i++) {
          const a = shifted.re[i];
          const b = shifted.im[i];
          const theta = -omega[i] * t0;
          const c = Math.cos(theta);
          const s = Math.sin(theta);
          shifted.re[i] = a * c - b * s;
          shifted.im[i] = a * s + b * c;
        }
      }

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
        // Pref +pi over -pi for cleaner step plots
        if (Math.abs(ang + Math.PI) < 1e-12) ang = Math.PI;
        basePhase[i] = ang;
      }

      // A signal that has not died out at the window edge is analysed truncated,
      // so its spectrum shows leakage instead of the exact transform.
      const edgeSpan = Math.max(1, Math.round(0.02*N));
      let edgeMax = 0, signalMax = 0;
      for(let i=0;i<N;i++){
        const a = Math.hypot(yRe[i], yIm[i]);
        if(a > signalMax) signalMax = a;
        if(i < edgeSpan || i >= N - edgeSpan){ if(a > edgeMax) edgeMax = a; }
      }
      const truncated = signalMax > 0 && edgeMax > 0.01*signalMax;

      cachedKey = cacheKey;
      cachedBase = {
        t,
        yRe,
        yIm,
        omega,
        magnitude,
        magnitudeMax: maxMag,
        specRe: shifted.re,
        specIm: shifted.im,
        basePhase,
        lowMagnitude,
        truncated
      };
    }

    if (!cachedBase) {
      return {error:'Error evaluating function'};
    }
    const {t, yRe, yIm, omega, magnitude, magnitudeMax, specRe, specIm,
           basePhase, lowMagnitude, truncated} = cachedBase;

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


    // Below the noise floor the phase is numerically meaningless. A plain array
    // holding null leaves a gap in the plot; writing 0 would read as "the phase
    // is zero here" instead of "there is nothing to read here".
    const phase = new Array(N);
    for(let i=0;i<N;i++){
      if (lowMagnitude[i]) {
        phase[i] = null;
        continue;
      }
      let ang = basePhase[i] + phaseRad;
      if(ang > Math.PI) ang -= 2 * Math.PI;
      if(ang < -Math.PI) ang += 2 * Math.PI;
      // Pref -II-
      if (Math.abs(ang + Math.PI) < 1e-12) ang = Math.PI;
      phase[i] = ang;
    }

    // The global phase rotates the spectrum, so Re and Im follow it as well.
    const spec_re = new Float64Array(N);
    const spec_im = new Float64Array(N);
    for(let i=0;i<N;i++){
      spec_re[i] = specRe[i]*cosPhase - specIm[i]*sinPhase;
      spec_im[i] = specRe[i]*sinPhase + specIm[i]*cosPhase;
    }

    return {
      t,
      y_real: y_re,
      y_imag: y_im,
      omega,
      magnitude,
      magnitudeMax,
      spec_real: spec_re,
      spec_imag: spec_im,
      phase,
      truncated,
      properties: {shift, scale, modulate, phaseDeg}
    };
  }

  window.compute_fourier = compute_fourier;
})();