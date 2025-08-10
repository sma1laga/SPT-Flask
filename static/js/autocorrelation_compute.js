(function(){
  'use strict';
  
  function reverseT(expr){
    return expr.replace(/\bt\b/g, '(-t)');
  }
  function replaceExp(expr, funcName){
    return expr.replace(/exp_iwt\s*\(([^)]*)\)/g, funcName + '($1)');
  }

  function symmetrize(re, im){
    const n = re.length;
    for(let i=0;i<Math.floor(n/2);i++){
      const j = n-1-i;
      const reSym = 0.5*(re[i] + re[j]);
      const imSym = 0.5*(im[i] - im[j]);
      re[i] = re[j] = reSym;
      im[i] = imSym;
      im[j] = -imSym;
    }
  }

  function shiftAxis(t){
    const n = t.length;
    const zero = t[Math.floor(n/2)];
    const out = new Float64Array(n);
    for(let i=0;i<n;i++) out[i] = t[i] - zero;
    return out;
  }

  function compute_autocorrelation(func){
    const fRe = replaceExp(func, 'Math.cos');
    const fIm = replaceExp(func, 'Math.sin');

    const rr = compute_convolution(fRe, reverseT(fRe));
    if(rr.error) return rr;
    const ii = compute_convolution(fIm, reverseT(fIm));
    const ri = compute_convolution(fRe, reverseT(fIm));
    const ir = compute_convolution(fIm, reverseT(fRe));

    const n = rr.y_conv.length;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for(let i=0;i<n;i++){
      re[i] = rr.y_conv[i] + ii.y_conv[i];
      im[i] = ir.y_conv[i] - ri.y_conv[i];
    }
    symmetrize(re, im);
    const tau = shiftAxis(rr.t_conv);

    return {
      t: rr.t1,
      y_re: rr.y1,
      y_im: ir.y1,
      tau: Array.from(tau),
      r_re: Array.from(re),
      r_im: Array.from(im)
    };
    }
  
  window.compute_autocorrelation = compute_autocorrelation;
})();