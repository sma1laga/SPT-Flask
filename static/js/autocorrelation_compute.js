(function () {
  'use strict';

  // We Dont do any time inversion via strings anymore...
  function replaceExp(expr, funcName) {
    return expr.replace(/exp_iwt\s*\(([^)]*)\)/g, funcName + '($1)');
  }

  // Deterministic tau grid for full linear convolution: length = 2N-1, zero at index N-1.
function buildTauGridByLen(dt, n) {
  const mid = (n - 1) / 2;
  const tau = new Float64Array(n);
  for (let i = 0; i < n; i++) tau[i] = (i - mid) * dt; // τ=0 at i=mid
  return tau;
}

  // Main entry point expected by the page.
function compute_autocorrelation(func) {
  // If the user used exp_iwt(...), split it into cos/sin for Re/Im sampling.
  // Otherwise: treat the expression as purely real (Im = 0).
  const usesExp = /exp_iwt\s*\(/.test(func);
  const fRe = usesExp ? replaceExp(func, 'Math.cos') : func;
  const fIm = usesExp ? replaceExp(func, 'Math.sin') : '0';

  // --- Autocorrelation via 4 real convolutions with numeric time-reversal ---
  const rr = compute_convolution(fRe, fRe, true); if (rr.error) return rr;
  const ii = compute_convolution(fIm, fIm, true); if (ii.error) return ii;
  const ri = compute_convolution(fRe, fIm, true); if (ri.error) return ri;
  const ir = compute_convolution(fIm, fRe, true); if (ir.error) return ir;

  const n = rr.y_conv.length;
  const dt = rr.t1[1] - rr.t1[0];
  const r_re = new Float64Array(n);
  const r_im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    r_re[i] = rr.y_conv[i] + ii.y_conv[i];
    r_im[i] = ir.y_conv[i] - ri.y_conv[i];
  }

  const tau = buildTauGridByLen(dt, n);

  return {
    t: rr.t1,
    y_re: rr.y1,
    y_im: ir.y1,
    tau: Array.from(tau),
    r_re: Array.from(r_re),
    r_im: Array.from(r_im)
  };
}


  window.compute_autocorrelation = compute_autocorrelation;
})();
