(function(){
  'use strict';
  
  function reverseT(expr){
    return expr.replace(/\bt\b/g, '(-t)');
  }

  function compute_autocorrelation(func){
    return compute_convolution(func, reverseT(func));
  }
  
  window.compute_autocorrelation = compute_autocorrelation;
})();