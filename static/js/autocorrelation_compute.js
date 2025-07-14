(function(){
  'use strict';
  function compute_autocorrelation(func){
    return compute_convolution(func, func);
  }
  window.compute_autocorrelation = compute_autocorrelation;
})();