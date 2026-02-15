(function () {
  'use strict';

  let plotlyPromise;

  function ensurePlotlyLoaded() {
    if (window.Plotly) {
      return Promise.resolve(window.Plotly);
    }

    if (plotlyPromise) {
      return plotlyPromise;
    }

    plotlyPromise = new Promise((resolve, reject) => {
      const existingScript = document.querySelector('script[data-plotly-loader="true"]');
      if (existingScript) {
        existingScript.addEventListener('load', () => resolve(window.Plotly));
        existingScript.addEventListener('error', () => reject(new Error('Failed to load Plotly.')));
        return;
      }

      const script = document.createElement('script');
      script.src = '/static/js/plotly.min.js';
      script.defer = true;
      script.dataset.plotlyLoader = 'true';
      script.onload = function () {
        if (window.Plotly) {
          resolve(window.Plotly);
          return;
        }
        reject(new Error('Plotly script loaded, but Plotly is unavailable.'));
      };
      script.onerror = function () {
        reject(new Error('Failed to load Plotly.'));
      };
      document.head.appendChild(script);
    }).catch((error) => {
      plotlyPromise = undefined;
      throw error;
    });

    return plotlyPromise;
  }

  window.ensurePlotlyLoaded = ensurePlotlyLoaded;
})();