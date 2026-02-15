(function () {
  const tooltipSelector = '[data-tooltip], .tooltip, [onclick*="toggleTooltip"]';
  const hasTooltips = document.querySelector(tooltipSelector);

  if (!hasTooltips) {
    return;
  }

  const loaderScript = document.currentScript;
  const tooltipCssHref = loaderScript?.dataset.tooltipCss || '/static/css/tooltips.css';
  const tooltipJsSrc = loaderScript?.dataset.tooltipJs || '/static/js/tooltips.js';

  const cssAlreadyLoaded = Array.from(document.querySelectorAll('link[rel="stylesheet"], link[rel="preload"][as="style"]'))
    .some((link) => (link.href || '').includes('/static/css/tooltips.css'));
  if (!cssAlreadyLoaded) {
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = tooltipCssHref;
    document.head.appendChild(css);
  }

  const jsAlreadyLoaded = Array.from(document.querySelectorAll('script[src]'))
    .some((script) => (script.src || '').includes('/static/js/tooltips.js'));
  if (!jsAlreadyLoaded) {
    const script = document.createElement('script');
    script.src = tooltipJsSrc;
    script.defer = true;
    document.head.appendChild(script);
  }
})();