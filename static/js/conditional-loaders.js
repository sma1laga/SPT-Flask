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
    css.rel = 'preload';
    css.as = 'style';
    css.href = tooltipCssHref;
    css.onload = function () {
      this.onload = null;
      this.rel = 'stylesheet';
    };
    document.head.appendChild(css);

    const noscript = document.createElement('noscript');
    noscript.innerHTML = `<link rel="stylesheet" href="${tooltipCssHref}">`;
    document.head.appendChild(noscript);
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