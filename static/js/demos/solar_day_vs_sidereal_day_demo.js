(() => {
  const svg = document.getElementById('satcom6-svg');
  const progress = document.getElementById('satcom6-progress');
  const speed = document.getElementById('satcom6-speed');
  const playButton = document.getElementById('satcom6-play');
  const resetButton = document.getElementById('satcom6-reset');

  if (!svg || !progress || !speed || !playButton || !resetButton) {
    return;
  }

  const pval = document.getElementById('satcom6-pval');
  const sval = document.getElementById('satcom6-sval');
  const msg = document.getElementById('satcom6-msg');

  let running = true;
  let last = null;

  function el(name, attrs) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', name);
    Object.keys(attrs).forEach((key) => node.setAttribute(key, attrs[key]));
    svg.appendChild(node);
    return node;
  }

  function draw(ts) {
    const isDark = document.body.classList.contains('dark-mode');
    const palette = isDark
      ? {
          caption: '#9fb8df',
          orbit: '#9fb0c8',
          starLine: '#7fb0ff',
          starText: '#8fbeff',
          sunLine: '#ff7b7b',
          sunText: '#ff9b9b',
          earthText: '#0f172a',
          marker: '#dbe7ff',
          timeline: '#b7c4d8',
          start: '#dbe7ff',
          sidereal: '#6ea8ff',
          solar: '#ff7b7b',
        }
      : {
          caption: '#4b5563',
          orbit: '#cbd5e1',
          starLine: '#2563eb',
          starText: '#1d4ed8',
          sunLine: '#dc2626',
          sunText: '#b91c1c',
          earthText: '#111827',
          marker: '#111827',
          timeline: '#94a3b8',
          start: '#111827',
          sidereal: '#2563eb',
          solar: '#dc2626',
        };

    if (last === null) last = ts;
    const dt = (ts - last) / 1000;
    last = ts;

    if (running) {
      progress.value = Math.min(1, parseFloat(progress.value) + dt * parseFloat(speed.value));
    }

    const p = parseFloat(progress.value);
    const sun = { x: 120, y: 280 };
    const orbitCenter = { x: 520, y: 280 };
    const orbitRadius = 185;
    const earthRadius = 38;

    const theta = ((-12 + p * 22) * Math.PI) / 180;
    const earth = {
      x: orbitCenter.x + orbitRadius * Math.cos(theta),
      y: orbitCenter.y + orbitRadius * Math.sin(theta),
    };

    const rotation = ((-90 + p * 360.986) * Math.PI) / 180;
    const marker = {
      x: earth.x + earthRadius * Math.cos(rotation),
      y: earth.y + earthRadius * Math.sin(rotation),
    };

    const sidereal = ((-90 + p * 360) * Math.PI) / 180;
    const siderealPoint = {
      x: earth.x + earthRadius * Math.cos(sidereal),
      y: earth.y + earthRadius * Math.sin(sidereal),
    };

    const sunDirection = Math.atan2(sun.y - earth.y, sun.x - earth.x);

    svg.innerHTML = '';

    el('text', { x: 28, y: 34, 'font-size': 14, fill: palette.caption }).textContent =
      'Earth orbit motion is exaggerated so the difference is visible.';

    el('circle', { cx: sun.x, cy: sun.y, r: 42, fill: '#fef3c7', stroke: '#f59e0b', 'stroke-width': 2 });
    el('text', { x: sun.x, y: sun.y + 6, 'text-anchor': 'middle', 'font-size': 18, 'font-weight': 700, fill: '#111827' }).textContent = 'Sun';

    el('circle', { cx: orbitCenter.x, cy: orbitCenter.y, r: orbitRadius, fill: 'none', stroke: palette.orbit, 'stroke-width': 2, 'stroke-dasharray': '6 6' });

    el('line', { x1: earth.x, y1: earth.y, x2: earth.x, y2: earth.y - 115, stroke: palette.starLine, 'stroke-width': 2 });
    el('text', { x: Math.min(earth.x + 10, 760), y: earth.y - 110, 'font-size': 13, fill: palette.starText, 'font-weight': 600 }).textContent = 'fixed star direction';

    el('line', {
      x1: earth.x,
      y1: earth.y,
      x2: earth.x + 100 * Math.cos(sunDirection),
      y2: earth.y + 100 * Math.sin(sunDirection),
      stroke: palette.sunLine,
      'stroke-width': 2,
    });

    el('text', {
      x: Math.max(40, Math.min(earth.x + 60 * Math.cos(sunDirection) - 55, 760)),
      y: earth.y + 60 * Math.sin(sunDirection) - 8,
      'font-size': 13,
      fill: palette.sunText,
      'font-weight': 600,
    }).textContent = 'direction to Sun';

    el('circle', { cx: earth.x, cy: earth.y, r: earthRadius, fill: '#dbeafe', stroke: '#1d4ed8', 'stroke-width': 2 });
    el('text', { x: earth.x, y: earth.y + 5, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: palette.earthText }).textContent = 'Earth';

    el('line', { x1: earth.x, y1: earth.y, x2: marker.x, y2: marker.y, stroke: palette.marker, 'stroke-width': 2 });
    el('circle', { cx: marker.x, cy: marker.y, r: 6, fill: palette.marker });

    if (p > 0.93) {
      el('circle', { cx: siderealPoint.x, cy: siderealPoint.y, r: 6, fill: palette.sidereal, opacity: 0.75 });
      el('text', { x: siderealPoint.x + 12, y: siderealPoint.y + 22, 'font-size': 12, fill: palette.starText }).textContent = 'sidereal-day marker';
    }

    el('line', { x1: 120, y1: 505, x2: 790, y2: 505, stroke: palette.timeline, 'stroke-width': 2 });

    [
      ['start', 120, palette.start, 530],
      ['sidereal day', 680, palette.sidereal, 530],
      ['solar day', 770, palette.solar, 550],
    ].forEach(([text, x, color, y]) => {
      el('circle', { cx: x, cy: 505, r: 5, fill: color });
      el('text', { x, y, 'text-anchor': 'middle', 'font-size': 13, fill: color }).textContent = text;
    });

    el('text', { x: 720, y: 470, 'text-anchor': 'middle', 'font-size': 13, fill: palette.solar }).textContent = 'extra ≈ 3 min 56 s';

    pval.textContent = `${(p * 24).toFixed(2)} h`;
    sval.textContent = `${parseFloat(speed.value).toFixed(2)} day/s`;
    if (p < 0.98) {
      msg.textContent = 'Earth rotates while also moving around the Sun.';
    } else if (p < 0.999) {
      msg.textContent = 'After one sidereal rotation, Earth still needs a little extra turn to face the Sun again.';
    } else {
      msg.textContent = 'After one solar day, the same point faces the Sun again.';
    }

    requestAnimationFrame(draw);
  }

  playButton.onclick = () => {
    running = !running;
    playButton.textContent = running ? 'Pause' : 'Play';
  };

  resetButton.onclick = () => {
    progress.value = 0;
    running = true;
    playButton.textContent = 'Pause';
  };

  requestAnimationFrame(draw);
})();