(() => {
  const EARTH_RADIUS_KM = 6371;
  const MIN_ELEVATION_DEG = 10;
  const ORBITS = [
    { id: "A", label: "Orbit A", m: 1, r: 11543, periodLabel: "T ≈ 205.7 min" },
    { id: "B", label: "Orbit B", m: 2, r: 7640, periodLabel: "T ≈ 110.8 min" },
  ];

  const svg = document.getElementById("satcom-vis-svg");
  const choices = document.getElementById("satcom-vis-choices");
  const playBtn = document.getElementById("satcom-vis-play");
  const pauseBtn = document.getElementById("satcom-vis-pause");
  const resetBtn = document.getElementById("satcom-vis-reset");
  const speedInput = document.getElementById("satcom-vis-speed");
  const minElevEl = document.getElementById("satcom-vis-min-elev");
  const psiEl = document.getElementById("satcom-vis-psi");
  const windowEl = document.getElementById("satcom-vis-window");
  const timeEl = document.getElementById("satcom-vis-time");

  if (!svg || !choices || !playBtn || !pauseBtn || !resetBtn || !speedInput) return;

  let orbitIndex = 0;
  let isPlaying = true;
  let speed = Number(speedInput.value);
  let deltaLambda = -90;
  let lastTs = null;

  function isDarkMode() {
    return document.body.classList.contains("dark-mode");
  }

  function palette() {
    const dark = isDarkMode();
    return {
      orbit: dark ? "#475569" : "#d4d4d8",
      earthFill: dark ? "#1e293b" : "#f5f5f5",
      earthStroke: dark ? "#cbd5e1" : "#52525b",
      equator: dark ? "#475569" : "#9ca3af",
      muted: dark ? "#94a3b8" : "#52525b",
      text: dark ? "#f8fafc" : "#111827",
      gridText: dark ? "#94a3b8" : "#6b7280",
      visibilityArc: dark ? "#64748b" : "#cbd5e1",
      delta: dark ? "#fbbf24" : "#d97706",
      deltaText: dark ? "#fcd34d" : "#b45309",
      psi: dark ? "#94a3b8" : "#64748b",
      psiText: dark ? "#cbd5e1" : "#475569",
      epsilon: dark ? "#c4b5fd" : "#7c3aed",
      epsilonText: dark ? "#ddd6fe" : "#6d28d9",
      satellite: dark ? "#60a5fa" : "#2563eb",
      visible: dark ? "#4ade80" : "#15803d",
      linkHidden: dark ? "#64748b" : "#9ca3af",
    };
  }

  function degToRad(deg) {
    return (deg * Math.PI) / 180;
  }

  function radToDeg(rad) {
    return (rad * 180) / Math.PI;
  }

  function pointOnCircle(cx, cy, r, deg) {
    const angle = degToRad(deg);
    return { x: cx + r * Math.cos(angle), y: cy - r * Math.sin(angle) };
  }

  function polyArc(cx, cy, r, startDeg, endDeg, steps = 28) {
    const parts = [];
    for (let i = 0; i <= steps; i += 1) {
      const t = i / steps;
      const deg = startDeg + (endDeg - startDeg) * t;
      const p = pointOnCircle(cx, cy, r, deg);
      parts.push(`${i === 0 ? "M" : "L"} ${p.x} ${p.y}`);
    }
    return parts.join(" ");
  }

  function setText(element, value) {
    if (element) element.textContent = value;
  }

  function updateChoiceButtons() {
    [...choices.children].forEach((choice, index) => {
      choice.className = `satcom-vis-choice${index === orbitIndex ? " is-active" : ""}`;
      choice.setAttribute("aria-pressed", index === orbitIndex ? "true" : "false");
    });
  }

  function addChoiceButton(item, index) {
    const button = document.createElement("button");
    button.className = `satcom-vis-choice${index === orbitIndex ? " is-active" : ""}`;
    button.type = "button";
    button.setAttribute("aria-pressed", index === orbitIndex ? "true" : "false");

    const title = document.createElement("div");
    title.className = "satcom-vis-choice-title";
    title.textContent = item.label;

    const mValue = document.createElement("div");
    mValue.className = "satcom-vis-muted";
    mValue.textContent = `m = ${item.m}`;

    const period = document.createElement("div");
    period.className = "satcom-vis-muted";
    period.textContent = item.periodLabel;

    button.append(title, mValue, period);
    button.addEventListener("click", () => {
      orbitIndex = index;
      deltaLambda = -90;
      lastTs = null;
      updateChoiceButtons();
      render();
    });
    choices.appendChild(button);
  }

  function render() {
    const orbit = ORBITS[orbitIndex];
    const colors = palette();
    const psi = radToDeg(
      Math.acos((EARTH_RADIUS_KM / orbit.r) * Math.cos(degToRad(MIN_ELEVATION_DEG))) - degToRad(MIN_ELEVATION_DEG)
    );
    const omegaRel = 90 * orbit.m;
    const visibleTimeMin = ((2 * psi) / omegaRel) * 60;
    const isVisible = Math.abs(deltaLambda) <= psi;
    const cx = 240;
    const cy = 220;
    const earthR = 110;
    const orbitR = 175;
    const center = { x: cx, y: cy };
    const ground = { x: cx + earthR, y: cy };
    const satellite = pointOnCircle(cx, cy, orbitR, deltaLambda);
    const leftBoundary = pointOnCircle(cx, cy, orbitR, -psi);
    const rightBoundary = pointOnCircle(cx, cy, orbitR, psi);
    const losDeg = radToDeg(Math.atan2(-(satellite.y - ground.y), satellite.x - ground.x));
    const epsStart = Math.min(90, losDeg);
    const epsEnd = Math.max(90, losDeg);
    const epsLabel = pointOnCircle(ground.x, ground.y, 44, (epsStart + epsEnd) / 2);
    const deltaLabel = pointOnCircle(cx, cy, 92, deltaLambda / 2);
    const psiLeftLabel = pointOnCircle(cx, cy, 74, -psi / 2);
    const psiRightLabel = pointOnCircle(cx, cy, 74, psi / 2);

    svg.innerHTML = `
      <circle cx="${cx}" cy="${cy}" r="${orbitR}" fill="none" stroke="${colors.orbit}" stroke-width="2"/>
      <circle cx="${cx}" cy="${cy}" r="${earthR}" fill="${colors.earthFill}" stroke="${colors.earthStroke}" stroke-width="2"/>
      <line x1="${cx - 205}" y1="${cy}" x2="${cx + 205}" y2="${cy}" stroke="${colors.equator}" stroke-width="2" stroke-dasharray="8 6"/>
      <text x="${cx + 100}" y="${cy - 10}" font-size="13" fill="${colors.gridText}">equatorial plane</text>
      <path d="${polyArc(cx, cy, orbitR, -psi, psi, 36)}" fill="none" stroke="${colors.visibilityArc}" stroke-width="10" stroke-linecap="round" opacity="0.9"/>
      <path d="${polyArc(cx, cy, 82, Math.min(0, deltaLambda), Math.max(0, deltaLambda), 24)}" fill="none" stroke="${colors.delta}" stroke-width="2.5"/>
      <path d="${polyArc(cx, cy, 64, -psi, 0, 20)}" fill="none" stroke="${colors.psi}" stroke-width="2" stroke-dasharray="4 4"/>
      <path d="${polyArc(cx, cy, 64, 0, psi, 20)}" fill="none" stroke="${colors.psi}" stroke-width="2" stroke-dasharray="4 4"/>
      <path d="${polyArc(ground.x, ground.y, 34, epsStart, epsEnd, 16)}" fill="none" stroke="${colors.epsilon}" stroke-width="2.6"/>
      <line x1="${center.x}" y1="${center.y}" x2="${ground.x}" y2="${ground.y}" stroke="${colors.text}" stroke-width="2.5"/>
      <line x1="${center.x}" y1="${center.y}" x2="${satellite.x}" y2="${satellite.y}" stroke="${colors.satellite}" stroke-width="2.5"/>
      <line x1="${ground.x}" y1="${ground.y}" x2="${satellite.x}" y2="${satellite.y}" stroke="${isVisible ? colors.visible : colors.linkHidden}" stroke-width="2.2" stroke-dasharray="7 5"/>
      <line x1="${center.x}" y1="${center.y}" x2="${leftBoundary.x}" y2="${leftBoundary.y}" stroke="${colors.psi}" stroke-width="1.7" stroke-dasharray="5 4"/>
      <line x1="${center.x}" y1="${center.y}" x2="${rightBoundary.x}" y2="${rightBoundary.y}" stroke="${colors.psi}" stroke-width="1.7" stroke-dasharray="5 4"/>
      <line x1="${ground.x}" y1="${ground.y}" x2="${ground.x}" y2="${ground.y - 80}" stroke="${colors.epsilon}" stroke-width="1.8" stroke-dasharray="5 4"/>
      <circle cx="${ground.x}" cy="${ground.y}" r="6" fill="${colors.text}"/>
      <circle cx="${satellite.x}" cy="${satellite.y}" r="7" fill="${isVisible ? colors.visible : colors.satellite}"/>
      <circle cx="${center.x}" cy="${center.y}" r="3.5" fill="${colors.text}"/>
      <text x="${center.x - 18}" y="${center.y + 5}" font-size="14" fill="${colors.text}">O</text>
      <text x="${ground.x + 10}" y="${ground.y + 2}" font-size="14" fill="${colors.text}">G</text>
      <text x="${satellite.x + 10}" y="${satellite.y - 6}" font-size="14" fill="${isVisible ? colors.visible : colors.satellite}">S</text>
      <text x="${leftBoundary.x - 16}" y="${leftBoundary.y - 10}" font-size="12" fill="${colors.psi}">-ψ</text>
      <text x="${rightBoundary.x + 6}" y="${rightBoundary.y - 10}" font-size="12" fill="${colors.psi}">+ψ</text>
      <text x="${psiLeftLabel.x - 10}" y="${psiLeftLabel.y - 6}" font-size="13" fill="${colors.psiText}">ψ</text>
      <text x="${psiRightLabel.x + 6}" y="${psiRightLabel.y - 6}" font-size="13" fill="${colors.psiText}">ψ</text>
      <text x="${deltaLabel.x + 8}" y="${deltaLabel.y - 6}" font-size="13" fill="${colors.deltaText}">Δλ</text>
      <text x="${epsLabel.x + 6}" y="${epsLabel.y - 4}" font-size="15" fill="${colors.epsilonText}">ε</text>
      <text x="16" y="24" font-size="13" fill="${colors.muted}">Equatorial visibility geometry</text>
      <text x="16" y="42" font-size="13" fill="${colors.muted}">ε is measured from the local tangent at G</text>`;

    setText(minElevEl, `${MIN_ELEVATION_DEG.toFixed(0)}°`);
    setText(psiEl, `${psi.toFixed(1)}°`);
    setText(windowEl, `${(2 * psi).toFixed(1)}°`);
    setText(timeEl, `${visibleTimeMin.toFixed(1)} min`);
  }

  ORBITS.forEach(addChoiceButton);

  playBtn.addEventListener("click", () => {
    isPlaying = true;
  });

  pauseBtn.addEventListener("click", () => {
    isPlaying = false;
  });

  resetBtn.addEventListener("click", () => {
    deltaLambda = -90;
    lastTs = null;
    render();
  });

  speedInput.addEventListener("input", () => {
    speed = Number(speedInput.value);
  });

  const observer = new MutationObserver(render);
  observer.observe(document.body, { attributes: true, attributeFilter: ["class"] });

  function step(ts) {
    if (lastTs === null) lastTs = ts;
    const dt = (ts - lastTs) / 1000;
    lastTs = ts;
    const orbit = ORBITS[orbitIndex];
    const omegaRel = 90 * orbit.m;
    if (isPlaying) {
      deltaLambda += omegaRel * speed * dt;
      if (deltaLambda > 120) deltaLambda = -120;
    }
    render();
    requestAnimationFrame(step);
  }

  updateChoiceButtons();
  render();
  requestAnimationFrame(step);
})();