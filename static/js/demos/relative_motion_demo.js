(() => {
  const EARTH_RATE = 15;
  const ORBITS = [
    { m: 1, label: "m = 1", periodH: 24 / 7, note: "slower candidate" },
    { m: 2, label: "m = 2", periodH: 24 / 13, note: "faster candidate" },
  ];

  const svg = document.getElementById("satcom-rel-svg");
  const choices = document.getElementById("satcom-rel-choices");
  const playBtn = document.getElementById("satcom-rel-play");
  const resetBtn = document.getElementById("satcom-rel-reset");
  const speedInput = document.getElementById("satcom-rel-speed");
  const earthRateEl = document.getElementById("satcom-rel-earth-rate");
  const satRateEl = document.getElementById("satcom-rel-sat-rate");
  const relRateEl = document.getElementById("satcom-rel-rel-rate");
  const gapEl = document.getElementById("satcom-rel-gap");
  const gain4hEl = document.getElementById("satcom-rel-gain4h");
  const timelineMarker = document.getElementById("satcom-rel-timeline-marker");

  if (!svg || !choices || !playBtn || !resetBtn || !speedInput) return;

  let choice = 0;
  let timeH = 0;
  let playing = true;
  let speed = Number(speedInput.value);
  let lastTs = null;

  function isDarkMode() {
    return document.body.classList.contains("dark-mode");
  }

  function palette() {
    const dark = isDarkMode();
    return {
      orbit: dark ? "#64748b" : "#d4d4d8",
      earthFill: dark ? "#1e293b" : "#f5f5f5",
      earthStroke: dark ? "#cbd5e1" : "#52525b",
      grid: dark ? "#334155" : "#e5e7eb",
      station: dark ? "#f8fafc" : "#111827",
      satellite: dark ? "#60a5fa" : "#2563eb",
      satelliteText: dark ? "#93c5fd" : "#1d4ed8",
      satRadius: dark ? "#cbd5e1" : "#4b5563",
      link: dark ? "#94a3b8" : "#94a3b8",
      pass: dark ? "#4ade80" : "#15803d",
      arc: dark ? "#fbbf24" : "#d97706",
      arcText: dark ? "#fcd34d" : "#b45309",
      muted: dark ? "#94a3b8" : "#52525b",
    };
  }

  function createSvgElement(name, attrs = {}) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    svg.appendChild(node);
    return node;
  }

  function normDeg(deg) {
    let x = deg % 360;
    if (x < 0) x += 360;
    return x;
  }

  function shortestSignedDiff(a, b) {
    let d = normDeg(a - b);
    if (d > 180) d -= 360;
    return d;
  }

  function degToRad(deg) {
    return (deg * Math.PI) / 180;
  }

  function formatHours(hours) {
    const hrs = Math.floor(hours);
    const mins = Math.round((hours - hrs) * 60);
    if (hrs === 0) return `${mins} min`;
    if (mins === 0) return `${hrs} h`;
    return `${hrs} h ${mins} min`;
  }

  function polarPoint(cx, cy, r, deg) {
    const angle = degToRad(-90 + deg);
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
    };
  }

  function arcPath(cx, cy, r, startDeg, endDeg) {
    const p1 = polarPoint(cx, cy, r, startDeg);
    const p2 = polarPoint(cx, cy, r, endDeg);
    const delta = ((endDeg - startDeg) % 360 + 360) % 360;
    const largeArc = delta > 180 ? 1 : 0;
    return `M ${p1.x} ${p1.y} A ${r} ${r} 0 ${largeArc} 1 ${p2.x} ${p2.y}`;
  }

  function setText(element, value) {
    if (element) element.textContent = value;
  }

  function updateChoiceButtons() {
    [...choices.children].forEach((button, i) => {
      button.classList.toggle("is-active", i === choice);
      button.setAttribute("aria-pressed", i === choice ? "true" : "false");
    });
  }

  function addChoiceButton(orbit, index) {
    const button = document.createElement("button");
    button.className = `satcom-rel-choice${index === choice ? " is-active" : ""}`;
    button.type = "button";
    button.setAttribute("aria-pressed", index === choice ? "true" : "false");

    const title = document.createElement("div");
    title.className = "satcom-rel-choice-title";
    title.textContent = orbit.label;

    const period = document.createElement("div");
    period.className = "satcom-rel-muted";
    period.textContent = `T = ${formatHours(orbit.periodH)}`;

    const note = document.createElement("div");
    note.className = "satcom-rel-muted satcom-rel-note";
    note.textContent = orbit.note;

    button.append(title, period, note);
    button.addEventListener("click", () => {
      choice = index;
      updateChoiceButtons();
      render();
    });
    choices.appendChild(button);
  }

  function drawText(text, attrs) {
    const textNode = createSvgElement("text", attrs);
    textNode.textContent = text;
  }

  function render() {
    const orbit = ORBITS[choice];
    const satRate = 360 / orbit.periodH;
    const relRate = satRate - EARTH_RATE;
    const stationLon = normDeg(EARTH_RATE * timeH);
    const satLon = normDeg(satRate * timeH);
    const relDiff = shortestSignedDiff(satLon, stationLon);
    const isPass = Math.abs(relDiff) < 6;
    const cx = 210;
    const cy = 210;
    const earthR = 115;
    const orbitR = 165;
    const station = polarPoint(cx, cy, earthR, stationLon);
    const sat = polarPoint(cx, cy, orbitR, satLon);
    const arcEnd = relDiff >= 0 ? relDiff : 360 + relDiff;
    const arcLabel = polarPoint(cx, cy, 82, relDiff >= 0 ? relDiff / 2 : (360 + relDiff) / 2);
    const colors = palette();

    svg.innerHTML = "";
    createSvgElement("circle", { cx, cy, r: orbitR, fill: "none", stroke: colors.orbit, "stroke-width": 2 });
    createSvgElement("circle", { cx, cy, r: earthR, fill: colors.earthFill, stroke: colors.earthStroke, "stroke-width": 2 });
    createSvgElement("line", { x1: cx - 190, y1: cy, x2: cx + 190, y2: cy, stroke: colors.grid, "stroke-dasharray": "6 6" });
    createSvgElement("line", { x1: cx, y1: cy - 190, x2: cx, y2: cy + 190, stroke: colors.grid, "stroke-dasharray": "6 6" });
    createSvgElement("line", { x1: cx, y1: cy, x2: station.x, y2: station.y, stroke: colors.station, "stroke-width": 2.5 });
    createSvgElement("line", { x1: cx, y1: cy, x2: sat.x, y2: sat.y, stroke: colors.satRadius, "stroke-width": 2.5 });
    createSvgElement("line", {
      x1: station.x,
      y1: station.y,
      x2: sat.x,
      y2: sat.y,
      stroke: isPass ? colors.pass : colors.link,
      "stroke-width": 2,
      "stroke-dasharray": "7 5",
    });
    createSvgElement("path", { d: arcPath(cx, cy, 78, 0, arcEnd), fill: "none", stroke: colors.arc, "stroke-width": 2.5 });
    createSvgElement("circle", { cx: station.x, cy: station.y, r: 6, fill: colors.station });
    createSvgElement("circle", { cx: sat.x, cy: sat.y, r: 7, fill: colors.satellite });
    createSvgElement("circle", { cx, cy, r: 3.5, fill: colors.station });

    drawText("Earth", { x: cx - 18, y: cy + 5, "font-size": 14, fill: colors.station });
    drawText("station", { x: station.x + 10, y: station.y - 6, "font-size": 14, fill: colors.station });
    drawText("satellite", { x: sat.x + 10, y: sat.y - 6, "font-size": 14, fill: colors.satelliteText });
    drawText("Δlongitude", { x: arcLabel.x + 8, y: arcLabel.y - 8, "font-size": 13, fill: colors.arcText });
    drawText("Ground-fixed view", { x: 18, y: 24, "font-size": 13, fill: colors.muted });
    drawText(`Time: ${timeH.toFixed(2)} h`, { x: 18, y: 42, "font-size": 13, fill: colors.muted });

    setText(earthRateEl, `${EARTH_RATE.toFixed(1)}°/h`);
    setText(satRateEl, `${satRate.toFixed(1)}°/h`);
    setText(relRateEl, `${relRate.toFixed(1)}°/h`);
    setText(gapEl, `${relDiff.toFixed(1)}°`);
    setText(gain4hEl, `${(relRate * 4).toFixed(0)}° = ${orbit.m}·360°`);
    if (timelineMarker) timelineMarker.style.left = `calc(16px + ${timeH / 8} * (100% - 32px))`;
    playBtn.textContent = playing ? "Pause" : "Play";
  }

  ORBITS.forEach(addChoiceButton);

  playBtn.addEventListener("click", () => {
    playing = !playing;
    render();
  });

  resetBtn.addEventListener("click", () => {
    timeH = 0;
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
    if (playing) {
      timeH += dt * speed;
      if (timeH > 8) timeH -= 8;
    }
    render();
    requestAnimationFrame(step);
  }

  render();
  requestAnimationFrame(step);
})();