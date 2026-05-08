(() => {
  const svg = document.getElementById("satcom4-svg");
  if (!svg) return;

  const ecc = document.getElementById("satcom4-ecc");
  const speed = document.getElementById("satcom4-speed");
  const evalText = document.getElementById("satcom4-eval");
  const sval = document.getElementById("satcom4-sval");
  const result = document.getElementById("satcom4-result");
  const play = document.getElementById("satcom4-play");
  const reset = document.getElementById("satcom4-reset");

  let running = true;
  let meanAnomaly = 0;
  let last = null;

  const cx = 420;
  const cy = 260;
  const a = 230;
  const period = 11 * 3600 + 58 * 60 + 2;

  const el = (name, attrs = {}) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    svg.appendChild(node);
    return node;
  };

  const solveKepler = (M, e) => {
    let E = M;
    for (let i = 0; i < 8; i += 1) {
      E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    }
    return E;
  };

  const pos = (M, e) => {
    const b = a * Math.sqrt(1 - e * e);
    const E = solveKepler(M, e);
    return [cx + a * (Math.cos(E) - e), cy + b * Math.sin(E)];
  };

  const hms = (seconds) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.round(seconds % 60);
    return `${h} h ${m} min ${s} s`;
  };

  const draw = (ts) => {
    if (last === null) last = ts;
    const dt = (ts - last) / 1000;
    last = ts;

    const e = parseFloat(ecc.value);
    const b = a * Math.sqrt(1 - e * e);

    if (running) {
      meanAnomaly = (meanAnomaly + dt * parseFloat(speed.value) * 0.52) % (2 * Math.PI);
    }

    svg.innerHTML = "";

    let path = "";
    for (let i = 0; i <= 360; i += 1) {
      const E = i * Math.PI / 180;
      const x = cx + a * (Math.cos(E) - e);
      const y = cy + b * Math.sin(E);
      path += `${i ? " L " : "M "}${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    el("path", { d: `${path} Z`, fill: "none", stroke: "#111827", "stroke-width": 3 });

    let far = "";
    for (let i = 90; i <= 270; i += 1) {
      const E = i * Math.PI / 180;
      const x = cx + a * (Math.cos(E) - e);
      const y = cy + b * Math.sin(E);
      far += `${i > 90 ? " L " : "M "}${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    el("path", { d: far, fill: "none", stroke: "#16a34a", "stroke-width": 8, opacity: 0.35 });

    el("line", { x1: cx, y1: cy - b, x2: cx, y2: cy + b, stroke: "#94a3b8", "stroke-width": 1.5, "stroke-dasharray": "5 5" });
    el("circle", { cx, cy, r: 22, fill: "#dbeafe", stroke: "#111827", "stroke-width": 2 });
    const earthText = el("text", { x: cx - 38, y: cy + 48, "font-size": 13, "font-weight": 700 });
    earthText.textContent = "Earth at focus";

    const apo = pos(Math.PI, e);
    const per = pos(0, e);
    const sat = pos(meanAnomaly, e);

    el("circle", { cx: apo[0], cy: apo[1], r: 6, fill: "#7c3aed" });
    const apoText = el("text", { x: Math.max(apo[0] - 92, 45), y: apo[1] - 12, "font-size": 13, "font-weight": 700 });
    apoText.textContent = "Apogee / slow";

    el("circle", { cx: per[0], cy: per[1], r: 6, fill: "#16a34a" });
    const perText = el("text", { x: Math.min(per[0] + 12, 735), y: per[1] - 12, "font-size": 13, "font-weight": 700 });
    perText.textContent = "Perigee / fast";

    el("circle", { cx: sat[0], cy: sat[1], r: 8, fill: "#f97316", stroke: "white", "stroke-width": 3 });
    const farText = el("text", { x: cx - 95, y: cy + b + 48, "font-size": 14, fill: "#15803d", "font-weight": 700 });
    farText.textContent = "far half: reception assumed";

    const frac = 0.5 + e / Math.PI;
    const time = period * frac;

    evalText.textContent = e.toFixed(2);
    sval.textContent = `${parseFloat(speed.value).toFixed(1)}x`;
    result.innerHTML = `Far-half time fraction:<br><b>1/2 + e/π = ${frac.toFixed(3)}</b><br><br>For T = 11 h 58 min 2 s:<br><b>${hms(time)}</b>`;

    window.requestAnimationFrame(draw);
  };

  play.addEventListener("click", () => {
    running = !running;
    play.textContent = running ? "Pause" : "Play";
  });

  reset.addEventListener("click", () => {
    meanAnomaly = 0;
    running = true;
    play.textContent = "Pause";
  });

  window.requestAnimationFrame(draw);
})();