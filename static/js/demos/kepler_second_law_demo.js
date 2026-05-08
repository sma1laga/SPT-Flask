(() => {
  const svg = document.getElementById("satcom-kepler2-svg");
  if (!svg) return;

  let running = true;
  let meanAnomaly = 0;
  let last = null;

  const baseOx = 410;
  const oy = 260;
  const a = 230;

  const ecc = document.getElementById("satcom2-ecc");
  const speed = document.getElementById("satcom2-speed");
  const evalText = document.getElementById("satcom2-eval");
  const sval = document.getElementById("satcom2-sval");
  const play = document.getElementById("satcom2-play");
  const reset = document.getElementById("satcom2-reset");

  const sectors = [];
  const ticks = [];
  const lines = [];


  const isDarkMode = () => document.body.classList.contains("dark-mode");
  const getPalette = () => {
    if (isDarkMode()) {
      return {
        sectorA: "#1e3a8a",
        sectorB: "#334155",
        sectorStroke: "#60a5fa",
        orbit: "#93c5fd",
        center: "#94a3b8",
        centerText: "#cbd5e1",
        earthFill: "#bfdbfe",
        earthStroke: "#0f172a",
        earthText: "#e2e8f0",
        pointText: "#e5e7eb",
        tick: "#93c5fd",
        tickText: "#cbd5e1",
        radial: "#94a3b8",
        satStroke: "#f8fafc"
      };
    }
    return {
      sectorA: "#dbeafe",
      sectorB: "#f8fafc",
      sectorStroke: "#bfdbfe",
      orbit: "#111827",
      center: "#94a3b8",
      centerText: "#6b7280",
      earthFill: "#e0f2fe",
      earthStroke: "#111827",
      earthText: "#111827",
      pointText: "#111827",
      tick: "#475569",
      tickText: "#6b7280",
      radial: "#94a3b8",
      satStroke: "white"
    };
  };

  const el = (name, attrs = {}) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    svg.appendChild(node);
    return node;
  };

  for (let i = 0; i < 12; i += 1) {
    sectors.push(el("path", { "stroke-width": 1, opacity: 0.75 }));
  }

  const orbit = el("path", { fill: "none", "stroke-width": 3 });
  const center = el("circle", { r: 3 });
  const centerText = el("text", { "font-size": 14, "font-weight": "700", "text-anchor": "middle", "paint-order": "stroke", "stroke-width": 3 });
  centerText.textContent = "ellipse center";
  const earth = el("circle", { r: 22, "stroke-width": 2 });
  const earthText = el("text", { "font-size": 14, "font-weight": "700" });
  earthText.textContent = "Earth at focus";
  const per = el("circle", { r: 6, fill: "#16a34a" });
  const apo = el("circle", { r: 6, fill: "#7c3aed" });
  const perText = el("text", { "font-size": 14, "font-weight": "700" });
  const apoText = el("text", { "font-size": 14, "font-weight": "700" });
  perText.textContent = "Perigee: fastest";
  apoText.textContent = "Apogee: slowest";

  for (let i = 0; i < 12; i += 1) {
    lines.push(el("line", { "stroke-width": 1, opacity: 0.65 }));
    ticks.push(el("circle", { r: 4 }));
    const t = el("text", { "font-size": 11 });
    t.textContent = `t${i}`;
    ticks.push(t);
  }

  const sat = el("circle", { r: 8, fill: "#f97316", "stroke-width": 3 });

  const solveKepler = (M, e) => {
    let E = M;
    for (let i = 0; i < 8; i += 1) {
      E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    }
    return E;
  };

  const pos = (M, e, ox) => {
    const b = a * Math.sqrt(1 - e * e);
    const E = solveKepler(M, e);
    return [ox + a * (Math.cos(E) - e), oy + b * Math.sin(E)];
  };

  const draw = (ts) => {
    if (last === null) last = ts;
    const dt = (ts - last) / 1000;
    last = ts;

    const palette = getPalette();

    orbit.setAttribute("stroke", palette.orbit);
    center.setAttribute("fill", palette.center);
    centerText.setAttribute("fill", palette.centerText);
    centerText.setAttribute("stroke", isDarkMode() ? "#0f172a" : "#f8fafc");
    earth.setAttribute("fill", palette.earthFill);
    earth.setAttribute("stroke", palette.earthStroke);
    earthText.setAttribute("fill", palette.earthText);
    perText.setAttribute("fill", palette.pointText);
    apoText.setAttribute("fill", palette.pointText);
    sat.setAttribute("stroke", palette.satStroke);

    for (let i = 0; i < 12; i += 1) {
      sectors[i].setAttribute("fill", i % 2 ? palette.sectorB : palette.sectorA);
      sectors[i].setAttribute("stroke", palette.sectorStroke);
      lines[i].setAttribute("stroke", palette.radial);
      ticks[2 * i].setAttribute("fill", palette.tick);
      ticks[2 * i + 1].setAttribute("fill", palette.tickText);
    }

    const e = parseFloat(ecc.value);
    const b = a * Math.sqrt(1 - e * e);
    const ox = baseOx + Math.max(0, (e - 0.62) / (0.78 - 0.62)) * 32;

    if (running) meanAnomaly = (meanAnomaly + dt * parseFloat(speed.value) * 0.52) % (2 * Math.PI);

    let p = "";
    for (let i = 0; i <= 360; i += 1) {
      const E = i * Math.PI / 180;
      const x = ox + a * (Math.cos(E) - e);
      const y = oy + b * Math.sin(E);
      p += `${i ? " L " : "M "}${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    orbit.setAttribute("d", `${p} Z`);

    for (let k = 0; k < 12; k += 1) {
      const pts = [];
      for (let j = 0; j <= 18; j += 1) {
        const q = pos((k + j / 18) * 2 * Math.PI / 12, e, ox);
        pts.push(`${q[0].toFixed(1)},${q[1].toFixed(1)}`);
      }
      sectors[k].setAttribute("d", `M ${ox},${oy} L ${pts.join(" L ")} Z`);

      const q = pos(k * 2 * Math.PI / 12, e, ox);
      lines[k].setAttribute("x1", ox);
      lines[k].setAttribute("y1", oy);
      lines[k].setAttribute("x2", q[0]);
      lines[k].setAttribute("y2", q[1]);

      ticks[2 * k].setAttribute("cx", q[0]);
      ticks[2 * k].setAttribute("cy", q[1]);
      ticks[2 * k + 1].setAttribute("x", Math.min(q[0] + 7, 760));
      ticks[2 * k + 1].setAttribute("y", q[1] - 7);
    }

    center.setAttribute("cx", ox - a * e);
    center.setAttribute("cy", oy);
    centerText.setAttribute("x", ox - a * e);
    centerText.setAttribute("y", oy - 20);

    earth.setAttribute("cx", ox);
    earth.setAttribute("cy", oy);
    earthText.setAttribute("x", ox - 42);
    earthText.setAttribute("y", oy + 48);

    const pp = pos(0, e, ox);
    const aa = pos(Math.PI, e, ox);
    const s = pos(meanAnomaly, e, ox);

    per.setAttribute("cx", pp[0]);
    per.setAttribute("cy", pp[1]);
    perText.setAttribute("x", Math.min(pp[0] + 12, 700));
    perText.setAttribute("y", pp[1] - 12);

    apo.setAttribute("cx", aa[0]);
    apo.setAttribute("cy", aa[1]);
    apoText.setAttribute("x", Math.max(aa[0] - 120, 40));
    apoText.setAttribute("y", aa[1] - 12);

    sat.setAttribute("cx", s[0]);
    sat.setAttribute("cy", s[1]);

    evalText.textContent = e.toFixed(2);
    sval.textContent = `${parseFloat(speed.value).toFixed(1)}x`;

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