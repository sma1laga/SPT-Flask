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

  const el = (name, attrs = {}) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    svg.appendChild(node);
    return node;
  };

  for (let i = 0; i < 12; i += 1) {
    sectors.push(el("path", { fill: i % 2 ? "#f8fafc" : "#dbeafe", stroke: "#bfdbfe", "stroke-width": 1, opacity: 0.75 }));
  }

  const orbit = el("path", { fill: "none", stroke: "#111827", "stroke-width": 3 });
  const center = el("circle", { r: 3, fill: "#94a3b8" });
  const centerText = el("text", { "font-size": 12, fill: "#6b7280" });
  centerText.textContent = "ellipse center";
  const earth = el("circle", { r: 22, fill: "#e0f2fe", stroke: "#111827", "stroke-width": 2 });
  const earthText = el("text", { "font-size": 14, fill: "#111827", "font-weight": "700" });
  earthText.textContent = "Earth at focus";
  const per = el("circle", { r: 6, fill: "#16a34a" });
  const apo = el("circle", { r: 6, fill: "#7c3aed" });
  const perText = el("text", { "font-size": 14, fill: "#111827", "font-weight": "700" });
  const apoText = el("text", { "font-size": 14, fill: "#111827", "font-weight": "700" });
  perText.textContent = "Perigee: fastest";
  apoText.textContent = "Apogee: slowest";

  for (let i = 0; i < 12; i += 1) {
    lines.push(el("line", { stroke: "#94a3b8", "stroke-width": 1, opacity: 0.65 }));
    ticks.push(el("circle", { r: 4, fill: "#475569" }));
    const t = el("text", { "font-size": 11, fill: "#6b7280" });
    t.textContent = `t${i}`;
    ticks.push(t);
  }

  const sat = el("circle", { r: 8, fill: "#f97316", stroke: "white", "stroke-width": 3 });

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
    centerText.setAttribute("x", ox - a * e - 42);
    centerText.setAttribute("y", oy - 14);

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