(() => {
  const svg = document.getElementById("satcom3-svg");
  const alt = document.getElementById("satcom3-alt");
  const speed = document.getElementById("satcom3-speed");

  if (!svg || !alt || !speed) return;

  let running = true;
  let simMin = 0;
  let last = null;

  const RE = 6371;
  const MU = 398600.4418;
  const cx = 430;
  const cy = 315;

  function periodMin(a) {
    return (2 * Math.PI * Math.sqrt((a * a * a) / MU)) / 60;
  }

  function fmt(m) {
    return m < 180 ? `${m.toFixed(0)} min` : `${(m / 60).toFixed(1)} h`;
  }

  function el(name, attrs) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    svg.appendChild(node);
    return node;
  }

  function sats() {
    const customA = RE + parseFloat(alt.value);
    return [
      { name: "LEO", alt: 550, color: "#2563eb", rad: 120 },
      { name: "MEO", alt: 20200, color: "#7c3aed", rad: 205 },
      { name: "GEO", alt: 35786, color: "#16a34a", rad: 285 },
      {
        name: "Custom",
        alt: parseFloat(alt.value),
        color: "#f97316",
        rad: 85 + 200 * Math.sqrt(customA / (RE + 35786)),
        dash: true,
      },
    ].map((s) => ({ ...s, a: RE + s.alt, T: periodMin(RE + s.alt) }));
  }

  function init() {
    svg.innerHTML = "";
    el("text", { x: 28, y: 34, "font-size": 14, fill: "#4b5563" }).textContent =
      "Visual radii are compressed so the orbits fit on one screen.";

    for (const s of sats()) {
      if (s.name === "Custom") continue;
      el("circle", {
        cx,
        cy,
        r: s.rad,
        fill: "none",
        stroke: s.color,
        "stroke-width": 2.5,
        "stroke-dasharray": s.dash ? "7 5" : "",
      });
      const txt = el("text", {
        x: Math.min(cx + s.rad + 8, 840),
        y: cy + 4,
        "font-size": 13,
        fill: "#374151",
      });
      txt.textContent = s.name;
    }

    el("circle", { cx, cy, r: 42, fill: "#dbeafe", stroke: "#111827", "stroke-width": 2 });
    const earthTxt = el("text", {
      x: cx,
      y: cy + 5,
      "text-anchor": "middle",
      "font-size": 13,
      "font-weight": 700,
      fill: "#111827",
    });
    earthTxt.textContent = "Earth";
  }

  function draw(ts) {
    if (last === null) last = ts;
    const dt = (ts - last) / 1000;
    last = ts;
    if (running) simMin += dt * parseFloat(speed.value);

    init();
    const data = sats();
    const custom = data.find((d) => d.name === "Custom");

    el("circle", {
      cx,
      cy,
      r: custom.rad,
      fill: "none",
      stroke: custom.color,
      "stroke-width": 3,
      "stroke-dasharray": "7 5",
    });

    const customText = el("text", {
      x: Math.min(cx + custom.rad + 8, 840),
      y: cy - 12,
      "font-size": 13,
      fill: "#111827",
      "font-weight": 700,
    });
    customText.textContent = "Custom";

    for (const s of data) {
      const th = (2 * Math.PI * simMin) / s.T - Math.PI / 2;
      const x = cx + s.rad * Math.cos(th);
      const y = cy + s.rad * Math.sin(th);

      el("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: s.color, "stroke-width": 1, opacity: 0.25 });
      el("circle", {
        cx: x,
        cy: y,
        r: s.name === "Custom" ? 8 : 6,
        fill: s.color,
        stroke: "white",
        "stroke-width": 2,
      });
    }

    document.getElementById("satcom3-altval").textContent = `${parseFloat(alt.value).toLocaleString()} km`;
    document.getElementById("satcom3-speedval").textContent = `${speed.value} min/s`;
    document.getElementById("satcom3-periods").innerHTML = data
      .map(
        (d) => `<div class="satcom-row"><b style="color:${d.color}">${d.name}</b>: altitude ${Math.round(d.alt).toLocaleString()} km<br>Period: <b>${fmt(d.T)}</b></div>`
      )
      .join("");

    requestAnimationFrame(draw);
  }

  document.getElementById("satcom3-play").addEventListener("click", () => {
    running = !running;
    document.getElementById("satcom3-play").textContent = running ? "Pause" : "Play";
  });

  document.getElementById("satcom3-reset").addEventListener("click", () => {
    simMin = 0;
    running = true;
    document.getElementById("satcom3-play").textContent = "Pause";
  });

  requestAnimationFrame(draw);
})();