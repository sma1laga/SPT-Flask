(() => {
  const svg = document.getElementById("satcom-svg");
  const lat = document.getElementById("satcom-lat");
  if (!svg || !lat) return;

  const RE = 6371;
  const RG = 42164;
  const psiMax = Math.acos(RE / RG) * 180 / Math.PI;
  const cx = 390;
  const cy = 300;
  const R = 145;
  const satR = 405;

  function el(n, a) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", n);
    Object.keys(a).forEach((k) => e.setAttribute(k, a[k]));
    svg.appendChild(e);
    return e;
  }

  function draw() {
    const latitude = Number.parseFloat(lat.value);
    const phi = latitude * Math.PI / 180;
    const user = { x: cx + R * Math.cos(phi), y: cy - R * Math.sin(phi) };
    const sat = { x: cx + satR, y: cy };
    const sub = { x: cx + R, y: cy };
    const visible = Math.abs(latitude) <= psiMax;

    const ux = Math.cos(phi);
    const uy = -Math.sin(phi);
    const tx = Math.sin(phi);
    const ty = Math.cos(phi);
    const losx = sat.x - user.x;
    const losy = sat.y - user.y;
    const len = Math.hypot(losx, losy);
    const lx = losx / len;
    const ly = losy / len;
    const up = lx * ux + ly * uy;
    const tangent = Math.abs(lx * tx + ly * ty);
    const elev = Math.atan2(up, tangent) * 180 / Math.PI;

    const nlim = { x: cx + R * Math.cos(psiMax * Math.PI / 180), y: cy - R * Math.sin(psiMax * Math.PI / 180) };
    const slim = { x: cx + R * Math.cos(-psiMax * Math.PI / 180), y: cy - R * Math.sin(-psiMax * Math.PI / 180) };

    svg.innerHTML = "";
    el("text", { x: 28, y: 35, "font-size": 14, fill: "#4b5563" }).textContent = "Cross-section through Earth and the GEO satellite. GEO is fixed above the equator.";
    el("line", { x1: cx, y1: cy, x2: sat.x, y2: sat.y, stroke: "#64748b", "stroke-width": 1.7, "stroke-dasharray": "5 5" });
    el("text", { x: (cx + sat.x) / 2 + 8, y: cy - 12, "font-size": 13, fill: "#4b5563" }).textContent = "r = 42,164 km";
    el("circle", { cx: sat.x, cy: sat.y, r: 8, fill: "#2563eb" });
    el("text", { x: Math.min(sat.x + 14, 770), y: sat.y + 4, "font-size": 13, fill: "#1d4ed8", "font-weight": 700 }).textContent = "GEO satellite";
    el("circle", { cx, cy, r: R, fill: "#f8fafc", stroke: "#111827", "stroke-width": 2 });
    el("circle", { cx, cy, r: 4, fill: "#111827" });
    el("text", { x: cx, y: cy + 6, "text-anchor": "middle", "font-size": 13, "font-weight": 700 }).textContent = "Earth";
    el("line", { x1: cx - R - 18, y1: cy, x2: cx + R + 18, y2: cy, stroke: "#cbd5e1", "stroke-width": 1.5 });
    el("text", { x: cx - R - 78, y: cy + 4, "font-size": 13, fill: "#6b7280" }).textContent = "equator";
    el("circle", { cx: sub.x, cy: sub.y, r: 5, fill: "#2563eb" });

    const pts = [];
    for (let i = -psiMax; i <= psiMax; i += 1) {
      const a = i * Math.PI / 180;
      pts.push(`${cx + R * Math.cos(a)},${cy - R * Math.sin(a)}`);
    }
    el("polyline", { points: pts.join(" "), fill: "none", stroke: "#16a34a", "stroke-width": 5, opacity: 0.45 });
    el("circle", { cx: nlim.x, cy: nlim.y, r: 5, fill: "#16a34a" });
    el("circle", { cx: slim.x, cy: slim.y, r: 5, fill: "#16a34a" });
    el("text", { x: nlim.x - 125, y: nlim.y - 12, "font-size": 13, fill: "#15803d" }).textContent = "visibility limit";
    el("text", { x: slim.x - 132, y: slim.y + 24, "font-size": 13, fill: "#15803d" }).textContent = "visibility limit";
    el("circle", { cx: user.x, cy: user.y, r: 7, fill: visible ? "#16a34a" : "#dc2626" });
    el("text", { x: Math.min(user.x + 12, 690), y: user.y - 10, "font-size": 13, fill: visible ? "#15803d" : "#b91c1c", "font-weight": 700 }).textContent = `ground station φ = ${latitude.toFixed(1)}°`;
    el("line", { x1: user.x, y1: user.y, x2: sat.x, y2: sat.y, stroke: visible ? "#16a34a" : "#dc2626", "stroke-width": 2.3 });
    el("text", { x: Math.min((user.x + sat.x) / 2 + 12, 700), y: (user.y + sat.y) / 2 - 8, "font-size": 13, fill: visible ? "#15803d" : "#b91c1c" }).textContent = "line of sight";
    el("line", { x1: user.x - 62 * tx, y1: user.y - 62 * ty, x2: user.x + 62 * tx, y2: user.y + 62 * ty, stroke: "#111827", "stroke-width": 1.8, "stroke-dasharray": "4 4" });
    el("text", { x: user.x - 70 * tx + 5, y: user.y - 70 * ty - 8, "font-size": 13, fill: "#374151" }).textContent = "local horizon";
    el("line", { x1: cx, y1: cy, x2: user.x, y2: user.y, stroke: "#94a3b8", "stroke-width": 1.5 });
    el("text", { x: (cx + user.x) / 2 - 22, y: (cy + user.y) / 2 - 8, "font-size": 13, fill: "#4b5563" }).textContent = "ψ";
    el("rect", { x: 55, y: 470, width: 350, height: 115, rx: 14, fill: "white", stroke: "#94a3b8" });
    el("text", { x: 75, y: 498, "font-size": 13, fill: "#111827", "font-weight": 700 }).textContent = "Theoretical visibility limit";
    el("text", { x: 75, y: 528, "font-size": 13, fill: "#374151" }).textContent = "cos ψmax = RE / rGEO";
    el("text", { x: 75, y: 558, "font-size": 13, fill: "#374151" }).textContent = `ψmax = cos⁻¹(6371 / 42164) ≈ ${psiMax.toFixed(1)}°`;

    document.getElementById("satcom-latval").textContent = `${latitude.toFixed(1)}°`;
    const visibleEl = document.getElementById("satcom-visible");
    visibleEl.className = `satcom-result ${visible ? "satcom-good" : "satcom-bad"}`;
    visibleEl.textContent = visible ? "Satellite is theoretically visible" : "Satellite is below the horizon";
    document.getElementById("satcom-elev").textContent = `${elev.toFixed(1)}°`;
  }

  lat.addEventListener("input", draw);
  document.querySelectorAll("button[data-lat]").forEach((b) => {
    b.addEventListener("click", () => {
      lat.value = b.dataset.lat;
      draw();
    });
  });

  draw();
})();