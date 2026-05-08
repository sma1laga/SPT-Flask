(() => {
  const orbit = document.getElementById("satcom-orbit");
  if (!orbit) return;

  const major = document.getElementById("satcom-major");
  const center = document.getElementById("satcom-center");
  const earth = document.getElementById("satcom-earth");
  const focus2 = document.getElementById("satcom-focus2");
  const sat = document.getElementById("satcom-sat");
  const rline = document.getElementById("satcom-rline");

  const centerText = document.getElementById("satcom-center-text");
  const earthText = document.getElementById("satcom-earth-text");
  const focus2Text = document.getElementById("satcom-focus2-text");
  const rtext = document.getElementById("satcom-rtext");

  const ecc = document.getElementById("satcom-ecc");
  const evalText = document.getElementById("satcom-eval");
  const speed = document.getElementById("satcom-speed");
  const sval = document.getElementById("satcom-sval");
  const playButton = document.getElementById("satcom-play");
  const resetButton = document.getElementById("satcom-reset");

  let running = true;
  let meanAnomaly = 0;
  let last = null;

  const cx = 390;
  const cy = 250;
  const a = 230;

  const solveKepler = (M, e) => {
    let E = M;
    for (let i = 0; i < 8; i += 1) {
      E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    }
    return E;
  };

  const draw = (ts) => {
    if (last === null) last = ts;
    const dt = (ts - last) / 1000;
    last = ts;

    const e = parseFloat(ecc.value);
    const b = a * Math.sqrt(1 - e * e);
    const focusX = cx + a * e;

    if (running) meanAnomaly = (meanAnomaly + dt * parseFloat(speed.value) * 0.55) % (2 * Math.PI);

    let path = "";
    for (let i = 0; i <= 360; i += 1) {
      const th = i * Math.PI / 180;
      const x = cx + a * Math.cos(th);
      const y = cy + b * Math.sin(th);
      path += `${i ? " L " : "M "}${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    path += " Z";

    orbit.setAttribute("d", path);
    major.setAttribute("x1", cx - a);
    major.setAttribute("y1", cy);
    major.setAttribute("x2", cx + a);
    major.setAttribute("y2", cy);

    center.setAttribute("cx", cx);
    center.setAttribute("cy", cy);
    centerText.setAttribute("x", cx - 44);
    centerText.setAttribute("y", cy + 25);

    earth.setAttribute("cx", focusX);
    earth.setAttribute("cy", cy);
    earthText.setAttribute("x", Math.min(focusX - 48, 650));
    earthText.setAttribute("y", cy + 48);

    focus2.setAttribute("cx", cx - a * e);
    focus2.setAttribute("cy", cy);
    focus2Text.setAttribute("x", cx - a * e - 36);
    focus2Text.setAttribute("y", cy - 15);

    const E = solveKepler(meanAnomaly, e);
    const sx = cx + a * Math.cos(E);
    const sy = cy + b * Math.sin(E);
    sat.setAttribute("cx", sx);
    sat.setAttribute("cy", sy);

    rline.setAttribute("x1", focusX);
    rline.setAttribute("y1", cy);
    rline.setAttribute("x2", sx);
    rline.setAttribute("y2", sy);

    rtext.setAttribute("x", Math.min((focusX + sx) / 2 + 8, 700));
    rtext.setAttribute("y", (cy + sy) / 2 - 8);

    evalText.textContent = e.toFixed(2);
    sval.textContent = `${parseFloat(speed.value).toFixed(1)}x`;

    window.requestAnimationFrame(draw);
  };

  playButton.addEventListener("click", () => {
    running = !running;
    playButton.textContent = running ? "Pause" : "Play";
  });

  resetButton.addEventListener("click", () => {
    meanAnomaly = 0;
    running = true;
    playButton.textContent = "Pause";
  });

  window.requestAnimationFrame(draw);
})();