(function () {
  const controls = [
    { id: 'mass', label: 'Mass m [kg]', min: 0.2, max: 5, step: 0.1, value: 1.2 },
    { id: 'damping', label: 'Damping \u03bc [N·s/m]', min: 0.0, max: 6, step: 0.1, value: 0.8 },
    { id: 'stiffness', label: 'Stiffness k [N/m]', min: 0.5, max: 20, step: 0.5, value: 6 },
    { id: 'forceAmp', label: 'Force amplitude F₀ [N]', min: 0.1, max: 5, step: 0.1, value: 1.4 },
    { id: 'forceFreq', label: 'Force frequency \u03c9 [rad/s]', min: 0.2, max: 6, step: 0.1, value: 1.4 },
    { id: 'initDisp', label: 'Initial displacement x(0) [m]', min: -0.6, max: 0.6, step: 0.05, value: 0.05 },
    { id: 'initVel', label: 'Initial velocity \u0307x(0) [m/s]', min: -1.5, max: 1.5, step: 0.05, value: 0 }
  ];

  const state = Object.fromEntries(controls.map(c => [c.id, c.value]));

  const controlGrid = document.getElementById('control-grid');
  controls.forEach(ctrl => {
    const wrapper = document.createElement('label');
    wrapper.className = 'control';
    wrapper.innerHTML = `
      <span style="color:var(--subtext-color); font-size:0.95rem;">${ctrl.label}</span>
      <input type="range" id="${ctrl.id}" min="${ctrl.min}" max="${ctrl.max}" step="${ctrl.step}" value="${ctrl.value}" aria-label="${ctrl.label}">
      <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#6b7280;">
        <span>${ctrl.min}</span><span id="${ctrl.id}-value">${ctrl.value}</span>
      </div>`;
    controlGrid.appendChild(wrapper);
  });

  const odeEq = document.getElementById('ode-eq');
  const odeEqGeneric = document.getElementById('ode-eq-generic');
  const transientStat = document.getElementById('transient-stat');
  const laplaceEq = document.getElementById('laplace-eq');
  const transferBox = document.getElementById('transfer-box');
  const freqStat = document.getElementById('freq-stat');
  const dampingStat = document.getElementById('damping-stat');
  const polesStat = document.getElementById('poles-stat');
  const regimeStat = document.getElementById('regime-stat');
  const massBlock = document.getElementById('mass-block');

  // --- Mass–spring SVG animation (for intuition) ---
  const msSvg = document.getElementById('ms-svg');
  const msSpringPath = document.getElementById('ms-spring-path');
  const msMass = document.getElementById('ms-mass');
  const msMassText = document.getElementById('ms-mass-text');
  const msEq = document.getElementById('ms-eq');
  const msForce = document.getElementById('ms-force');
  const msDamp = document.getElementById('ms-damp');
  const msReadout = document.getElementById('ms-readout');

  // Animation UI controls
  const msPlayPauseBtn = document.getElementById('ms-playpause');
  const msResetBtn = document.getElementById('ms-reset');
  const msSpeed = document.getElementById('ms-speed');
  const msSpeedVal = document.getElementById('ms-speed-val');

  // Time-trace SVG elements (written as the mass moves)
  const msTracePath = document.getElementById('ms-trace-path');
  const msTraceDot = document.getElementById('ms-trace-dot');
  const msTraceCursor = document.getElementById('ms-trace-cursor');
  const msTraceReadout = document.getElementById('ms-trace-readout');

  // Driving field arrows (multiple)
  const msForceFieldLines = Array.from({ length: 18 }, (_, i) => document.getElementById(`ms-ff${i}`)).filter(Boolean);

  const msPole1 = document.getElementById('ms-pole1');
  const msPole2 = document.getElementById('ms-pole2');
  const msInp1 = document.getElementById('ms-inp1');
  const msInp2 = document.getElementById('ms-inp2');
  const msPhasor = document.getElementById('ms-phasor');

  const msAnim = { running: true, t: 0, lastTs: null, speed: 1.0 };
  let msCache = null;
  let msParams = null;
  let msMetrics = null;
  let msLoopStarted = false;

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function lerp(a, b, t) { return a + (b - a) * t; }
  function signNonZero(x) { return x >= 0 ? 1 : -1; }

  function buildSpringPath(x0, x1, y, coils = 9, amp = 14) {
    // Simple zigzag spring that stretches/compresses with x1.
    const lead = 16;
    const tail = 16;
    const start = x0;
    const end = x1;
    const innerStart = start + lead;
    const innerEnd = end - tail;
    const usable = Math.max(innerEnd - innerStart, 10);
    const steps = coils * 2;
    const dx = usable / steps;

    let d = `M ${start} ${y} L ${innerStart} ${y}`;
    for (let i = 0; i < steps; i++) {
      const xi = innerStart + (i + 1) * dx;
      const yi = y + (i % 2 === 0 ? -amp : amp);
      d += ` L ${xi.toFixed(2)} ${yi.toFixed(2)}`;
    }
    d += ` L ${innerEnd.toFixed(2)} ${y} L ${end} ${y}`;
    return d;
  }

  function rebuildMsAnimation(params, metrics) {
    if (!msSvg || !msSpringPath || !msMass) return;
    const oldParams = msParams;
    msParams = { ...params };
    msMetrics = metrics;

    // Higher-res cache for smoother animation
    msCache = simulateResponse(params, { dt: 0.006, tMax: chooseTmax(metrics) });

    const maxAbs = msCache.disp.reduce((acc, val) => Math.max(acc, Math.abs(val)), 1e-6);
    msCache.maxAbsDisp = maxAbs;

    // Build the written time-trace path once (then reveal it with dashoffset)
    if (msTracePath) {
      const x0 = 18, x1 = 706;
      const yMid = 48, yAmp = 20;
      const n = msCache.disp.length;

      // Downsample to keep the SVG path light
      const targetPts = 1600;
      const stride = Math.max(1, Math.floor(n / targetPts));

      let d = '';
      for (let i = 0; i < n; i += stride) {
        const t = msCache.times[i];
        const xx = x0 + (t / msCache.tMax) * (x1 - x0);
        const yy = yMid - (msCache.disp[i] / maxAbs) * yAmp;
        d += (d ? ' L ' : 'M ') + `${xx.toFixed(2)} ${yy.toFixed(2)}`;
      }
      msTracePath.setAttribute('d', d);

      // Prepare dash animation
      try {
        const len = msTracePath.getTotalLength();
        msCache.traceLen = len;
        msTracePath.setAttribute('stroke-dasharray', `${len.toFixed(2)}`);
        msTracePath.setAttribute('stroke-dashoffset', `${len.toFixed(2)}`);
      } catch (_) {
        msCache.traceLen = null;
      }
    }

    const keepTime = (typeof msAnim.t === "number") ? msAnim.t : 0;
    const resetBecauseIC = (!oldParams) || (params.initDisp !== oldParams.initDisp) || (params.initVel !== oldParams.initVel);
    msAnim.t = resetBecauseIC ? 0 : (keepTime % msCache.tMax);
    msAnim.lastTs = null;
    renderMsFrame(msAnim.t);
  }

  function renderMsFrame(tSec) {
    if (!msCache || !msParams || !msMetrics) return;

    const { dt, tMax, disp, vel, maxAbsDisp } = msCache;
    const n = disp.length;

    // Smooth interpolation between samples
    const idxFloat = (tSec / dt);
    const i0 = ((Math.floor(idxFloat) % n) + n) % n;
    const i1 = (i0 + 1) % n;
    const a = idxFloat - Math.floor(idxFloat);

    const x = lerp(disp[i0], disp[i1], a);
    const v = lerp(vel[i0], vel[i1], a);

    // --- Physical panel geometry ---
    const wallX = 30;     // spring anchor right of the wall
    const y0 = 122;       // track line
    const massW = 74;

    // Track limits inside the physical panel (see ms-physical in the SVG).
    // Center the equilibrium so the mass never "pins" to the right edge.
    const trackLeft = 18;
    const trackRight = 468;
    const edgeMargin = 10;

    const minCenter = trackLeft + massW / 2 + edgeMargin;
    const maxCenter = trackRight - massW / 2 - edgeMargin;
    const eqX = (minCenter + maxCenter) / 2; // center of travel

    // Scale meters -> pixels so the largest simulated displacement stays inside bounds.
    const maxAbs = Math.max(maxAbsDisp, 1e-6);
    const paddedMaxAbs = 1.04 * maxAbs;
    const halfRange = Math.min(maxCenter - eqX, eqX - minCenter);
    const dispScale = 0.96 * halfRange / paddedMaxAbs;

    const massCenterX = eqX + x * dispScale;
    const massLeft = massCenterX - massW / 2;

    msMass.setAttribute('x', massLeft.toFixed(2));
    msMassText.setAttribute('x', massCenterX.toFixed(2));
    if (msEq) { msEq.setAttribute('x1', eqX.toFixed(2)); msEq.setAttribute('x2', eqX.toFixed(2)); }

    msSpringPath.setAttribute('d', buildSpringPath(wallX, massLeft, y0));

    // Driving force: F(t) = F0 cos(ωt)
    const F = msParams.forceAmp * Math.cos(msParams.forceFreq * tSec);
    const signF = signNonZero(F);
    const magNorm = clamp(Math.abs(F) / Math.max(msParams.forceAmp, 1e-6), 0, 1);

    // Make the arrow always visible (length + opacity encodes magnitude)
    const forceLen = signF * clamp(22 + Math.abs(F) * 18, 22, 110);
    msForce.setAttribute('x1', massCenterX.toFixed(2));
    msForce.setAttribute('x2', (massCenterX + forceLen).toFixed(2));
    msForce.setAttribute('stroke-opacity', (0.25 + 0.75 * magNorm).toFixed(2));

    // Drive “field” arrows (direction + opacity)
    if (msForceFieldLines.length) {
      const smallLen = signF * (10 + 26 * magNorm);
      const op = (0.15 + 0.75 * magNorm).toFixed(2);
      for (const ln of msForceFieldLines) {
        const x1 = parseFloat(ln.getAttribute('x1'));
        ln.setAttribute('x2', (x1 + smallLen).toFixed(2));
        ln.setAttribute('stroke-opacity', op);
      }
    }

    // Damping arrow: -μ xdot(t)
    const D = -msParams.damping * v;
    const dampLen = clamp(D * 18, -90, 90);
    msDamp.setAttribute('x1', massCenterX.toFixed(2));
    msDamp.setAttribute('x2', (massCenterX + dampLen).toFixed(2));
    msDamp.setAttribute('stroke-opacity', (0.25 + 0.75 * clamp(Math.abs(v) / 1.5, 0, 1)).toFixed(2));

    if (msReadout) {
      msReadout.textContent = `x(t)=${x.toFixed(3)} m · v(t)=${v.toFixed(3)} m/s · F(t)=${F.toFixed(2)} N`;
    }

    // --- Complex plane panel (left) ---
    const centerX = 110;
    const centerY = 122;

    const omegaIn = msParams.forceFreq;
    const sigma = msMetrics.realPart;
    const omegaD = msMetrics.imagPart;

    const maxS = Math.max(Math.abs(sigma), Math.abs(omegaD), Math.abs(omegaIn), 1);
    const sScale = 70 / maxS;

    // system poles (gold)
    const p1 = msMetrics.poles[0];
    const p2 = msMetrics.poles[1];
    const poleToXY = (p) => ({
      cx: centerX + p.real * sScale,
      cy: centerY - p.imag * sScale
    });

    if (msPole1 && msPole2) {
      const a1 = poleToXY(p1);
      const b1 = poleToXY(p2);
      msPole1.setAttribute('cx', a1.cx.toFixed(2));
      msPole1.setAttribute('cy', a1.cy.toFixed(2));
      msPole2.setAttribute('cx', b1.cx.toFixed(2));
      msPole2.setAttribute('cy', b1.cy.toFixed(2));
    }

    // input markers jω (pink)
    if (msInp1 && msInp2) {
      msInp1.setAttribute('cx', centerX);
      msInp1.setAttribute('cy', (centerY - omegaIn * sScale).toFixed(2));
      msInp2.setAttribute('cx', centerX);
      msInp2.setAttribute('cy', (centerY + omegaIn * sScale).toFixed(2));
    }

    // decaying rotating phasor corresponding to the dominant mode (cyan)
    if (msPhasor) {
      const mag = Math.exp(sigma * tSec); // sigma is negative for stable systems
      const ang = omegaD * tSec;

      const phMax = 60;
      const phLen = phMax * clamp(mag, 0, 1.2);

      const x2 = centerX + phLen * Math.cos(ang);
      const y2 = centerY - phLen * Math.sin(ang);

      msPhasor.setAttribute('x1', centerX);
      msPhasor.setAttribute('y1', centerY);
      msPhasor.setAttribute('x2', x2.toFixed(2));
      msPhasor.setAttribute('y2', y2.toFixed(2));
    }

    // --- Time-trace reveal (bottom) ---
    if (msTracePath && msCache.traceLen) {
      const frac = (tSec % tMax) / tMax;
      const off = msCache.traceLen * (1 - frac);
      msTracePath.setAttribute('stroke-dashoffset', off.toFixed(2));

      const x0 = 18, x1 = 706;
      const yMid = 48, yAmp = 20;
      const tx = x0 + frac * (x1 - x0);
      const ty = yMid - (x / maxAbsDisp) * yAmp;

      if (msTraceCursor) {
        msTraceCursor.setAttribute('x1', tx.toFixed(2));
        msTraceCursor.setAttribute('x2', tx.toFixed(2));
      }
      if (msTraceDot) {
        msTraceDot.setAttribute('cx', tx.toFixed(2));
        msTraceDot.setAttribute('cy', ty.toFixed(2));
      }
      if (msTraceReadout) {
        msTraceReadout.textContent = `t=${tSec.toFixed(2)} s · x(t)=${x.toFixed(3)} m`;
      }
    }
  }

  function msLoop(ts) {
    if (!msSvg) return;
    if (!msLoopStarted) return;

    if (!msCache) {
      window.requestAnimationFrame(msLoop);
      return;
    }

    if (msAnim.lastTs == null) msAnim.lastTs = ts;
    const dt = (ts - msAnim.lastTs) / 1000;
    msAnim.lastTs = ts;

    if (msAnim.running) {
      msAnim.t += dt * msAnim.speed;
      if (msAnim.t > msCache.tMax) msAnim.t -= msCache.tMax;
      renderMsFrame(msAnim.t);
    }

    window.requestAnimationFrame(msLoop);
  }


  function typeset(elements) {
    if (!window.MathJax) return;
    const targets = elements.filter(Boolean);
    if (!targets.length) return;
    const promise = window.MathJax.startup?.promise || Promise.resolve();
    promise
      .then(() => window.MathJax.typesetPromise(targets))
      .catch((err) => console.error('MathJax typeset error', err));
  }

  function rk4Step(x, v, t, dt, params) {
    const accel = (xPos, vPos, time) => (
      (-params.damping * vPos - params.stiffness * xPos + params.forceAmp * Math.cos(params.forceFreq * time)) / params.mass
    );

    const k1x = v;
    const k1v = accel(x, v, t);

    const k2x = v + 0.5 * dt * k1v;
    const k2v = accel(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t + 0.5 * dt);

    const k3x = v + 0.5 * dt * k2v;
    const k3v = accel(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t + 0.5 * dt);

    const k4x = v + dt * k3v;
    const k4v = accel(x + dt * k3x, v + dt * k3v, t + dt);

    const newX = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    const newV = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
    return [newX, newV];
  }

  function simulateResponse(params, opts = {}) {
    const dt = (typeof opts.dt === 'number') ? opts.dt : 0.02;
    const tMax = (typeof opts.tMax === 'number') ? opts.tMax : 20;

    let x = params.initDisp;
    let v = params.initVel;

    const times = [];
    const disp = [];
    const vel = [];
    const forcing = [];

    for (let t = 0; t <= tMax; t += dt) {
      times.push(t);
      disp.push(x);
      vel.push(v);

      forcing.push((params.forceAmp / params.stiffness) * Math.cos(params.forceFreq * t));

      [x, v] = rk4Step(x, v, t, dt, params);
    }

    return { dt, tMax, times, disp, vel, forcing };
  }

  function computePoles(params) {
    const { mass: m, damping: mu, stiffness: k } = params;
    const sqrtTerm = Math.sqrt(Math.max(mu * mu - 4 * m * k, 0));
    const realPart = -mu / (2 * m);
    const imagPart = Math.sqrt(Math.max(4 * m * k - mu * mu, 0)) / (2 * m);
    const poles = imagPart > 0
      ? [
          { real: realPart, imag: imagPart },
          { real: realPart, imag: -imagPart }
        ]
      : [
          { real: (-mu + sqrtTerm) / (2 * m), imag: 0 },
          { real: (-mu - sqrtTerm) / (2 * m), imag: 0 }
        ];
    const wn = Math.sqrt(k / m);
    const zeta = mu / (2 * Math.sqrt(m * k));
    return { wn, zeta, poles, realPart, imagPart, wd: imagPart };
  }

  function chooseTmax(metrics) {
    const sigma = metrics?.realPart ?? -0.5;     // stable systems: sigma < 0
    const a = Math.abs(sigma);

    // Rule of thumb: transient ~ e^{sigma t}. For 0.7% remaining use 5/|sigma|.
    let tTransient = (a < 1e-4) ? 40 : (5 / a);

    // Also show several oscillation cycles if underdamped
    const wd = Math.abs(metrics?.imagPart ?? 0);
    const period = (wd > 1e-3) ? (2 * Math.PI / wd) : 0;
    const tCycles = period ? (10 * period) : 0;

    // Clamp to a reasonable range
    return Math.max(14, Math.min(60, Math.max(tTransient, tCycles, 18)));
  }


  function formatPole(p) {
    const re = Math.round(p.real * 100) / 100;
    const im = Math.round(p.imag * 100) / 100;
    if (Math.abs(im) < 1e-6) return `${re}`;
    return `${re} ${im >= 0 ? '+' : '-'} ${Math.abs(im)}j`;
  }

function updateEquations(params) {
  const { mass: m, damping: mu, stiffness: k, forceAmp: F0, forceFreq: omega } = params;

  // String.raw prevents JS from interpreting \b, \t, \f, inside Tex
  odeEq.innerHTML = String.raw`$$ ${m.toFixed(2)}\,\ddot{x}(t)+${mu.toFixed(2)}\,\dot{x}(t)+${k.toFixed(2)}\,x(t)=${F0.toFixed(2)}\,\cos(${omega.toFixed(2)}\,t) $$`;
  laplaceEq.innerHTML = String.raw`$$ X(s)=\frac{${F0.toFixed(2)}\,s}{\bigl(s^2+(${omega.toFixed(2)})^{2}\bigr)\,\bigl(${m.toFixed(2)}\, s^2+${mu.toFixed(2)}\, s+${k.toFixed(2)}\bigr)} $$`;
  transferBox.innerHTML = String.raw`$$ H(s)=\frac{X(s)}{F(s)}=\frac{1}{${m.toFixed(2)}\, s^2+${mu.toFixed(2)}\, s+${k.toFixed(2)}} $$`;

  typeset([odeEq, laplaceEq, transferBox]);
}

  function updateStats(params, metrics) {
    freqStat.innerHTML = `\\(\\omega_n = ${metrics.wn.toFixed(2)}\\,\\text{rad/s}\\)`;
    dampingStat.innerHTML = `\\(\\zeta = ${metrics.zeta.toFixed(2)}\\)`;

    const [p1, p2] = metrics.poles;
    if (metrics.zeta < 1 && metrics.imagPart > 0) {
      // Compact (prevents MathJax overflow in the status grid)
      polesStat.innerHTML = `\\(s_{1,2} = ${metrics.realPart.toFixed(2)} \\pm ${metrics.imagPart.toFixed(2)}j\\)`;
    } else {
      polesStat.innerHTML = `\\(s_1=${formatPole(p1)},\\; s_2=${formatPole(p2)}\\)`;
    }

    // "Transient time" how long the natural-mode envelope needs to get small
    if (transientStat) {
      const a = Math.abs(metrics.realPart);
      transientStat.innerHTML = (a < 1e-6)
        ? `\\(T_{tr}=\\infty\\; (\\mu\\approx 0)\\)`
        : `\\(T_{tr}\\approx ${(5 / a).toFixed(1)}\\,\\text{s}\\)`;
    }

    let regime = 'Under-damped (ringing)';
    if (Math.abs(metrics.zeta - 1) < 1e-2) regime = 'Critically damped';
    else if (metrics.zeta > 1) regime = 'Over-damped';
    regimeStat.textContent = regime;

    massBlock.textContent = `${params.mass.toFixed(1)} kg`;
  }

  function plotTime(params, metrics) {
    const tMax = chooseTmax(metrics);
    const { times, disp, forcing } = simulateResponse(params, { tMax, dt: 0.02 });

    const layout = {
      margin: { l: 48, r: 48, b: 40, t: 10 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { color: '#0f172a' },
      xaxis: { title: 'time [s]', gridcolor: '#e5e7eb', zerolinecolor: '#94a3b8' },
      yaxis: { title: 'x(t) [m]', gridcolor: '#e5e7eb', zerolinecolor: '#94a3b8' },
      yaxis2: { title: 'F(t) [N]', overlaying: 'y', side: 'right', gridcolor: '#e5e7eb', zerolinecolor: '#94a3b8' },
      showlegend: false
    };

    const traces = [
      { x: times, y: disp, mode: 'lines', line: { color: '#22d3ee', width: 3 }, name: 'x(t)' },
      { x: times, y: forcing, mode: 'lines', yaxis: 'y2', line: { color: '#fb7185', width: 2, dash: 'dash' }, name: 'F(t)' }
    ];
    Plotly.react('time-plot', traces, layout, { responsive: true });
  }

  function plotPoles(metrics) {
    const { realPart: sigma, imagPart: omegaD } = metrics;
    const span = Math.max(Math.abs(omegaD), 1);
    const data = [
      {
        x: [sigma, sigma],
        y: [omegaD, -omegaD],
        mode: 'markers',
        marker: { color: '#f59e0b', size: 12, symbol: 'x' },
        hovertemplate: 'Re(s)=%{x:.2f}<br>Im(s)=%{y:.2f}'
      }
    ];
    const layout = {
      margin: { l: 40, r: 10, b: 40, t: 10 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { color: '#0f172a' },
      xaxis: { title: 'Re{s}', zeroline: true, zerolinecolor: '#94a3b8', gridcolor: '#e5e7eb' },
      yaxis: { title: 'Im{s}', zeroline: true, zerolinecolor: '#94a3b8', gridcolor: '#e5e7eb' },
      shapes: [
        { type: 'line', x0: 0, x1: 0, y0: -span * 1.4, y1: span * 1.4, line: { color: '#cbd5e1', dash: 'dot' } },
        { type: 'line', x0: sigma * 1.4, x1: sigma * 1.4, y0: -span, y1: span, line: { color: '#cbd5e1', dash: 'dot' } }
      ]
    };
    Plotly.react('pole-plot', data, layout, { responsive: true });
  }

  function update() {
    const params = { ...state };
    const metrics = computePoles(params);
    updateEquations(params);
    updateStats(params, metrics);
    plotTime(params, metrics);
    plotPoles(metrics);
    rebuildMsAnimation(params, metrics);
    if (!msLoopStarted && msSvg) {
      msLoopStarted = true;
      window.requestAnimationFrame(msLoop);
    }
    typeset([freqStat, dampingStat, polesStat, transientStat, odeEqGeneric]);
  }

  let pending = false;
  function requestUpdate() {
    if (pending) return;
    pending = true;
    window.requestAnimationFrame(() => {
      pending = false;
      update();
    });
  }

  controls.forEach(ctrl => {
    const input = document.getElementById(ctrl.id);
    const valueLabel = document.getElementById(`${ctrl.id}-value`);
    input.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      state[ctrl.id] = val;
      valueLabel.textContent = val;
      requestUpdate();
    });
  });

  // --- Animation controls (play/pause/reset/speed) ---
  function syncPlayPauseUI() {
    if (!msPlayPauseBtn) return;
    msPlayPauseBtn.textContent = msAnim.running ? 'Pause' : 'Play';
    msPlayPauseBtn.setAttribute('aria-pressed', msAnim.running ? 'true' : 'false');
  }

  if (msPlayPauseBtn) {
    msPlayPauseBtn.addEventListener('click', () => {
      msAnim.running = !msAnim.running;
      syncPlayPauseUI();
    });
  }

  if (msResetBtn) {
    msResetBtn.addEventListener('click', () => {
      msAnim.t = 0;
      msAnim.lastTs = null;
      renderMsFrame(0);
      // If paused, keep it paused
    });
  }

  if (msSpeed) {
    msSpeed.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      msAnim.speed = clamp(v, 0.25, 2.5);
      if (msSpeedVal) msSpeedVal.textContent = `${msAnim.speed.toFixed(2)}×`;
    });
    // init label
    if (msSpeedVal) msSpeedVal.textContent = `${msAnim.speed.toFixed(2)}×`;
  }

  syncPlayPauseUI();

  update();
  if (window.MathJax?.startup?.promise) {
    window.MathJax.startup.promise.then(() => {
      typeset([odeEq, laplaceEq, transferBox, freqStat, dampingStat, polesStat]);
    });
  }
})();