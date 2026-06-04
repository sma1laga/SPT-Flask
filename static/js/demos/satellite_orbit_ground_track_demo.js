(() => {
  const root = document.querySelector(".satcom-groundtrack");
  if (!root) return;

  const COASTLINES = [
    [[-168,65.5],[-156,71],[-140,70],[-128,71],[-110,73],[-100,73.5],[-90,76],[-80,75],[-70,73],[-65,67],[-60,60],[-55,53],[-58,47],[-65,45],[-67,44],[-70,41],[-73,39],[-75,36],[-78,34],[-80,30],[-81,25],[-83,25],[-85,30],[-90,30],[-94,29],[-97,26],[-100,25],[-105,22],[-108,24],[-112,27],[-115,30],[-117,32],[-120,35],[-123,38],[-124,42],[-124,48],[-128,53],[-135,58],[-145,60],[-156,60],[-165,63]],
    [[-50,60],[-43,60],[-37,64],[-30,68],[-22,73],[-19,78],[-25,82],[-38,83],[-50,81],[-58,76],[-55,70],[-52,64]],
    [[-79,11],[-71,12],[-63,11],[-55,6],[-50,1],[-44,-2],[-38,-8],[-35,-13],[-37,-20],[-43,-23],[-48,-27],[-55,-33],[-62,-39],[-68,-45],[-72,-50],[-71,-54],[-68,-54],[-66,-50],[-71,-42],[-72,-36],[-71,-28],[-71,-20],[-77,-14],[-79,-6],[-81,0],[-79,7]],
    [[-9,36],[-6,36],[-2,36],[3,39],[8,41],[12,42],[15,41],[18,41],[12,45],[5,47],[3,50],[1,51],[-2,49],[-5,48],[-9,43]],
    [[5,58],[12,58],[15,60],[18,62],[22,65],[25,68],[28,70],[24,71],[18,70],[12,66],[8,63],[6,60]],
    [[-6,50],[-3,51],[1,51],[1,53],[-1,55],[-3,57],[-7,58],[-8,55],[-6,52]],
    [[-24,64],[-14,64],[-13,66],[-19,67],[-24,66]],
    [[-17,14],[-16,21],[-12,28],[-7,32],[-2,36],[5,36],[10,35],[15,32],[22,31],[28,31],[32,30],[34,28],[37,22],[40,16],[43,12],[48,12],[51,11],[51,5],[45,-2],[42,-8],[40,-16],[35,-22],[32,-28],[25,-33],[18,-34],[15,-32],[13,-25],[12,-16],[10,-8],[8,-2],[3,4],[-3,6],[-9,8],[-13,11]],
    [[43,-12],[47,-16],[50,-22],[49,-25],[46,-25],[44,-20],[43,-16]],
    [[28,60],[40,64],[45,60],[50,56],[55,58],[60,60],[65,65],[70,68],[75,72],[85,75],[95,76],[105,76],[115,75],[130,73],[140,73],[150,70],[160,70],[170,68],[180,67],[180,60],[170,60],[160,58],[150,58],[135,50],[130,45],[125,50],[120,50],[115,50],[105,51],[95,50],[85,50],[75,50],[65,50],[55,52],[45,52],[40,50],[35,52],[30,56]],
    [[28,32],[35,33],[40,35],[45,38],[48,38],[52,35],[55,28],[52,22],[48,18],[43,14],[40,18],[37,22],[34,26],[30,30]],
    [[62,25],[68,24],[72,25],[75,23],[77,18],[78,13],[78,8],[82,6],[88,12],[90,22],[92,25],[88,26],[78,32],[72,32],[68,32]],
    [[95,5],[100,8],[104,5],[105,0],[108,-3],[115,-5],[120,-2],[125,-1],[132,-2],[140,-4],[140,1],[135,2],[128,3],[120,5],[113,4],[105,5],[100,5]],
    [[130,32],[134,35],[140,36],[143,41],[145,45],[141,43],[136,38],[132,34]],
    [[105,22],[110,22],[118,22],[122,30],[122,38],[118,40],[115,38],[108,35],[105,30]],
    [[114,-22],[115,-32],[118,-34],[123,-34],[130,-33],[134,-32],[138,-36],[141,-38],[146,-38],[149,-37],[152,-32],[153,-28],[148,-22],[145,-17],[141,-14],[135,-13],[132,-12],[128,-14],[124,-16],[120,-18],[117,-20]],
    [[172,-34],[176,-37],[178,-38],[175,-41],[171,-39]],
    [[167,-41],[172,-41],[174,-45],[171,-47],[167,-46],[166,-43]],
    [[130,-1],[140,-2],[146,-2],[150,-6],[148,-10],[140,-10],[135,-8],[130,-5]],
    [[-180,-71],[-170,-75],[-150,-78],[-130,-76],[-110,-74],[-90,-73],[-65,-76],[-45,-77],[-30,-75],[-10,-71],[15,-69],[40,-68],[65,-67],[90,-67],[115,-66],[140,-67],[160,-73],[175,-77],[180,-78],[180,-90],[-180,-90]],
    [[-92,16],[-87,13],[-83,9],[-79,8],[-78,12],[-83,15],[-88,18]],
  ];

  const SVG_NS = "http://www.w3.org/2000/svg";
  const byId = (id) => document.getElementById(`satcom-groundtrack-${id}`);
  const deg2rad = (d) => d * Math.PI / 180;
  const rad2deg = (r) => r * 180 / Math.PI;

  const landGroup = byId("land-group");
  const globeGroup = byId("globe-land");
  const anMarker = byId("an-marker");
  const satDot = byId("sat-dot");
  const satHalo = byId("sat-halo");
  const subsat = byId("subsat");
  const projLine = byId("proj-line");
  const groundTrack = byId("track");
  const mapSat = byId("map-sat");

  const inclSlider = byId("incl-slider");
  const periodSlider = byId("period-slider");
  const speedSlider = byId("speed-slider");
  const inclOut = byId("incl-out");
  const periodOut = byId("period-out");
  const speedOut = byId("speed-out");
  const playBtn = byId("play-btn");
  const resetBtn = byId("reset-btn");
  const presetBtn = byId("preset-btn");
  const rdTime = byId("rd-time");
  const rdLat = byId("rd-lat");
  const rdLon = byId("rd-lon");
  const rdOrb = byId("rd-orb");

  let inclinationDeg = 65;
  let periodMin = 358;
  let speed = 800;
  let simTime = 0;
  let playing = true;
  let lastReal = performance.now();
  const trackPts = [];
  const MAX_PTS = 1500;
  let lastGlobeUpdate = -100;

  const SIDEREAL = 86164;
  const RAAN_DEG = 233.33;
  const VIEW_TILT = 0.2;
  const R_ORBIT_SCREEN = 55;
  const R_EARTH_SCREEN = 32;

  COASTLINES.forEach((coords) => {
    const pts = coords.map(([lon, lat]) => {
      const x = (lon + 180) / 360 * 360;
      const y = 100 - (lat / 90) * 90;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    const poly = document.createElementNS(SVG_NS, "polygon");
    poly.setAttribute("points", pts);
    poly.setAttribute("fill", "#1f3a1f");
    poly.setAttribute("stroke", "#0a1f0a");
    poly.setAttribute("stroke-width", "0.2");
    poly.setAttribute("opacity", "0.85");
    landGroup.appendChild(poly);
  });

  function drawGlobe(rotationDeg) {
    globeGroup.innerHTML = "";
    const R = 32;
    COASTLINES.forEach((coords) => {
      const segments = [[]];
      let visible = false;
      coords.forEach(([lonGeo, lat]) => {
        const phi = deg2rad(lonGeo + rotationDeg);
        const theta = deg2rad(lat);
        const xi = Math.cos(phi) * Math.cos(theta);
        const yi = Math.sin(phi) * Math.cos(theta);
        const zi = Math.sin(theta);
        if (xi > 0) {
          segments[segments.length - 1].push(`${(yi * R).toFixed(1)},${(-zi * R).toFixed(1)}`);
          visible = true;
        } else if (visible) {
          segments.push([]);
          visible = false;
        }
      });

      segments.forEach((seg) => {
        if (seg.length < 3) return;
        const poly = document.createElementNS(SVG_NS, "polygon");
        poly.setAttribute("points", seg.join(" "));
        poly.setAttribute("fill", "#2c4a30");
        poly.setAttribute("stroke", "#0a1f0a");
        poly.setAttribute("stroke-width", "0.15");
        poly.setAttribute("opacity", "0.95");
        globeGroup.appendChild(poly);
      });
    });
  }

  function inertialFromOrbit(uRad, iRad, raanRad) {
    const xo = Math.cos(uRad);
    const yo = Math.sin(uRad);
    const x1 = xo;
    const y1 = yo * Math.cos(iRad);
    const z1 = yo * Math.sin(iRad);
    return {
      X: x1 * Math.cos(raanRad) - y1 * Math.sin(raanRad),
      Y: x1 * Math.sin(raanRad) + y1 * Math.cos(raanRad),
      Z: z1,
    };
  }

  function project(p, R) {
    return {
      sx: p.X * R,
      sy: -(p.Z * Math.cos(VIEW_TILT) + p.Y * Math.sin(VIEW_TILT)) * R,
    };
  }

  function rebuildOrbit() {
    const iRad = deg2rad(inclinationDeg);
    const raanRad = deg2rad(RAAN_DEG);
    const N = 96;
    let path = "";
    for (let k = 0; k <= N; k += 1) {
      const u = (k / N) * 2 * Math.PI;
      const s = project(inertialFromOrbit(u, iRad, raanRad), R_ORBIT_SCREEN);
      path += `${k === 0 ? "M" : "L"}${s.sx.toFixed(2)} ${s.sy.toFixed(2)} `;
    }

    let orbitCurve = byId("orbit-curve");
    if (orbitCurve.tagName.toLowerCase() === "ellipse") {
      const newPath = document.createElementNS(SVG_NS, "path");
      newPath.setAttribute("id", "satcom-groundtrack-orbit-curve");
      newPath.setAttribute("fill", "none");
      newPath.setAttribute("stroke", "#7F77DD");
      newPath.setAttribute("stroke-width", "0.7");
      orbitCurve.parentNode.replaceChild(newPath, orbitCurve);
      orbitCurve = newPath;
    }
    orbitCurve.setAttribute("d", path);

    const sAN = project(inertialFromOrbit(0, iRad, raanRad), R_ORBIT_SCREEN);
    anMarker.setAttribute("cx", sAN.sx.toFixed(2));
    anMarker.setAttribute("cy", sAN.sy.toFixed(2));
  }

  function renderGroundTrackPath() {
    const segments = [];
    let curSeg = [];
    trackPts.forEach((p) => {
      if (p.brk) {
        if (curSeg.length) segments.push(curSeg);
        curSeg = [];
      } else {
        const px = (p.lon + 180) / 360 * 360;
        const py = 100 - (p.lat / 90) * 90;
        curSeg.push(`${px.toFixed(1)} ${py.toFixed(1)}`);
      }
    });
    if (curSeg.length) segments.push(curSeg);
    groundTrack.setAttribute("d", segments.map((seg) => `M${seg.join(" L")}`).join(" "));
  }

  function update() {
    const iRad = deg2rad(inclinationDeg);
    const raanRad = deg2rad(RAAN_DEG);
    const periodSec = periodMin * 60;
    const u = ((simTime / periodSec) * 2 * Math.PI) % (2 * Math.PI);
    const pSat = inertialFromOrbit(u, iRad, raanRad);
    const sSat = project(pSat, R_ORBIT_SCREEN);
    satDot.setAttribute("cx", sSat.sx.toFixed(2));
    satDot.setAttribute("cy", sSat.sy.toFixed(2));
    satHalo.setAttribute("cx", sSat.sx.toFixed(2));
    satHalo.setAttribute("cy", sSat.sy.toFixed(2));

    const subProj = project(pSat, R_EARTH_SCREEN);
    subsat.setAttribute("cx", subProj.sx.toFixed(2));
    subsat.setAttribute("cy", subProj.sy.toFixed(2));
    projLine.setAttribute("x2", sSat.sx.toFixed(2));
    projLine.setAttribute("y2", sSat.sy.toFixed(2));

    const earthAngleDeg = (simTime / SIDEREAL) * 360;
    if (Math.abs(earthAngleDeg - lastGlobeUpdate) > 3) {
      drawGlobe(earthAngleDeg);
      lastGlobeUpdate = earthAngleDeg;
    }

    const betaS = rad2deg(Math.asin(pSat.Z));
    const lonInertial = rad2deg(Math.atan2(pSat.Y, pSat.X));
    let lambdaS = lonInertial - earthAngleDeg;
    lambdaS = ((lambdaS + 180) % 360 + 360) % 360 - 180;

    const prev = trackPts[trackPts.length - 1];
    if (!prev || prev.brk || Math.abs(lambdaS - prev.lon) < 100) {
      trackPts.push({ lon: lambdaS, lat: betaS });
    } else {
      trackPts.push({ brk: true }, { lon: lambdaS, lat: betaS });
    }
    while (trackPts.length > MAX_PTS) trackPts.shift();
    renderGroundTrackPath();

    mapSat.setAttribute("cx", ((lambdaS + 180) / 360 * 360).toFixed(1));
    mapSat.setAttribute("cy", (100 - (betaS / 90) * 90).toFixed(1));

    const hh = Math.floor(simTime / 3600);
    const mm = Math.floor((simTime % 3600) / 60);
    const ss = Math.floor(simTime % 60);
    rdTime.textContent = `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
    rdLat.textContent = `${betaS.toFixed(1)}°`;
    rdLon.textContent = `${lambdaS.toFixed(1)}°`;
    rdOrb.textContent = (simTime / periodSec).toFixed(2);
  }

  function tick(now) {
    const dt = (now - lastReal) / 1000;
    lastReal = now;
    if (playing) {
      simTime += dt * speed;
      update();
    }
    requestAnimationFrame(tick);
  }

  inclSlider.addEventListener("input", () => {
    inclinationDeg = Number(inclSlider.value);
    inclOut.textContent = `${inclinationDeg}°`;
    rebuildOrbit();
    trackPts.length = 0;
    renderGroundTrackPath();
  });

  periodSlider.addEventListener("input", () => {
    periodMin = Number(periodSlider.value);
    const h = Math.floor(periodMin / 60);
    const m = Math.round(periodMin % 60);
    periodOut.textContent = `${h}h ${String(m).padStart(2, "0")}m`;
    trackPts.length = 0;
    renderGroundTrackPath();
  });

  speedSlider.addEventListener("input", () => {
    speed = Number(speedSlider.value);
    speedOut.textContent = `${Math.round(speed)}×`;
  });

  playBtn.addEventListener("click", () => {
    playing = !playing;
    playBtn.textContent = playing ? "Pause" : "Play";
    if (playing) lastReal = performance.now();
  });

  resetBtn.addEventListener("click", () => {
    simTime = 0;
    trackPts.length = 0;
    update();
  });

  presetBtn.addEventListener("click", () => {
    inclSlider.value = 65;
    inclinationDeg = 65;
    inclOut.textContent = "65°";
    periodSlider.value = 358;
    periodMin = 358;
    periodOut.textContent = "5h 58m";
    speedSlider.value = 800;
    speed = 800;
    speedOut.textContent = "800×";
    simTime = 0;
    trackPts.length = 0;
    rebuildOrbit();
    update();
  });

  drawGlobe(0);
  rebuildOrbit();
  update();
  requestAnimationFrame(tick);
})();