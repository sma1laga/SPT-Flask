/* =========================================================================
   Block-Diagram UI  –  v0.3  (editable TF blocks with KaTeX overlay)
   ========================================================================= */

const canvas = document.getElementById("diagramCanvas");
const ctx    = canvas.getContext("2d");
let arrowCol = "#000";   // current colour for connections
function getArrowColor(){
  const val = getComputedStyle(canvas).getPropertyValue('--arrow-col').trim();
  if (val) return val;
  return document.body.classList.contains('dark-mode') ? '#ddd' : '#000';
}
/* --- modal DOM shortcuts (must exist before we call openEditModal) --- */
const srcSelect = document.getElementById("srcSelect");
const srcCustom = document.getElementById("srcCustom");
const srcNum    = document.getElementById("srcNum");
const srcDen    = document.getElementById("srcDen");
const btnSimulate = document.getElementById("btnSimulate");
const simCanvas  = document.getElementById("simCanvas");


/* -----------  side selection & geometry helpers  ------------------- */
const DIRS = {E:[1,0], W:[-1,0], N:[0,-1], S:[0,1]};

/* first free face in a fixed priority order */
function chooseSide(node, isOut){
  const used = edges
     .filter(e => (isOut ? e.from===node.id : e.to===node.id))
     .map (e => isOut ? e.outSide : e.inSide);
  const pref = isOut ? ["E","S","N","W"] : ["W","N","S","E"];
  return pref.find(p => !used.includes(p)) || pref[0];
}

function portPos(node, side){
  switch(side){
    case "E": return {x: node.x + node.w,     y: node.y + node.h/2};
    case "W": return {x: node.x,              y: node.y + node.h/2};
    case "N": return {x: node.x + node.w/2,   y: node.y};
    case "S": return {x: node.x + node.w/2,   y: node.y + node.h};
  }
}


/* ---------------- internal graph model -------------------------------- */
let nodes = [];
let edges = [];
let nextId = 1;

let selectedNode = null, selectedEdge = null;

/* palette block definitions */
const BLOCK_LIBRARY = [
  { name: 'Source',      type: 'Input',      label: 'X(s)',  kx: 'X(s)' },
  { name: 'Gain',        type: 'Gain',       label: 'K',     kx: 'K' },
  { name: 'TF',          type: 'TF',         label: 'H(s)',  kx: '\\dfrac{N(s)}{D(s)}' },
  { name: 'Zero-Pole',   type: 'ZeroPole',   label: 'Z/P',   kx: 'Z/P' },
  { name: 'Integrator',  type: 'Integrator', label: '1/s',   kx: '\\dfrac{1}{s}' },
  { name: 'Derivative',  type: 'Derivative', label: 's',     kx: 's' },
  { name: 'Saturation',  type: 'Saturation', label: 'Sat',   kx: 'Sat' },
  { name: 'Delay',       type: 'Delay',      label: 'Delay', kx: 'Delay' },
  { name: 'Scope',       type: 'Scope',      label: 'Scope', kx: 'Scope' },
  { name: 'Sum',         type: 'Adder',      label: 'Σ',     kx: '\\Sigma' },
  { name: 'Mux',         type: 'Mux',        label: 'Mux',   kx: 'Mux' },
  { name: 'Demux',       type: 'Demux',      label: 'Demux', kx: 'Demux' },
  { name: 'PID',         type: 'PID',        label: 'PID',   kx: 'PID' }
];

let ghostBlock = null;

function initPalette(){
  const list = document.getElementById('blockList');
  BLOCK_LIBRARY.forEach(b => {
    const li = document.createElement('li');
    li.dataset.type  = b.type;
    li.dataset.label = b.label;
    li.dataset.name  = b.name.toLowerCase();
    li.innerHTML = `<span>${b.name}</span><span class="kx">${b.kx}</span>`;
    li.addEventListener('click', () => {
      ghostBlock = { ...b, x:0, y:0, w:90, h:45 };
    });
    list.appendChild(li);
  });

  document.getElementById('blockSearch').addEventListener('input', e => {
    const term = e.target.value.toLowerCase();
    list.querySelectorAll('li').forEach(li => {
      li.style.display = li.dataset.name.includes(term) ? 'flex' : 'none';
    });
  });

  list.querySelectorAll('.kx').forEach(el => katex.render(el.textContent, el));
}


/* ---------------- palette helpers ------------------------------------ */
function addNode(type, label, x = 120, y = 80) {
  const node = {
    id: nextId++,
    type,
    label,
    x, y,
    w: 90,
    h: 45,
    params: {},            // for TF blocks: {num:'', den:''}
    latexEl: null          // DOM <div> where KaTeX renders
  };
  if(type === "Mux")   node.params.inputs  = 2;
  if(type === "Demux") node.params.outputs = 2;
  nodes.push(node);
  drawAll();
}

/* default source & sink ------------------------------------------------ */
(() => {
  addNode("Input",  "X(s)", 40, canvas.height/2 - 60);
  addNode("Output", "Y(s)", canvas.width - 120, canvas.height/2 - 60);
})();

/* ---------------- connection mode & drag ------------------------------ */
let connectMode = false, connectFrom = null;
let dragNode = null, dragOffset = {x:0,y:0};

canvas.addEventListener("mousedown", ev => {
  const p = mouse(ev);

  if (ghostBlock) {
    addNode(ghostBlock.type, ghostBlock.label, p.x - ghostBlock.w/2, p.y - ghostBlock.h/2);
    ghostBlock = null;
    drawAll();
    return;
  }

  const n = nodeAt(p.x, p.y);
  /* connect mode */
  if (connectMode) {
    if (!n) return;
    if (!connectFrom) { connectFrom = n; return; }
    if (n !== connectFrom) 
        edges.push({
        from: connectFrom.id,
        to:   n.id,
        sign: "+",

        });
    connectFrom = null;
    drawAll(); return;
  }

  /* drag, select or edit */
  if (n) {                                  // clicked a block
    if (ev.detail === 2 &&                 // double-click → open editor
        (n.type === "TF" || n.type === "Gain" || n.type === "Input" || n.type === "PID" || n.type === "Mux" || n.type === "Demux")) {

      openEditModal(n);
    } else {                               // single-click → select block
      selectedNode = n; selectedEdge = null;
      dragNode = n;                     // start drag on mouse-move
      dragOffset = { x: p.x - n.x, y: p.y - n.y };
      canvas.style.cursor = "grabbing";
    }
  } else {                                 // maybe we hit a wire
    const eHit = edgeAt(p.x, p.y);
    selectedEdge = eHit; selectedNode = null;
  }
  drawAll();

});

canvas.addEventListener("dblclick", ev=>{
  const {x, y} = mouse(ev);
  const hit = edges.find(e=>{
    const a = nodes.find(n=>n.id===e.from);
    const b = nodes.find(n=>n.id===e.to);
    if(!a||!b)return false;
    // simple bounding-box hit test
    const ax=a.x+a.w, ay=a.y+a.h/2, bx=b.x, by=b.y+b.h/2;
    const minX=Math.min(ax,bx), maxX=Math.max(ax,bx);
    const minY=Math.min(ay,by)-4, maxY=Math.max(ay,by)+4;
    return x>=minX&&x<=maxX&&y>=minY&&y<=maxY;
  });
  if(hit){ hit.sign = hit.sign==="+"? "–" : "+"; drawAll(); }
});


canvas.addEventListener("mousemove", ev => {
  const p = mouse(ev);
  if (ghostBlock) {
    ghostBlock.x = p.x - ghostBlock.w/2;
    ghostBlock.y = p.y - ghostBlock.h/2;
    drawAll();
    return;
  }
  if (!dragNode) return;
  dragNode.x = p.x - dragOffset.x;
  dragNode.y = p.y - dragOffset.y;
  drawAll();
});
canvas.addEventListener("mouseup", () => {
  dragNode = null; canvas.style.cursor = "default";
});


/* hit-test an edge centre-line - same bbox math you already use in dbl-click */
function edgeAt(x, y) {
  return edges.find(e => {
    const a = nodes.find(n => n.id === e.from);
    const b = nodes.find(n => n.id === e.to);
    if (!a || !b) return false;
    const ax = a.x + a.w,         ay = a.y + a.h / 2;
    const bx = b.x,               by = b.y + b.h / 2;
    const minX = Math.min(ax, bx) - 4, maxX = Math.max(ax, bx) + 4;
    const minY = Math.min(ay, by) - 4, maxY = Math.max(ay, by) + 4;
    return x >= minX && x <= maxX && y >= minY && y <= maxY;
  });
}



/* ---- modal management ------------------------------------------------ */
let editTarget = null;

function openEditModal(node) {
  editTarget = node;
  const isTF    = node.type === "TF";
  const isGain  = node.type === "Gain";
  const isInput = (node.type === "Input");
  const isPID   = node.type === "PID";
  const isMux   = node.type === "Mux";
  const isDemux = node.type === "Demux";



  document.getElementById("modalTitle").textContent =
        isTF ? "Edit Transfer Function" :
        isGain ? "Edit Gain" :
        isPID ? "Edit PID" :
        isMux ? "Edit Mux" :
        isDemux ? "Edit Demux" : "Edit";
    // Toggle field groups
  document.getElementById("tfFields").style.display    = isTF ? "block" : "none";
  document.getElementById("gainFields").style.display  = isGain ? "block" : "none";
  document.getElementById("pidFields").style.display   = isPID ? "block" : "none";
  document.getElementById("muxFields").style.display   = isMux ? "block" : "none";
  document.getElementById("demuxFields").style.display = isDemux ? "block" : "none";
  document.getElementById("srcFields").style.display   = isInput ? "block":"none";

if (isInput) {
  const kind = node.params.kind || "step";
  srcSelect.value = kind;
  srcCustom.style.display = kind === "custom" ? "block" : "none";
  srcNum.value = node.params.num || "";
  srcDen.value = node.params.den || "";
}
  // Pre-fill
  if (isTF) {
    document.getElementById("numInput").value = node.params.num || "";
    document.getElementById("denInput").value = node.params.den || "";
  } else if (isGain) {
    document.getElementById("gainInput").value = node.params.k ?? "";
  } else if (isPID) {
    document.getElementById("kpInput").value = node.params.kp ?? "";
    document.getElementById("kiInput").value = node.params.ki ?? "";
    document.getElementById("kdInput").value = node.params.kd ?? "";
  } else if (isMux) {
    document.getElementById("muxInput").value = node.params.inputs ?? "";
  } else if (isDemux) {
    document.getElementById("demuxInput").value = node.params.outputs ?? "";
  }

  new bootstrap.Modal("#editModal").show();

  document.getElementById("srcSelect").onchange = (e)=>{
  document.getElementById("srcCustom").style.display =
        e.target.value === "custom" ? "block" : "none";
};
}

document.getElementById("btnModalSave").onclick = () => {
  if (!editTarget) return;
  if (editTarget.type === "TF") {
    editTarget.params.num = document.getElementById("numInput").value.trim();
    editTarget.params.den = document.getElementById("denInput").value.trim();
  } else if (editTarget.type === "Gain") {
    editTarget.params.k = parseFloat(document.getElementById("gainInput").value);
  } else if (editTarget.type === "Input") {
    const kind = srcSelect.value;
    editTarget.params.kind = kind;
    if (kind === "custom") {
      editTarget.params.num = srcNum.value.trim();
      editTarget.params.den = srcDen.value.trim();
    }
  } else if (editTarget.type === "PID") {
    editTarget.params.kp = parseFloat(document.getElementById("kpInput").value);
    editTarget.params.ki = parseFloat(document.getElementById("kiInput").value);
    editTarget.params.kd = parseFloat(document.getElementById("kdInput").value);
  } else if (editTarget.type === "Mux") {
    editTarget.params.inputs = parseInt(document.getElementById("muxInput").value);
  } else if (editTarget.type === "Demux") {
    editTarget.params.outputs = parseInt(document.getElementById("demuxInput").value);
  }

  drawAll();
  bootstrap.Modal.getInstance(document.getElementById("editModal")).hide();
};


let lastOutputTf = null;  // <-- up at top of file
let simChart = null;

function compileDiagram(){
  const domain = "s";          
  const serial = { nodes: nodes.map(n => { let c={...n}; delete c.latexEl; return c; }),
                 edges,
                 domain };

  fetch("/block_diagram/compile", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(serial)
  })
  .then(r=>r.json())
  .then(js=>{
    const box = document.getElementById("resultsBox");
    if(js.error){
      box.textContent = "Error: " + js.error;
      return;
    }

    // remember for simulate
    lastOutputTf = js.output_tf;

    // build new HTML
    box.innerHTML = `
      <h5>Loop Transfer Function</h5>
      <div id="loopOut" class="kx"></div>

      <h5 class="mt-3">Input X(s)</h5>
      <div id="inOut" class="kx"></div>

      <h5 class="mt-3">Output Y(s)</h5>
      <div id="outOut" class="kx"></div>

      <h5 class="mt-3">State-Space Form</h5>
      <div id="ssOut" class="kx"></div>

      <h5 class="mt-3">Differential / Difference Eq.</h5>
      <div id="odeOut" class="kx"></div>
    `;

    // render each one
    katex.render(js.loop_tf.latex,  document.getElementById("loopOut"));
    katex.render(js.input_tf.latex, document.getElementById("inOut"));
    katex.render(js.output_tf.latex,document.getElementById("outOut"));
    // if you want LaTeX for SS or ODE you'll need to return them too
    // for now show the raw text:
    // now render LaTeX for state-space and ODE
    katex.render(js.state_space.latex, document.getElementById("ssOut"),
                 {displayMode:true});
    katex.render(js.ode_latex,     document.getElementById("odeOut"),
                 {displayMode:false});

    // show simulate button
    document.getElementById("btnSimulate").style.display = "inline-block";
  })
  .catch(err => alert(err));
}



btnSimulate.onclick = async () => {
  // 1) Grab the last output_tf object you received
  //    (you may want to store it in a module-level variable inside compileDiagram)
  const tf = lastOutputTf;  

  // 2) Call the new simulate endpoint
  const resp = await fetch("/block_diagram/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(tf)
  });
  const sim = await resp.json();  // { time: [...], y: [...] }

  // 3) Un-hide the canvas
  simCanvas.style.display = "block";

  // 4) Draw with Chart.js
  new Chart(simCanvas.getContext("2d"), {
    type: "line",
    data: {
      labels: sim.time,
      datasets: [{
        label: "y(t)",
        data: sim.y,
        fill: false,
        borderWidth: 2
      }]
    },
    options: {
      scales: { x: { title: { display: true, text: "Time (s)" } },
                y: { title: { display: true, text: "Response" } } }
    }
  });
};


/* ── Save / Load ------------------------------------------------------ */
function downloadDiagram(){
  /* strip the KaTeX overlay handles so JSON is clean                     */
  const cleanNodes = nodes.map(({latexEl, ...rest}) => rest);
  const blob = new Blob(
      [JSON.stringify({nodes: cleanNodes, edges}, null, 2)],
      {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "diagram.bdiag";
  a.click();
  URL.revokeObjectURL(a.href);
}

function uploadDiagram(ev){
  const file = ev.target.files[0];
  if(!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    try{
      const data = JSON.parse(e.target.result);
      /* wipe existing overlays */
      document.querySelectorAll(".latexNode").forEach(el=>el.remove());
      nodes   = data.nodes || [];
      edges   = data.edges || [];
      nextId  = nodes.reduce((m,n)=>Math.max(m,n.id), 0) + 1;
      selectedNode = selectedEdge = null;
      drawAll();
    }catch(err){ alert("Invalid diagram file"); }
  };
  reader.readAsText(file);
  /* reset the <input> so the same file can be picked twice in a row */
  ev.target.value = "";
}



/* ---------------- helper fns ------------------------------------------ */
function mouse(ev){const r=canvas.getBoundingClientRect();
  return {x:ev.clientX - r.left, y:ev.clientY - r.top};}
function nodeAt(x,y){return nodes.find(n=>x>=n.x&&x<=n.x+n.w &&
                                         y>=n.y&&y<=n.y+n.h);}
function toggleConnect(){connectMode=!connectMode;
  document.getElementById("btnConnect").classList.toggle("active",connectMode);
  if(!connectMode)connectFrom=null;}
function clearScene(){nodes=[];edges=[];nextId=1;selectedNode = selectedEdge = null;
  document.querySelectorAll(".latexNode").forEach(el=>el.remove());
  (()=>{addNode("Input","X(s)",40,canvas.height/2-60);
        addNode("Output","Y(s)",canvas.width-120,canvas.height/2-60);})();
}
/* ── Load an example chosen in the <select> ───────────────────────── */
function loadSelectedPre(sel){
  const url = sel.value;
  if(!url) return;

  fetch(url)
    .then(r => r.json())
    .then(data => {
      /* clear live KaTeX overlays */
      document.querySelectorAll(".latexNode").forEach(el => el.remove());

      nodes  = data.nodes || [];
      edges  = data.edges || [];
      nextId = nodes.reduce((m,n) => Math.max(m,n.id), 0) + 1;
      selectedNode = selectedEdge = null;
      drawAll();
      sel.value = "";              // reset drop-down text
    })
    .catch(() => alert("Could not load example diagram."));
}

/* ---------------- drawing --------------------------------------------- */
function drawAll(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  arrowCol = getArrowColor();

  /* --------- edges (orthogonal) ------------------------------------ */
  ctx.strokeStyle = arrowCol;
  ctx.fillStyle   = arrowCol;
  ctx.lineWidth   = 2;
  ctx.lineJoin    = "round";
  ctx.lineCap     = "round";

 
  let changed = false;


  /* nodes */
  nodes.forEach(n=>{
    let expr = null;
    if (n.type === "TF" && n.params.num && n.params.den) {
      expr = `\\displaystyle\\frac{${n.params.num}}{${n.params.den}}`;
    } else if (n.type === "TF") {
      expr = "H(s)";
    } else if (n.type === "Gain" && Number.isFinite(n.params.k)) {
      expr = String(n.params.k);
    } else if (n.type === "Derivative") {
      expr = "s";
    } else if (n.type === "Input") {
      if      (n.params.kind === "step")    expr = "\\frac{1}{s}";
      else if (n.params.kind === "impulse") expr = "1";
      else if (n.params.kind === "custom" &&
               n.params.num && n.params.den)
        expr = `\\displaystyle\\frac{${n.params.num}}{${n.params.den}}`;
      else expr = "X(s)";
    } else if (n.type === "PID") {
      const kp = n.params.kp, ki = n.params.ki, kd = n.params.kd;
      if ([kp, ki, kd].some(v => Number.isFinite(v))) {
        const parts = [];
        if (Number.isFinite(kp)) parts.push(`${kp}`);
        if (Number.isFinite(ki)) parts.push(`\\frac{${ki}}{s}`);
        if (Number.isFinite(kd)) parts.push(`${kd} s`);
        expr = parts.join(" + ");
      } else {
        expr = "PID";
      }
    } else if (n.type === "Mux") {
      const m = n.params.inputs;
      expr = m ? `\\text{Mux}_{${m}}` : `\\text{Mux}`;
    } else if (n.type === "Demux") {
      const m = n.params.outputs;
      expr = m ? `\\text{Demux}_{${m}}` : `\\text{Demux}`;
    }

    if (expr) {
      if (renderLatex(n, expr)) changed = true;
    }

    const sel = n === selectedNode;
    ctx.fillStyle  = "#fff";
    ctx.strokeStyle = sel ? "#d9534f" : arrowCol;   // red when selected
    ctx.lineWidth   = sel ? 3 : 2;
    ctx.fillRect(n.x, n.y, n.w, n.h);
    ctx.strokeRect(n.x, n.y, n.w, n.h);

    if (!expr) {
      ctx.fillStyle="#000";ctx.font="14px sans-serif";
      const tw=ctx.measureText(n.label).width;
      ctx.fillText(n.label,n.x+n.w/2-tw/2,n.y+n.h/2+5);
      if(n.latexEl){n.latexEl.style.display="none";}
    }

  });

  edges.forEach(e => drawOrthEdge(ctx, e));

  if (ghostBlock) {
    ctx.save();
    ctx.globalAlpha = 0.5;
    ctx.setLineDash([5,5]);
    ctx.fillStyle = '#f0f0f0';
    ctx.strokeStyle = '#777';
    ctx.fillRect(ghostBlock.x, ghostBlock.y, ghostBlock.w, ghostBlock.h);
    ctx.strokeRect(ghostBlock.x, ghostBlock.y, ghostBlock.w, ghostBlock.h);
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    ctx.fillStyle = '#000';
    ctx.font = '14px sans-serif';
    const tw = ctx.measureText(ghostBlock.label).width;
    ctx.fillText(ghostBlock.label, ghostBlock.x + ghostBlock.w/2 - tw/2, ghostBlock.y + ghostBlock.h/2 + 5);
    ctx.restore();
  }

  if (changed) requestAnimationFrame(drawAll);
}

/* render LaTeX inside/above block (DOM overlay) –– centred ---------- */
function renderLatex(node, expr) {
  if (!node.latexEl) {
    node.latexEl = document.createElement("div");
    node.latexEl.className = "latexNode";
    node.latexEl.style.position = "absolute";
    node.latexEl.style.pointerEvents = "none";
    node.latexEl.style.transform = "translate(-50%, -50%)";
    document.body.appendChild(node.latexEl);
  }

  katex.render(expr, node.latexEl, { throwOnError: false });

  const rect = node.latexEl.getBoundingClientRect();
  const minW = 90, minH = 45;
  const newW = Math.max(minW, rect.width + 20);
  const newH = Math.max(minH, rect.height + 10);
  const changed = newW !== node.w || newH !== node.h;
  node.w = newW;
  node.h = newH;
  node.latexEl.style.left =
    canvas.offsetLeft + node.x + node.w / 2 + "px";
  node.latexEl.style.top =
    canvas.offsetTop + node.y + node.h / 2 + "px";
  node.latexEl.style.display = "block";
  
  return changed;
}

function arrow(x0,y0,x1,y1){
  const len=10, ang=Math.atan2(y1-y0,x1-x0);
  ctx.beginPath();ctx.moveTo(x1,y1);
  ctx.lineTo(x1-len*Math.cos(ang-Math.PI/6),
             y1-len*Math.sin(ang-Math.PI/6));
  ctx.lineTo(x1-len*Math.cos(ang+Math.PI/6),
             y1-len*Math.sin(ang+Math.PI/6));
  ctx.closePath();ctx.fill();
}

/* ------------------------------------------------------------------- */
/*  orthogonal routing that honours chosen faces                       */
/* ------------------------------------------------------------------- */

function orthRoute(p, q, exitSide){
  const gap = 30;                       // basic clearance
  const pts = [p];

  /* 1) step out of the source by 'gap'                                    */
  pts.push({
    x: p.x + DIRS[exitSide][0]*gap,
    y: p.y + DIRS[exitSide][1]*gap
  });

  /* 2) L-shaped jog to align with destination y OR x                      */
  if (exitSide === "E" || exitSide === "W"){
    pts.push({ x: pts[1].x, y: q.y });
  } else {
    pts.push({ x: q.x,      y: pts[1].y });
  }

  /* 3) finally straight into the target                                   */
  pts.push(q);
  return pts;
}


/* ------------------------------------------------------------------ */
/*  automatically pick the best sides and route a neat 3-segment path */
/* ------------------------------------------------------------------ */

function autoSides(ax, ay, bx, by){
  /* choose horizontal faces when blocks are left-vs-right, otherwise vertical */
  if (Math.abs(bx - ax) >= Math.abs(by - ay))
    return { out: (bx >= ax ? "E" : "W"),
             inn: (bx >= ax ? "W" : "E") };
  else
    return { out: (by >= ay ? "S" : "N"),
             inn: (by >= ay ? "N" : "S") };
}

function smartRoute(p, q, outSide, inSide){
  const GAP = 30;                    // clearance away from the block face
  const pts = [p];

  /* 1) step clear of the source block                                  */
  pts.push({ x: p.x + DIRS[outSide][0]*GAP,
             y: p.y + DIRS[outSide][1]*GAP });

  /* 2) orthogonal jog that lines us up with the entry face             */
  if (outSide === "E" || outSide === "W")
       pts.push({ x: pts[1].x, y: q.y });
  else pts.push({ x: q.x,      y: pts[1].y });

  /* 3) straight into the destination                                   */
  pts.push(q);
  return pts;
}


function drawOrthEdge(ctx, e){
  const a = nodes.find(n => n.id === e.from);
  const b = nodes.find(n => n.id === e.to);
  ctx.strokeStyle = (e === selectedEdge) ? "#d9534f" : arrowCol;
  ctx.fillStyle   = (e === selectedEdge) ? "#d9534f" : arrowCol;
  ctx.lineWidth   = (e === selectedEdge) ? 3 : 2;
  if (!a || !b) return;

  /* centre-points for relative geometry -------------------------------- */
  const ac = { x: a.x + a.w/2, y: a.y + a.h/2 };
  const bc = { x: b.x + b.w/2, y: b.y + b.h/2 };

  /* faces decided fresh every frame – no more “wrong side” wires ------- */
  const face = autoSides(ac.x, ac.y, bc.x, bc.y);

  const from = portPos(a, face.out);
  const  to  = portPos(b, face.inn);

  const pts  = smartRoute(from, to, face.out, face.inn);

  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.stroke();

  /* arrow-head on final leg */
  const n = pts.length;
  arrow(pts[n-2].x, pts[n-2].y, pts[n-1].x, pts[n-1].y);

  /* badge on negative edges – halfway down the 2nd segment */
  if (e.sign === "–"){
    const mid = {
      x:(pts[1].x + pts[2].x)/2,
      y:(pts[1].y + pts[2].y)/2
    };
    ctx.fillStyle="#fff"; ctx.strokeStyle="#000";
    ctx.beginPath(); ctx.arc(mid.x, mid.y, 7, 0, 2*Math.PI);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle="#000"; ctx.font="12px sans-serif";
    ctx.fillText("–", mid.x-3, mid.y+4);
  }
}

/* ---------------- toolbar hooks --------------------------------------- */
document.getElementById("btnClear").onclick=clearScene;
document.getElementById("btnCompile").onclick=compileDiagram;

document.addEventListener("keydown", ev => {
  if (ev.key !== "Delete" && ev.key !== "Backspace") return;

  // dot delete anything while the edit modal is open
  const modal = document.getElementById("editModal");
  if (modal && modal.classList.contains("show")) return;

  if (selectedNode) {
    // remove latex overlay first
    if (selectedNode.latexEl) selectedNode.latexEl.remove();
    nodes  = nodes.filter(n => n !== selectedNode);
    edges  = edges.filter(e => e.from !== selectedNode.id && e.to !== selectedNode.id);
    selectedNode = null;
  } else if (selectedEdge) {
    edges = edges.filter(e => e !== selectedEdge);
    selectedEdge = null;
  }
  drawAll();
});

/* toolbar button does the same */
document.getElementById("btnDelete").onclick =
  () => document.dispatchEvent(new KeyboardEvent("keydown",{key:"Delete"}));



document.getElementById("btnSimulate").onclick = async () => {
  if (!lastOutputTf) return alert("Nothing to simulate!");

  // 1) POST the most recent TF
  const resp = await fetch("/block_diagram/simulate", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(lastOutputTf)
  });
  const sim = await resp.json();

  // 2) Reveal and clear out old chart
  simCanvas.style.display = "block";
  if (simChart) simChart.destroy();

  // 3) Draw new one and save the instance
  simChart = new Chart(simCanvas.getContext("2d"), {
    type: "line",
    data: {
      labels: sim.time,
      datasets: [{
        label:       "y(t)",
        data:        sim.y,
        fill:        false,
        borderWidth: 2
      }]
    },
    options: {
      scales: {
        x: { title: { display: true, text: "Time (s)" } },
        y: { title: { display: true, text: "Response" } }
      }
    }
  });
};


/* initial paint */
initPalette();
document.querySelectorAll('.kx-btn').forEach(el=>katex.render(el.textContent, el));
drawAll();
