/* =========================================================================
   Block-Diagram UI  –  v0.3  (editable TF blocks with KaTeX overlay)
   ========================================================================= */

const canvas = document.getElementById("diagramCanvas");
const ctx    = canvas.getContext("2d");
/* --- modal DOM shortcuts (must exist before we call openEditModal) --- */
const srcSelect = document.getElementById("srcSelect");
const srcCustom = document.getElementById("srcCustom");
const srcNum    = document.getElementById("srcNum");
const srcDen    = document.getElementById("srcDen");


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

/* ---------------- palette helpers ------------------------------------ */
function addNode(type, label, x = 120, y = 80) {
  nodes.push({
    id: nextId++,
    type,
    label,
    x, y,
    w: 90,
    h: 45,
    params: {},            // for TF blocks: {num:'', den:''}
    latexEl: null          // DOM <div> where KaTeX renders
  });
  drawAll();
}

/* default source & sink ------------------------------------------------ */
(() => {
  addNode("Input",  "x(t)", 40, canvas.height/2 - 60);
  addNode("Output", "y(t)", canvas.width - 120, canvas.height/2 - 60);
})();

/* ---------------- connection mode & drag ------------------------------ */
let connectMode = false, connectFrom = null;
let dragNode = null, dragOffset = {x:0,y:0};

canvas.addEventListener("mousedown", ev => {
  const p = mouse(ev), n = nodeAt(p.x, p.y);

  /* connect mode */
  if (connectMode) {
    if (!n) return;
    if (!connectFrom) { connectFrom = n; return; }
    if (n !== connectFrom) 
        edges.push({
        from: connectFrom.id,
        to:   n.id,
        sign: "+",
        outSide: chooseSide(connectFrom, /*out =*/true),
        inSide:  chooseSide(n,          /*out =*/false)
        });
    connectFrom = null;
    drawAll(); return;
  }

  /* drag or edit */
  if (n) {
    if (n && ev.detail === 2 && (n.type === "TF" || n.type === "Gain" || n.type === "Input" || n.type === "Source")) openEditModal(n); // double-click
    else {  // drag
      dragNode = n;
      dragOffset = {x:p.x - n.x, y:p.y - n.y};
      canvas.style.cursor = "grabbing";
    }
  }
});

canvas.addEventListener("dblclick", ev=>{
  const {x, y} = mousePos(ev);
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
  if (!dragNode) return;
  const p = mouse(ev);
  dragNode.x = p.x - dragOffset.x;
  dragNode.y = p.y - dragOffset.y;
  drawAll();
});
canvas.addEventListener("mouseup", () => {
  dragNode = null; canvas.style.cursor = "default";
});



/* ---- modal management ------------------------------------------------ */
let editTarget = null;

function openEditModal(node) {
  editTarget = node;
  const isTF   = node.type === "TF";
  const isGain = node.type === "Gain";
  const isSource = (node.type === "Source" || node.type === "Input");


  document.getElementById("modalTitle").textContent =
        isTF ? "Edit Transfer Function" :
        isGain ? "Edit Gain" : "Edit";

  // Toggle field groups
  document.getElementById("tfFields").style.display   = isTF ? "block" : "none";
  document.getElementById("gainFields").style.display = isGain ? "block" : "none";
  document.getElementById("srcFields").style.display  = isSource ? "block":"none";

if (isSource) {
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
  }
  else if (editTarget.type === "Source") {
  const kind = document.getElementById("srcSelect").value;
  editTarget.params.kind = kind;
  if (kind === "custom") {
    editTarget.params.num =
        document.getElementById("srcNum").value.trim();
    editTarget.params.den =
        document.getElementById("srcDen").value.trim();
  }
}
else if (editTarget.type === "Input") {          // NEW
  const kind = srcSelect.value;
  editTarget.params.kind = kind;
  if (kind === "custom") {
    editTarget.params.num = srcNum.value.trim();
    editTarget.params.den = srcDen.value.trim();
  }
}
  drawAll();
  bootstrap.Modal.getInstance(document.getElementById("editModal")).hide();
};


/* ---------------- compile --------------------------------------------- */
function compileDiagram(){
  const domain = document.getElementById("domainSel").value;

  // detach KaTeX div refs before serialising
  const serialNodes = nodes.map(n=>{
    const copy = {...n};
    delete copy.latexEl;
    return copy;
  });

  fetch("/block_diagram/compile", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({nodes:serialNodes, edges, domain})
  })
  .then(r=>r.json())
    .then(js=>{
      const box = document.getElementById("resultsBox");
      if (js.error) { box.textContent = "Error: " + js.error; return; }

      // clear old content
      box.innerHTML = `
        <h5>Transfer&nbsp;Function</h5>
        <div id="tfOut"  class="kx"></div>
        <h5 class="mt-3">State-Space&nbsp;form</h5>
        <div id="ssOut"  class="kx"></div>
        <h5 class="mt-3">Differential / Difference&nbsp;Eq.</h5>
        <div id="odeOut" class="kx"></div>
      `;
      katex.render(js.transfer_function.latex, document.getElementById("tfOut"));
      katex.render(js.state_space.latex,      document.getElementById("ssOut"),
                   {displayMode:true});
      katex.render(js.ode_latex,              document.getElementById("odeOut"));
  })
  .catch(alert);
}

/* ---------------- helper fns ------------------------------------------ */
function mouse(ev){const r=canvas.getBoundingClientRect();
  return {x:ev.clientX - r.left, y:ev.clientY - r.top};}
function nodeAt(x,y){return nodes.find(n=>x>=n.x&&x<=n.x+n.w &&
                                         y>=n.y&&y<=n.y+n.h);}
function toggleConnect(){connectMode=!connectMode;
  document.getElementById("btnConnect").classList.toggle("active",connectMode);
  if(!connectMode)connectFrom=null;}
function clearScene(){nodes=[];edges=[];nextId=1;
  document.querySelectorAll(".latexNode").forEach(el=>el.remove());
  (()=>{addNode("Input","x(t)",40,canvas.height/2-60);
        addNode("Output","y(t)",canvas.width-120,canvas.height/2-60);})();
}

/* ---------------- drawing --------------------------------------------- */
function drawAll(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = getComputedStyle(canvas).getPropertyValue('--arrow-col').trim() || "#000";

  /* --------- edges (orthogonal) ------------------------------------ */
  ctx.strokeStyle = "#000";
  ctx.lineWidth   = 2;
  ctx.lineJoin    = "round";
  ctx.lineCap     = "round";

  edges.forEach(e => drawOrthEdge(ctx, e));


  /* nodes */
  nodes.forEach(n=>{
    ctx.fillStyle="#fff";ctx.strokeStyle="#000";ctx.lineWidth=2;
    ctx.fillRect(n.x,n.y,n.w,n.h);ctx.strokeRect(n.x,n.y,n.w,n.h);

    if (n.type === "TF" && n.params.num && n.params.den) {
      renderLatex(n, `\\displaystyle\\frac{${n.params.num}}{${n.params.den}}`);
    } else if (n.type === "Gain" && Number.isFinite(n.params.k)) {          // NEW
      renderLatex(n, String(n.params.k));
    } 
    else if (n.type === "Source" || n.type === "Input") {
        let expr;
        if      (n.params.kind === "step")    expr = "\\frac{1}{s}";
        else if (n.params.kind === "impulse") expr = "1";
        else if (n.params.kind === "custom" &&
                n.params.num && n.params.den)
                expr = `\\displaystyle\\frac{${n.params.num}}{${n.params.den}}`;
        renderLatex(n, expr || (n.type === "Input" ? "x(t)" : "SRC"));
        }
    else if (n.type === "Input") {                   // NEW
    let expr;
    if      (n.params.kind === "step")    expr = "\\frac{1}{s}";
    else if (n.params.kind === "impulse") expr = "1";
    else if (n.params.kind === "custom" &&
            n.params.num && n.params.den)
            expr = `\\displaystyle\\frac{${n.params.num}}{${n.params.den}}`;
    renderLatex(n, expr || "x(t)");
    }

    else {
      ctx.fillStyle="#000";ctx.font="14px sans-serif";
      const tw=ctx.measureText(n.label).width;
      ctx.fillText(n.label,n.x+n.w/2-tw/2,n.y+n.h/2+5);
      if(n.latexEl){n.latexEl.style.display="none";}
    }

  });
}

/* render LaTeX inside/above block (DOM overlay) ------------------------ */
/* render LaTeX inside/above block (DOM overlay) –– centred ---------- */
function renderLatex(node, expr) {
  if (!node.latexEl) {
    node.latexEl = document.createElement("div");
    node.latexEl.className = "latexNode";
    node.latexEl.style.position = "absolute";
    node.latexEl.style.pointerEvents = "none";
    node.latexEl.style.transform = "translate(-50%, -50%)"; // <<< NEW
    document.body.appendChild(node.latexEl);
  }

  katex.render(expr, node.latexEl, { throwOnError: false });

  // place origin at the block centre, let CSS do the centring
  node.latexEl.style.left =
    canvas.offsetLeft + node.x + node.w / 2 + "px";
  node.latexEl.style.top =
    canvas.offsetTop + node.y + node.h / 2 + "px";
  node.latexEl.style.display = "block";
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

function drawOrthEdge(ctx, e){
  const a = nodes.find(n => n.id === e.from);
  const b = nodes.find(n => n.id === e.to);
  if (!a || !b) return;

  const from = portPos(a, e.outSide);
  const to   = portPos(b, e.inSide);
  const pts  = orthRoute(from, to, e.outSide);

  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.stroke();

  /* arrow-head on final leg */
  const n = pts.length;
  arrow(pts[n-2].x, pts[n-2].y, pts[n-1].x, pts[n-1].y);

  /* ⊖ badge on negative edges – halfway down the 2nd segment */
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
document.getElementById("btnAddSource").onclick = () => addNode("Source", "", 60, 80);
document.getElementById("btnAddGain").onclick =() => addNode("Gain", "", 180, 80);
document.getElementById("btnAddAdder").onclick=()=>addNode("Adder","Σ");
document.getElementById("btnAddIntegrator").onclick=()=>addNode("Integrator","1/s");
document.getElementById("btnAddDelay").onclick=()=>addNode("Delay","z⁻¹");
document.getElementById("btnAddTF").onclick=()=>addNode("TF","",180,80);  // NEW
document.getElementById("btnClear").onclick=clearScene;
document.getElementById("btnCompile").onclick=compileDiagram;

/* initial paint */
drawAll();
