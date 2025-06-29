/* ------------------------------------------------------------------
   Process‑Chain front‑end  (canvas UI + modals)
------------------------------------------------------------------ */

// Return the first unused connection letter algotithm
function nextLetter() {
  const used = new Set(lines.map(l => l.letter.toLowerCase()));
  let code = 'a'.charCodeAt(0);
  while (used.has(String.fromCharCode(code))) {
    code++;
  }
  return String.fromCharCode(code);
}
let lineSelected  = null;                // currently highlighted line

let blocks = [];            // all block objects
let lines  = [];            // connection objects
let blockIdCounter = 0;

let connectMode   = false;
let connectStart  = null;

let selectedId    = null;
let draggingBlock = null;
let dragDX = 0, dragDY = 0;

let blockInModal       = null;   // generic
let blockInFilterModal = null;   // filter
let blockInMultModal   = null;   // multiplication

// cached DOM
const canvas           = document.getElementById("chainCanvas");
const ctx              = canvas.getContext("2d");
function getArrowColor(){
  const val = getComputedStyle(canvas).getPropertyValue('--arrow-col').trim();
  if (val) return val;
  return document.body.classList.contains('dark-mode') ? '#ddd' : '#000';
}

function getBlockFillColor(){
  return document.body.classList.contains('dark-mode') ? '#2a2a2a' : '#fff';
}

function getTextColor(){
  return document.body.classList.contains('dark-mode') ? '#f5f5f5' : '#000';
}

const multType         = document.getElementById("multType");
const multParamDiv     = document.getElementById("multParamDiv");
const multParamValue   = document.getElementById("multParamValue");
const samplingDiv      = document.getElementById("samplingDiv");
const samplingInterval = document.getElementById("samplingInterval");
const multHint         = document.getElementById("multHint");

/* ===============================================================
/* circle helper                                              */
function circle(ctx, x, y, r){
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2*Math.PI);
}


/* ================================================================ */
/*  Block helpers                                                   */
/* ================================================================ */
function getBlock(id) {
  return blocks.find(b => b.id === id);
}
function findBlockAt(x, y) {
  return blocks.slice().reverse().find(b =>
    x >= b.x && x <= b.x + b.width &&
    y >= b.y && y <= b.y + b.height
  ) || null;
}
function findLineAt(x, y) {
  const tol2 = 12 * 12;
  for (let i = lines.length - 1; i >= 0; --i) {
    const l = lines[i];
    const a = getBlock(l.fromId), b = getBlock(l.toId);
    if (!a || !b) continue;
    const p1 = connPt(a, b), p2 = connPt(b, a);
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len2 = dx*dx + dy*dy || 1;
    const t = ((x - p1.x)*dx + (y - p1.y)*dy) / len2;
    const tc = Math.max(0, Math.min(1, t));
    const px = p1.x + tc * dx, py = p1.y + tc * dy;
    const d2 = (x - px)**2 + (y - py)**2;
    if (d2 < tol2) return l;
  }
  return null;
}
// create Node func.
function createNode(x, y, text, nodeType, nonDeletable = false) {
  /* block dimensions */
  let w = 90, h = 90;                   // rectangles (default)
  if (nodeType === "Multiplication" || nodeType === "Addition") {
      w = h = 48;                       // 24‑px radius circle
  }
  if (nodeType === "Dot") {
      w = h = 8;                        // tiny dot
  }

  blocks.push({
      id: ++blockIdCounter,
      type: nodeType,
      label: text,
      x, y,
      width:  w,
      height: h,
      param: null,
      displayExpr: null,
      nonDeletable,
      selected: false
  });
  drawAll();
  return blocks[blocks.length - 1];
}




function addBlock(type,label,x,y,nonDel=false) {
  const bx = x ?? (120 + blocks.length*12);
  const by = y ?? (60  + blocks.length*12);
  blocks.push({
    id: ++blockIdCounter, type, label,
    x: bx, y: by, width: 100, height: 50,
    param:null, displayExpr:null,
    nonDeletable:nonDel, selected:false
  });
  drawAll();
}
/* ================================================================ */
/*  Drawing                                                         */
/* ================================================================ */
function connPt(a, b) {
  /* centres */
  const ax = a.x + a.width  / 2,
        ay = a.y + a.height / 2,
        bx = b.x + b.width  / 2,
        by = b.y + b.height / 2;

  /* circle blocks ------------------------------------------ */
  if (a.type === "Multiplication" || a.type === "Addition") {
      const outline = 3;                         // stroke thickness
      const r  = a.width / 2 - outline / 2;      // visual radius
      const dx = bx - ax,  dy = by - ay;
      const len = Math.hypot(dx, dy) || 1;       // normalise
      return { x: ax + (dx / len) * r,
               y: ay + (dy / len) * r };
  }

  /* tiny dot ------------------------------------------------ */
  if (a.type === "Dot") {
      const r = a.width / 2;
      const dx = bx - ax,  dy = by - ay;
      const len = Math.hypot(dx, dy) || 1;
      return { x: ax + (dx / len) * r,
               y: ay + (dy / len) * r };
  }

  /* rectangle ---------------------------------------------- */
  if (bx < ax) return { x: a.x,           y: ay };   // left centre
  return          { x: a.x + a.width, y: ay };       // right centre
}


/* ------------------------------------------------------------------
   Redraw everything: connections (with arrow‑heads) and blocks
------------------------------------------------------------------ */
function drawAll () {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const arrowCol = getArrowColor();
  const fillCol  = getBlockFillColor();
  const textCol  = getTextColor();

  /* ========== draw connecting lines (with arrow heads) ========== */
  ctx.strokeStyle = arrowCol;
  ctx.lineWidth   = 2;
  ctx.fillStyle   = arrowCol;

  for (const l of lines) {
    const a = getBlock(l.fromId), b = getBlock(l.toId);
    if (!a || !b) continue;

    const p1 = connPt(a, b);          // start point on block a
    const p2 = connPt(b, a);          // end   point on block b
    ctx.strokeStyle = l.selected ? "red" : arrowCol;

    /* main segment */
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    /* ==== letter at mid-point ==== */
    const mx = (p1.x + p2.x) / 2,
    my = (p1.y + p2.y) / 2;

    ctx.font = "15px sans-serif";
    ctx.fillStyle = l.selected ? "red" : arrowCol;
    ctx.fillText(`${l.letter}(t)`, mx + 6, my - 6);

    /* arrow head (size 6 px) at p2, pointing towards b */
    const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);
    const ah    = 6;
    ctx.beginPath();
    ctx.moveTo(p2.x, p2.y);
    ctx.lineTo(p2.x - ah * Math.cos(angle - Math.PI / 6),
               p2.y - ah * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(p2.x - ah * Math.cos(angle + Math.PI / 6),
               p2.y - ah * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  /* helper for circles */
  const circle = (x,y,r)=>{
    ctx.beginPath();
    ctx.arc(x,y,r,0,2*Math.PI);
  };

  /* ========== draw every block (shapes + labels) ========== */
  for (const b of blocks) {
    const cx = b.x + b.width  / 2;
    const cy = b.y + b.height / 2;

    ctx.lineWidth   = b.selected ? 4 : 3;
    ctx.strokeStyle = arrowCol;
    ctx.fillStyle   = fillCol;

    /* --- Multiplication circle -------------------------- */
    if (b.type === "Multiplication") {
      const r = 24;
      circle(cx, cy, r); ctx.fill(); ctx.stroke();

      /* × symbol */
      ctx.beginPath();
      ctx.moveTo(cx - r * 0.6, cy - r * 0.6);
      ctx.lineTo(cx + r * 0.6, cy + r * 0.6);
      ctx.moveTo(cx + r * 0.6, cy - r * 0.6);
      ctx.lineTo(cx - r * 0.6, cy + r * 0.6);
      ctx.stroke();

      /* expression + arrow from above */
      if (b.displayExpr) {
        ctx.font = "16px serif"; ctx.fillStyle = arrowCol;
        const tw   = ctx.measureText(b.displayExpr).width;
        const txtY = b.y - 30;                    // 30 px above top
        ctx.fillText(b.displayExpr, cx - tw / 2, txtY);

        /* arrow shaft */
        ctx.beginPath();
        ctx.moveTo(cx, txtY + 4);
        ctx.lineTo(cx, cy - r);
        ctx.stroke();

        /* arrow head into circle */
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy - r + 8);
        ctx.lineTo(cx,     cy - r);
        ctx.lineTo(cx + 5, cy - r + 8);
        ctx.stroke();
      }

    /* --- Addition circle -------------------------------- */
    } else if (b.type === "Addition") {
      const r = 22;
      circle(cx, cy, r); ctx.fill(); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - r * 0.6, cy);
      ctx.lineTo(cx + r * 0.6, cy);
      ctx.moveTo(cx, cy - r * 0.6);
      ctx.lineTo(cx, cy + r * 0.6);
      ctx.stroke();

    /* --- tiny Dot -------------------------------------- */
    } else if (b.type === "Dot") {
      circle(cx, cy, 4); ctx.fill(); ctx.stroke();

    /* --- Rectangles (Re, Im, Filter, x(t), y(t)… ) ------ */
    } else {
      ctx.fillRect(b.x, b.y, b.width, b.height);
      ctx.strokeRect(b.x, b.y, b.width, b.height);

      ctx.font = "17px serif"; ctx.fillStyle = textCol;
      const tw = ctx.measureText(b.label).width;
      ctx.fillText(b.label, cx - tw / 2, cy + 6);
    }
  }
}


// ================================================================
//  Mouse interaction
// ================================================================
canvas.addEventListener("mousedown", e => {
  const r = canvas.getBoundingClientRect(),
        x = e.clientX - r.left,
        y = e.clientY - r.top;

          /**** CONNECT-MODE: click twice to draw an arrow ****/
          if (connectMode) {
            const b = findBlockAt(x, y);
            if (!connectStart && b) {
              // first click: pick the “from” block
              connectStart = b;
              b.selected   = true;
              drawAll();
              return;
            }
            if (connectStart && b && connectStart !== b) {
              // second click: push a new connection into lines[]
              lines.push({
                fromId: connectStart.id,
                toId:   b.id,
                letter: nextLetter()            // reuse first available letter
              });
              connectStart.selected = false;
              connectStart = null;
              drawAll();
              return;
            }
            // click on empty or same block — stay in connectMode
            return;
          }

  // Not in connect-mode → selection or dragging
  let b = findBlockAt(x, y);
  let l = !b ? findLineAt(x, y) : null;

  if (b) {
    // start dragging or selecting a block
    draggingBlock = b; 
    dragDX = x - b.x; 
    dragDY = y - b.y;

    // clear previous highlight
    if (selectedId && selectedId !== b.id) getBlock(selectedId).selected = false;
    if (lineSelected) lineSelected.selected = false;

    // select this block
    b.selected    = true;
    selectedId    = b.id;
    lineSelected  = null;
    drawAll();
    return;
  }

  if (l) {
    // clicked a line → select it
    if (lineSelected) lineSelected.selected = false;
    if (selectedId)   getBlock(selectedId).selected = false;

    l.selected      = true;
    lineSelected    = l;
    selectedId      = null;
    drawAll();
    return;
  }

  // clicked empty space → clear selection
  if (selectedId)   { getBlock(selectedId).selected = false; selectedId = null; }
  if (lineSelected) { lineSelected.selected = false; lineSelected = null; }
  drawAll();
});

canvas.addEventListener("mousemove", e => {
  if (!draggingBlock) return;
  const r = canvas.getBoundingClientRect();
  draggingBlock.x = e.clientX - r.left - dragDX;
  draggingBlock.y = e.clientY - r.top  - dragDY;
  drawAll();
});

canvas.addEventListener("mouseup", () => {
  draggingBlock = null;
});

window.addEventListener("keydown", e => {
  if (e.key === "Delete") deleteSelected();
});


/* ================================================================
   Single-click on a letter → partial plot
================================================================ */
canvas.addEventListener("click", e => {
  if (connectMode) return;
  const { x, y } = getMousePos(e);
  if (findBlockAt(x, y)) return;
  const l = findLineAt(x, y);
  if (!l) return;
  fetch("/process_chain/compute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: document.getElementById("inputExpression").value,
      blocks, lines,
      until: l.fromId
    })
  })
  .then(r => r.json())
  .then(d => {
    if (d.error) alert("Compute error: " + d.error);
    else document.getElementById("plotResult").innerHTML =
      `<img src="data:image/png;base64,${d.plot_data}">`;
  });
});

canvas.addEventListener("dblclick", e => {
  const { x, y } = getMousePos(e);
  const b = findBlockAt(x, y);
  if (b) openBlockModal(b);
});

function getMousePos(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

/* ================================================================ */
/*  Modals                                                          */
/* ================================================================ */
function openBlockModal(b){
  if (["Hilbert","Derivative","d/dt","Re","Im",
     "Integrator","Power","Conjugate","Input",
     "Output","Addition","Subtraction"].includes(b.type)) return;
  if(b.type==="Filter")        openFilterModal(b);
  else if(b.type==="Multiplication") openMultiplicationModal(b);
  else{
    document.getElementById("blockModal").style.display="block";
    document.getElementById("blockTypeLabel").innerText="Editing: "+b.type;
    document.getElementById("blockParamInput").value=b.param||"";
    blockInModal=b;
  }
}
/* ---------- Multiplication modal ---------- */
const multHintMap = {
  constant    :'Format: K → <code>4</code>',
  imaginary   :'Optional factor: K → <code>imaginary:3</code>',
  linear      :'Format: A → <code>2</code> (result = A·ω)',
  sin         :'A,ω₀ or j,A,ω₀ → <code>4,2</code> or <code>j,4,2</code>',
  cos         :'A,ω₀ or j,A,ω₀ → <code>3,1</code> or <code>j,3,1</code>',
  exponential :'K,±,ω₀ → <code>2,+,5</code> or <code>+,5</code>',
  sampling    :'Sampling interval T – keep ω = kT'
};
function updateMultHint(){
  multHint.innerHTML=multHintMap[multType.value];
  /* toggle inputs */
  if(multType.value==="sampling"){
    samplingDiv.style.display="block"; multParamDiv.style.display="none";
  }else if(multType.value==="imaginary"){
    samplingDiv.style.display="none";  multParamDiv.style.display="block";
  }else{
    samplingDiv.style.display="none";  multParamDiv.style.display="block";
  }
}
multType.addEventListener("change",updateMultHint);

function openMultiplicationModal(b){
  blockInMultModal=b;
  /* preload */
  if(b.param){
    const [tp,val=""] = b.param.split(":");
    multType.value=tp; multParamValue.value=val;
    if(tp==="sampling") samplingInterval.value=parseFloat(val||1);
  }else{
    multType.value="constant"; multParamValue.value="";
  }
  updateMultHint();
  document.getElementById("multiplicationModal").style.display="block";
}
function closeMultModal(){
  document.getElementById("multiplicationModal").style.display="none";
  blockInMultModal=null;
}
document.getElementById("multModalCancel").onclick=closeMultModal;
document.getElementById("multModalOk").onclick = function () {

  if (!blockInMultModal) { closeMultModal(); return; }

  const tp = multType.value;                     // dropdown choice
  const raw = multParamValue.value.trim();       // user text
  let paramStr = "";

  /* ---------- build param string (internal) ---------- */
  if (tp === "sampling") {
      paramStr = `sampling:${samplingInterval.value}`;
  } else if (tp === "imaginary") {
      paramStr = raw ? `imaginary:${raw}` : "imaginary";
  } else {
      paramStr = `${tp}:${raw}`;
  }

  /* ---------- prettify ---------- */
  function nice(tp, raw){
      const ω = "ω";                 // feel free to replace with ω₀ if needed

      if (tp === "constant") return raw;

      if (tp === "imaginary"){
          return raw ? `${raw} j` : "j";
      }

      if (tp === "linear"){          // A →  A ω
          return `${raw} ${ω}`;
      }

      if (tp === "sin" || tp === "cos"){
          // possible raw formats:  "4,2"   or  "j,4,2"
          const parts = raw.split(",").map(s=>s.trim()).filter(s=>s!=="");
          let jflag = false;
          if (parts[0] === "j"){ jflag = true; parts.shift(); }
          const A   = parts[0] || "1";
          const w0  = parts[1] || "1";
          const core = `${A} ${tp}(${w0}${ω} t)`;
          return jflag ? `j · ${core}` : core;
      }

      if (tp === "exponential"){
          // raw =  "K,+,w0"  or "+,w0"
          const p = raw.split(",").map(s=>s.trim());
          let K, sign, w0;
          if (p.length === 3){ K=p[0]; sign=p[1]; w0=p[2]; }
          else { K=""; sign=p[0]; w0=p[1]; }
          const amp = K && K!=="1" ? `${K} ` : "";
          return `${amp}e^{${sign==="-"?"‑":"+"}j ${w0}${ω} t}`;
      }

      if (tp === "sampling") return "⟂";

      return raw;   // fallback
  }

  const pretty = (()=>{
      const [kind,value=""] = paramStr.split(":");
      return nice(kind,value);
  })();

  /* ---------- save to block ---------- */
  blockInMultModal.param       = paramStr;
  blockInMultModal.label       = "×";
  blockInMultModal.displayExpr = pretty;

  closeMultModal();
  drawAll();
};


/* ---------- Filter‑modal ---------- */
function openFilterModal(b){
  blockInFilterModal = b;
  const fm = document.getElementById("filterModal");
  fm.style.display = "block";

  // preload controls
  if(b.param){
    const [type,rest=""] = b.param.split(":");
    document.getElementById("filterType").value = type;
    if(type==="bandpass"){
      document.getElementById("bandCutoffs").style.display="block";
      document.getElementById("singleCutoff").style.display="none";
      const [lo="0.5",hi="2"] = rest.split(",");
      document.getElementById("lowCutoff").value  = parseFloat(lo);
      document.getElementById("highCutoff").value = parseFloat(hi);
    }else{
      document.getElementById("singleCutoff").style.display="block";
      document.getElementById("bandCutoffs").style.display="none";
      document.getElementById("cutoffValue").value = parseFloat(rest||1);
    }
  }else{
    document.getElementById("filterType").value="lowpass";
    document.getElementById("singleCutoff").style.display="block";
    document.getElementById("bandCutoffs").style.display="none";
  }
}

/* dropdown toggle */
document.getElementById("filterType").addEventListener("change",e=>{
  const t=e.target.value;
  if(t==="bandpass"){
    document.getElementById("bandCutoffs").style.display="block";
    document.getElementById("singleCutoff").style.display="none";
  }else{
    document.getElementById("bandCutoffs").style.display="none";
    document.getElementById("singleCutoff").style.display="block";
  }
});

/* OK / Cancel */
document.getElementById("filterModalOk").onclick = ()=>{
  if(blockInFilterModal){
    const t=document.getElementById("filterType").value;
    let paramStr="";
    if(t==="bandpass"){
      const lo=document.getElementById("lowCutoff").value;
      const hi=document.getElementById("highCutoff").value;
      paramStr=`bandpass:${lo},${hi}`;
    }else{
      const c=document.getElementById("cutoffValue").value;
      paramStr=`${t}:${c}`;
    }
    blockInFilterModal.param = paramStr;
    blockInFilterModal.label = "Filter\n("+paramStr+")";
  }
  closeFilterModal(); drawAll();
};
function closeFilterModal(){
  document.getElementById("filterModal").style.display="none";
  blockInFilterModal=null;
}
document.getElementById("filterModalCancel").onclick = closeFilterModal;
/* ---------------------------------- */

/* ================================================================
   Toolbar actions
================================================================ */
window.toggleConnectMode = () => {
  connectMode = !connectMode;
  document.getElementById("btnConnect").classList.toggle("active", connectMode);
  if (connectStart) {
    connectStart.selected = false;
    connectStart = null;
  }
  drawAll();
};

function deleteSelected() {
  if (selectedId) {
    const b = getBlock(selectedId);
    if (b.nonDeletable) return;
    lines = lines.filter(l => l.fromId !== selectedId && l.toId !== selectedId);
    blocks = blocks.filter(bl => bl.id !== selectedId);
    selectedId = null;
    drawAll();
    return;
  }
  if (lineSelected) {
    lines = lines.filter(l => l !== lineSelected);
    lineSelected = null;
    drawAll();
  }
}
window.deleteSelected = deleteSelected;

window.clearAll = () => {
  blocks = blocks.filter(b => b.nonDeletable);
  lines  = [];
  selectedId   = null;
  lineSelected = null;
  drawAll();
};

/* ================================================================ */
/*  Save / Load scene                                               */
/* ================================================================ */

/* ---------- download ---------- */
window.downloadScene = () => {
  const payload = {
    blocks,                      // current block array
    lines,                       // current connections
    canvas: { w: canvas.width, h: canvas.height },   // optional
    version: "1.0"
  };

  const blob = new Blob(
      [ JSON.stringify(payload, null, 2) ],
      { type: "application/json" });

  const url = URL.createObjectURL(blob);
  const a   = document.createElement("a");
  a.href = url;
  a.download = "process_chain.chain";    // custom extension
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};


/* ---------- upload ---------- */
window.uploadScene = evt => {
  const file = evt.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = e => {
    try {
      const state = JSON.parse(e.target.result);

      /* re-hydrate core arrays (fall back to empty if missing) */
      blocks = state.blocks || [];
      lines  = state.lines  || [];

      /* you can restore extra view settings here if you add them later */
      drawAll();                 // redraw everything
    }
    catch (err) {
      alert("Could not load chain: " + err.message);
    }
    evt.target.value = "";       // allow re-selecting the same file
  };
  reader.readAsText(file);
};

/* Load a pre-packaged .chain file chosen in <select> */
function loadSelectedPreloaded(sel) {
  const url = sel.value;
  if (!url) { return; }                  // user picked the placeholder line

  fetch(url)
    .then(r => r.text())
    .then(text => {
      /* fake a “file-input change” event so we can re-use uploadScene() */
      const blob   = new Blob([text], {type: "application/json"});
      const file   = new File([blob], "from_preloaded.chain");
      const event  = { target: { files: [file] } };
      uploadScene(event);
      sel.selectedIndex = 0;             // reset dropdown label
    })
    .catch(err => alert("Could not load chain:\n" + err));
}



window.computeChain = () => {
  fetch("/process_chain/compute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: document.getElementById("inputExpression").value, blocks, lines })
  })
  .then(r => r.json())
  .then(d => {
    if (d.error) alert("Compute error: " + d.error);
    else document.getElementById("plotResult").innerHTML =
      `<h3>Y(jω)</h3><img src="data:image/png;base64,${d.plot_data}">`;
  });
};

window.computeChain = () => {
  fetch("/process_chain/compute", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      input:  document.getElementById("inputExpression").value,
      blocks: blocks,
      lines:  lines
    })
  })
  .then(r => r.json())
  .then(d => {
    if (d.error) alert("Compute error: " + d.error);
    else document.getElementById("plotResult").innerHTML =
           `<h3>Y(jω)</h3><img src="data:image/png;base64,${d.plot_data}">`;
  });
};
/* ================================================================ */
/*  AJAX compute                                                    */
/* ================================================================ */
window.computeChain = ()=>{
  fetch("/process_chain/compute",{
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({
      input : document.getElementById("inputExpression").value,
      blocks: blocks,
      lines : lines
    })
  })
  .then(r=>r.json())
  .then(d=>{
    if(d.error) alert("Compute error: "+d.error);
    else document.getElementById("plotResult").innerHTML=
         `<img src="data:image/png;base64,${d.plot_data}">`;
  });
};

/* ================================================================ */
/*  Init fixed x(t) → y(t) nodes                                    */
/* ================================================================ */
addBlock("Input" ,"x(t)", 20, canvas.height/2-25,true);
addBlock("Output","y(t)", canvas.width-120, canvas.height/2-25,true);