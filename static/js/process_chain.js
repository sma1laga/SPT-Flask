// static/js/process_chain.js

let blocks = [];   // store block objects
let lines = [];    // store connections
let blockIdCounter = 0;
let connectMode = false;
let connectStartId = null;
let selectedId = null;

const canvas = document.getElementById("chainCanvas");
const ctx = canvas.getContext("2d");

canvas.addEventListener("mousedown", onMouseDown);
// plus mousemove, mouseup for dragging if you want

function addBlock(type, label, x, y) {
  let bx = x || 100 + blocks.length*10;
  let by = y || 50 + blocks.length*10;
  let w = 100, h = 50;
  blocks.push({
    id: ++blockIdCounter,
    type: type,
    label: label,
    x: bx,
    y: by,
    width: w,
    height: h,
    param: null,   // store extra param user sets
    selected: false
  });
  drawAll();
}

function addDot() {
  // or store as a special block with type="Dot" & radius=5
}

function toggleConnectMode() {
  connectMode = !connectMode;
  if(!connectMode && connectStartId) {
    let startBlock = getBlockById(connectStartId);
    if(startBlock) startBlock.selected=false;
    connectStartId=null;
  }
  drawAll();
}

function deleteSelected() {
  if(!selectedId) return;
  // remove from blocks, remove lines referencing that block
  blocks = blocks.filter(b => b.id!=selectedId);
  lines = lines.filter(l => l.fromId!=selectedId && l.toId!=selectedId);
  selectedId=null;
  drawAll();
}

function clearAll() {
  // except maybe input & output if you want
  blocks = [];
  lines = [];
  selectedId=null;
  drawAll();
}

function drawAll() {
  ctx.clearRect(0,0,canvas.width,canvas.height);

  // draw lines first
  for(let l of lines) {
    let a = getBlockById(l.fromId);
    let b = getBlockById(l.toId);
    if(!a||!b) continue;
    let start = getConnectionPoint(a,b);
    let end = getConnectionPoint(b,a);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.strokeStyle="black";
    ctx.lineWidth=2;
    ctx.stroke();
  }

  // draw blocks
  for(let b of blocks) {
    ctx.beginPath();
    ctx.fillStyle= b.selected?"#ffd":"#ccf";
    ctx.fillRect(b.x,b.y,b.width,b.height);
    ctx.strokeStyle= b.selected?"orange":"black";
    ctx.lineWidth= b.selected?3:1;
    ctx.strokeRect(b.x,b.y,b.width,b.height);

    // text
    ctx.font="16px sans-serif";
    ctx.fillStyle="black";
    let textW = ctx.measureText(b.label).width;
    let tx = b.x + (b.width - textW)/2;
    let ty = b.y + b.height/2+5;
    ctx.fillText(b.label, tx, ty);
  }
}

function onMouseDown(e) {
  let x=e.offsetX, y=e.offsetY;
  // find a block if any
  let b = findBlockAt(x,y);
  if(connectMode) {
    if(!connectStartId && b) {
      connectStartId=b.id; b.selected=true;
    } else if(connectStartId && b && connectStartId!=b.id){
      lines.push({fromId:connectStartId, toId:b.id});
      let startBlock = getBlockById(connectStartId);
      if(startBlock) startBlock.selected=false;
      connectStartId=null;
    } else {
      // same or no block => stop connecting
      if(connectStartId){
        let st = getBlockById(connectStartId);
        if(st) st.selected=false;
      }
      connectStartId=null;
    }
  } else {
    // normal selection
    if(b){
      if(selectedId && selectedId!=b.id){
        let old = getBlockById(selectedId);
        if(old) old.selected=false;
      }
      b.selected=true;
      selectedId=b.id;
    } else {
      // clicked empty
      if(selectedId){
        let old = getBlockById(selectedId);
        if(old) old.selected=false;
      }
      selectedId=null;
    }
  }
  drawAll();
}

function getBlockById(id) {
  return blocks.find(b => b.id==id);
}

function findBlockAt(x,y) {
  // loop in reverse to get topmost
  for(let i=blocks.length-1; i>=0; i--){
    let b=blocks[i];
    if(x>=b.x && x<=b.x+b.width && y>=b.y && y<=b.y+b.height){
      return b;
    }
  }
  return null;
}

function getConnectionPoint(a,b){
  // if b.x < a.x => left side, else right side
  let ax=a.x+a.width/2, ay=a.y+a.height/2;
  let bx=b.x+b.width/2, by=b.y+b.height/2;
  if(bx < ax){
    return {x:a.x,y:a.y+a.height/2};
  } else {
    return {x:a.x+a.width,y:a.y+a.height/2};
  }
}

// block double-click to open param editor
canvas.addEventListener("dblclick", e=>{
  let mx=e.offsetX,my=e.offsetY;
  let b=findBlockAt(mx,my);
  if(b){
    openBlockModal(b);
  }
});

function openBlockModal(block){
  // show #blockModal, fill with block param
  const modal=document.getElementById("blockModal");
  modal.style.display="block";
  document.getElementById("blockTypeLabel").innerText="Editing: "+block.type;
  const input=document.getElementById("blockParamInput");
  input.value = block.param||"";
  blockInModal=block; // store globally
}
function closeBlockModal(){
  document.getElementById("blockModal").style.display="none";
}
let blockInModal=null;
function saveBlockParam(){
  let val=document.getElementById("blockParamInput").value.trim();
  if(blockInModal){
    blockInModal.param=val;
    blockInModal.label=blockInModal.type+"\n("+val+")";
  }
  closeBlockModal();
  drawAll();
}

// "Compute" example
function computeChain(){
  // gather chain data in an object, send to server via fetch/axios
  let chainData = {
    input: document.getElementById("inputExpression").value,
    blocks: blocks,
    lines: lines
  };
  // do an AJAX POST -> /process_chain/compute, get back base64
  fetch("/process_chain/compute", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(chainData)
  })
  .then(r=>r.json())
  .then(data=>{
    if(data.error){
      alert("Compute error: "+data.error);
    } else {
      let plotDiv=document.getElementById("plotResult");
      plotDiv.innerHTML='<img src="data:image/png;base64,'+data.plot_data+'" alt="Plot">';
    }
  })
  .catch(err=>console.error(err));
}
