// Global variables
let blocks = [];   // Array to store block objects.
let lines = [];    // Array to store connection objects.
let blockIdCounter = 0;
let connectMode = false;
let connectStartId = null;
let selectedId = null;
let draggingBlock = null;
let dragOffsetX = 0;
let dragOffsetY = 0;
let blockInModal = null;        // For generic block modal.
let blockInFilterModal = null;  // For the filter modal.
let blockInMultModal = null;    // For Multiplication blocks


// References
const canvas = document.getElementById("chainCanvas");
const ctx = canvas.getContext("2d");

// Utility: Get block by id.
function getBlockById(id) {
  return blocks.find(b => b.id == id);
}

// Utility: Find block under mouse coordinates.
function findBlockAt(x, y) {
  for (let i = blocks.length - 1; i >= 0; i--) {
    let b = blocks[i];
    if (x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height) {
      return b;
    }
  }
  return null;
}

// Utility: Determine a connection point on a block for drawing lines.
function getConnectionPoint(a, b) {
  let ax = a.x + a.width / 2, ay = a.y + a.height / 2;
  let bx = b.x + b.width / 2, by = b.y + b.height / 2;
  if (bx < ax) {
    return { x: a.x, y: a.y + a.height / 2 };
  } else {
    return { x: a.x + a.width, y: a.y + a.height / 2 };
  }
}

// Draw all: clears canvas then draws lines and blocks.
function drawAll() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Draw connections.
  for (let l of lines) {
    let a = getBlockById(l.fromId);
    let b = getBlockById(l.toId);
    if (!a || !b) continue;
    let start = getConnectionPoint(a, b);
    let end = getConnectionPoint(b, a);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = "black";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  // Draw blocks.
  for (let b of blocks) {
    ctx.beginPath();
    ctx.fillStyle = b.nonDeletable ? "#cfc" : "#ccf";
    ctx.fillRect(b.x, b.y, b.width, b.height);
    if (b.selected) {
      ctx.strokeStyle = "orange";
      ctx.lineWidth = 3;
    } else if (b.nonDeletable) {
      ctx.strokeStyle = "green";
      ctx.lineWidth = 2;
    } else {
      ctx.strokeStyle = "black";
      ctx.lineWidth = 1;
    }
    ctx.strokeRect(b.x, b.y, b.width, b.height);
    ctx.font = "16px sans-serif";
    ctx.fillStyle = "black";
    let linesText = b.label.split("\n");
    linesText.forEach((line, index) => {
      let textWidth = ctx.measureText(line).width;
      let tx = b.x + (b.width - textWidth) / 2;
      let ty = b.y + b.height / 2 + 5 + index * 18 - ((linesText.length - 1) * 9);
      ctx.fillText(line, tx, ty);
    });
  }
}

// Adding a new block.
function addBlock(type, label, x, y, nonDeletable = false) {
  let bx = x !== undefined ? x : 150 + blocks.length * 10;
  let by = y !== undefined ? y : 50 + blocks.length * 10;
  let w = 100, h = 50;
  blocks.push({
    id: ++blockIdCounter,
    type: type,
    label: label,
    x: bx,
    y: by,
    width: w,
    height: h,
    param: null,        // Additional parameter.
    selected: false,
    nonDeletable: nonDeletable
  });
  drawAll();
}

// Toggle connect mode.
function toggleConnectMode() {
  connectMode = !connectMode;
  if (!connectMode && connectStartId) {
    let startBlock = getBlockById(connectStartId);
    if (startBlock) startBlock.selected = false;
    connectStartId = null;
  }
  drawAll();
}

// Delete selected block (if allowed).
function deleteSelected() {
  if (!selectedId) return;
  let b = getBlockById(selectedId);
  if (b && b.nonDeletable) return;
  lines = lines.filter(l => l.fromId !== selectedId && l.toId !== selectedId);
  blocks = blocks.filter(b => b.id !== selectedId);
  selectedId = null;
  drawAll();
}

// Clear all user-added blocks and connections (retain non-deletable blocks).
function clearAll() {
  blocks = blocks.filter(b => b.nonDeletable);
  lines = [];
  selectedId = null;
  drawAll();
}

// Mouse event handlers.
canvas.addEventListener("mousedown", (e) => {
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let b = findBlockAt(x, y);
  if (connectMode) {
    if (!connectStartId && b) {
      connectStartId = b.id;
      b.selected = true;
    } else if (connectStartId && b && connectStartId !== b.id) {
      lines.push({ fromId: connectStartId, toId: b.id });
      let startBlock = getBlockById(connectStartId);
      if (startBlock) startBlock.selected = false;
      connectStartId = null;
    } else {
      if (connectStartId) {
        let st = getBlockById(connectStartId);
        if (st) st.selected = false;
      }
      connectStartId = null;
    }
    drawAll();
    return;
  } else {
    if (b) {
      draggingBlock = b;
      dragOffsetX = x - b.x;
      dragOffsetY = y - b.y;
      if (selectedId && selectedId !== b.id) {
        let old = getBlockById(selectedId);
        if (old) old.selected = false;
      }
      b.selected = true;
      selectedId = b.id;
    } else {
      if (selectedId) {
        let old = getBlockById(selectedId);
        if (old) old.selected = false;
      }
      selectedId = null;
    }
    drawAll();
  }
});

canvas.addEventListener("mousemove", (e) => {
  if (draggingBlock) {
    let rect = canvas.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    draggingBlock.x = x - dragOffsetX;
    draggingBlock.y = y - dragOffsetY;
    drawAll();
  }
});

canvas.addEventListener("mouseup", () => {
  draggingBlock = null;
});

// Double-click: Open modal editor.
// For Filter blocks, open the dedicated filter modal.
canvas.addEventListener("dblclick", (e) => {
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let b = findBlockAt(x, y);
  if (b) {
    openBlockModal(b);
  }
});

// Standard block modal for non-filter blocks.
function openBlockModal(block) {
  if (block.type === "Filter") {
    openFilterModal(block);
  } else if (block.type === "Multiplication") {
    openMultiplicationModal(block);
  } else {
    // For all other types, use the standard modal.
    const modal = document.getElementById("blockModal");
    modal.style.display = "block";
    document.getElementById("blockTypeLabel").innerText = "Editing: " + block.type;
    const input = document.getElementById("blockParamInput");
    input.value = block.param || "";
    blockInModal = block;
  }
}

// --- New function: openMultiplicationModal ---
function openMultiplicationModal(block) {
  const modal = document.getElementById("multiplicationModal");
  modal.style.display = "block";
  blockInMultModal = block;
  // Initialize the dropdown and input fields.
  if (block.param) {
    // Expect the parameter string in format: "type:value"
    let parts = block.param.split(":");
    let multType = parts[0].trim();
    document.getElementById("multType").value = multType;
    if (multType === "sampling") {
      document.getElementById("samplingDiv").style.display = "block";
      document.getElementById("multParamDiv").style.display = "none";
      document.getElementById("samplingInterval").value = parts[1] ? parseFloat(parts[1]) : 1.0;
    } else if (multType === "imaginary" || multType === "sin" || multType === "cos") {
      document.getElementById("multParamDiv").style.display = "none";
      document.getElementById("samplingDiv").style.display = "none";
    } else {
      // For constant, linear, exponential, etc.
      document.getElementById("multParamDiv").style.display = "block";
      document.getElementById("samplingDiv").style.display = "none";
      document.getElementById("multParamValue").value = parts[1] ? parts[1].trim() : "";
    }
  } else {
    // Default settings if no parameter is set.
    document.getElementById("multType").value = "constant";
    document.getElementById("multParamDiv").style.display = "block";
    document.getElementById("samplingDiv").style.display = "none";
    document.getElementById("multParamValue").value = "";
  }
}

// Listen for changes in the multiplication type dropdown.
document.getElementById("multType").addEventListener("change", function() {
  const type = this.value;
  if (type === "sampling") {
    document.getElementById("samplingDiv").style.display = "block";
    document.getElementById("multParamDiv").style.display = "none";
  } else if (type === "imaginary" || type === "sin" || type === "cos") {
    document.getElementById("multParamDiv").style.display = "none";
    document.getElementById("samplingDiv").style.display = "none";
  } else {
    document.getElementById("multParamDiv").style.display = "block";
    document.getElementById("samplingDiv").style.display = "none";
  }
});

// Handler for Multiplication Modal OK button.
document.getElementById("multModalOk").addEventListener("click", function() {
  if (blockInMultModal) {
    const type = document.getElementById("multType").value;
    let paramStr = "";
    if (type === "sampling") {
      const interval = document.getElementById("samplingInterval").value;
      paramStr = `${type}:${interval}`;
    } else if (type === "imaginary" || type === "sin" || type === "cos") {
      paramStr = type;  // No additional parameter.
    } else {
      const value = document.getElementById("multParamValue").value;
      paramStr = `${type}:${value}`;
    }
    blockInMultModal.param = paramStr;
    blockInMultModal.label = "×\n(" + paramStr + ")";
  }
  closeMultiplicationModal();
  drawAll();
});

// Handler for Multiplication Modal Cancel button.
document.getElementById("multModalCancel").addEventListener("click", function() {
  closeMultiplicationModal();
});

// Function to close the multiplication modal.
function closeMultiplicationModal() {
  document.getElementById("multiplicationModal").style.display = "none";
  blockInMultModal = null;
}

// Existing event handlers (mousedown, mousemove, mouseup, etc.) remain unchanged.
// Also, ensure that your double-click event on the canvas calls openBlockModal.

canvas.addEventListener("dblclick", (e) => {
  let rect = canvas.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  let b = findBlockAt(x, y);
  if (b) {
    openBlockModal(b);
  }
});

// Expose any other global functions for UI buttons as before.
window.addMultiplication = function() {
  // Instead of directly prompting, we add a multiplication block.
  addBlock("Multiplication", "×", undefined, undefined, false);
};

// === New functions for Filter Modal ===
function openFilterModal(block) {
  const filterModal = document.getElementById("filterModal");
  filterModal.style.display = "block";
  blockInFilterModal = block;
  // Initialize modal fields from block.param if available.
  if (block.param) {
    const parts = block.param.split(":");
    if (parts.length === 2) {
      const type = parts[0].trim();
      document.getElementById("filterType").value = type;
      if (type === "bandpass") {
        document.getElementById("bandCutoffs").style.display = "block";
        document.getElementById("singleCutoff").style.display = "none";
        const freqs = parts[1].split(",");
        if (freqs.length === 2) {
          document.getElementById("lowCutoff").value = parseFloat(freqs[0]);
          document.getElementById("highCutoff").value = parseFloat(freqs[1]);
        }
      } else {
        document.getElementById("singleCutoff").style.display = "block";
        document.getElementById("bandCutoffs").style.display = "none";
        document.getElementById("cutoffValue").value = parseFloat(parts[1]);
      }
    }
  } else {
    document.getElementById("filterType").value = "lowpass";
    document.getElementById("singleCutoff").style.display = "block";
    document.getElementById("bandCutoffs").style.display = "none";
    document.getElementById("cutoffValue").value = 1.0;
  }
}

// Listen for changes in the filter type dropdown.
document.getElementById("filterType").addEventListener("change", function() {
  const type = this.value;
  if (type === "bandpass") {
    document.getElementById("bandCutoffs").style.display = "block";
    document.getElementById("singleCutoff").style.display = "none";
  } else {
    document.getElementById("bandCutoffs").style.display = "none";
    document.getElementById("singleCutoff").style.display = "block";
  }
});

document.getElementById("filterModalOk").addEventListener("click", function() {
  if (blockInFilterModal) {
    const type = document.getElementById("filterType").value;
    let paramStr = "";
    if (type === "bandpass") {
      const low = document.getElementById("lowCutoff").value;
      const high = document.getElementById("highCutoff").value;
      paramStr = `${type}:${low},${high}`;
    } else {
      const cutoff = document.getElementById("cutoffValue").value;
      paramStr = `${type}:${cutoff}`;
    }
    blockInFilterModal.param = paramStr;
    blockInFilterModal.label = "Filter\n(" + paramStr + ")";
  }
  closeFilterModal();
  drawAll();
});

document.getElementById("filterModalCancel").addEventListener("click", function() {
  closeFilterModal();
});

function closeFilterModal() {
  document.getElementById("filterModal").style.display = "none";
  blockInFilterModal = null;
}
// === End of Filter Modal Functions ===

// Expose UI functions globally.
window.addAddition = function() { addBlock("Addition", "+"); };
window.addSubtraction = function() { addBlock("Subtraction", "−"); };
window.addMultiplication = function() { 
  let expr = prompt("Enter multiplier expression:", "cos(ω₀w)");
  let label = "×";
  if(expr) {
    label += "\n(" + expr + ")";
  }
  addBlock("Multiplication", label);
};
window.addGenericBlock = function() { addBlock("Block", "Block"); };
window.addDot = function() { /* Extend to add dots if needed */ };
window.toggleConnectMode = toggleConnectMode;
window.deleteSelected = deleteSelected;
window.clearAll = clearAll;
window.computeChain = function() {
  let chainData = {
    input: document.getElementById("inputExpression").value,
    blocks: blocks,
    lines: lines
  };
  fetch("/process_chain/compute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(chainData)
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) {
      alert("Compute error: " + data.error);
    } else {
      let plotDiv = document.getElementById("plotResult");
      plotDiv.innerHTML = '<img src="data:image/png;base64,' + data.plot_data + '" alt="Plot">';
    }
  })
  .catch(err => console.error(err));
};

// Initialize chain with fixed Input and Output blocks.
function initChain() {
  addBlock("Input", "x(t)", 20, canvas.height / 2 - 25, true);
  addBlock("Output", "y(t)", canvas.width - 120, canvas.height / 2 - 25, true);
}
initChain();
