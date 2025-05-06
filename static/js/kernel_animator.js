document.addEventListener('DOMContentLoaded', () => {
  /* ---------- canvas handles & contexts -------------- */
  const oCan = document.getElementById('orig'),
        hCan = document.getElementById('hilite'),
        rCan = document.getElementById('result'),
        kCan = document.getElementById('kernelCanvas'),

        gO   = oCan.getContext('2d'),
        gH   = hCan.getContext('2d'),
        gR   = rCan.getContext('2d'),
        gK   = kCan.getContext('2d'),

        W = oCan.width,
        H = oCan.height;

  /* ---------- demo image ----------------------------- */
  const img = new Image();
  img.src = DEMO_SRC;
  img.crossOrigin = 'anonymous';

  /* ---------- preset kernels ------------------------- */
  const kernels = {
    /* 3×3 kernels  ----------------------------------------------------------- */
    identity : { kw: [[0,0,0],[0,1,0],[0,0,0]],                       norm: 1 },
  
    /* average box‑blur */
    blur     : { kw: [[1,1,1],[1,1,1],[1,1,1]],                       norm: 9 },
  
    /* Gaussian blur (σ ≈ 0.85) */
    gauss    : { kw: [[1,2,1],[2,4,2],[1,2,1]],                       norm: 16 },
  
    sharpen  : { kw: [[0,-1,0],[-1,5,-1],[0,-1,0]],                  norm: 1 },
  
    /* Laplacian‑like edge detector */
    edge     : { kw: [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],              norm: 1 },
  
    /* emboss / height‑map */
    emboss   : { kw: [[-2,-1,0],[-1,1,1],[0,1,2]],                   norm: 1 },
  
    /* simple outline (zero‑sum) */
    outline  : { kw: [[1,1,1],[1,-8,1],[1,1,1]],              norm: 1 }
  };
  let kernel = kernels.blur;

  /* ---------- state ---------------------------------- */
  let x = 0, y = 0, playing = false, rafID = null;

  /* ---------- diverging colour util ------------------ */
  function diverge(v){
    const t = (v + 1) / 2;               // 0–1
    const h = 240 - 240 * t;             // 240°→0°
    return `hsl(${h},85%,${70 - 20 * Math.abs(v)}%)`;
  }

  /* ---------- kernel UI ------------------------------ */
  function drawKernelGrid(){
    const ui  = document.getElementById('kernel-ui');
    ui.innerHTML = '';
    const flat = kernel.kw.flat();
    const max  = Math.max(...flat.map(Math.abs)) || 1;

    flat.forEach(v=>{
      const c = document.createElement('div');
      c.className = 'kernel-cell';
      c.textContent = v;
      c.style.background = diverge(v / max);
      ui.appendChild(c);
    });
    drawKernelHeatmap();
  }

  /* kernel heatmap (90×90) ---------------------------- */
  function drawKernelHeatmap(){
    const flat = kernel.kw.flat();
    const max  = Math.max(...flat.map(Math.abs)) || 1;
    const imgD = gK.createImageData(90,90);

    flat.forEach((v,i)=>{
      const col = diverge(v / max);
      /* tiny trick to get RGB from HSL */
      const tmp = document.createElement('canvas').getContext('2d');
      tmp.fillStyle = col; tmp.fillRect(0,0,1,1);
      const [r,g,b] = tmp.getImageData(0,0,1,1).data;

      const bx = (i%3)*30, by = (i/3|0)*30;
      for(let yy=0; yy<30; yy++){
        for(let xx=0; xx<30; xx++){
          const o = ((by+yy)*90+(bx+xx))*4;
          imgD.data[o]=r; imgD.data[o+1]=g; imgD.data[o+2]=b; imgD.data[o+3]=255;
        }
      }
    });
    gK.putImageData(imgD,0,0);
  }

  /* ---------- highlight box on overlay --------------- */
  function highlight(px,py){
    gH.clearRect(0,0,W,H);
    gH.strokeStyle = 'yellow';
    gH.lineWidth   = 2;
    gH.strokeRect(px,py,3,3);
  }

  /* ---------- convolution step ----------------------- */
  function convolveStep(){
    const patch = gO.getImageData(x,y,3,3).data;
    let rSum=0,gSum=0,bSum=0;

    kernel.kw.flat().forEach((kv,i)=>{
      rSum += kv * patch[i*4  ];
      gSum += kv * patch[i*4+1];
      bSum += kv * patch[i*4+2];
    });

    if(kernel.norm){
      rSum/=kernel.norm; gSum/=kernel.norm; bSum/=kernel.norm;
    }

    const r = Math.min(255,Math.max(0,Math.round(rSum)));
    const g = Math.min(255,Math.max(0,Math.round(gSum)));
    const b = Math.min(255,Math.max(0,Math.round(bSum)));

    gR.fillStyle = `rgb(${r},${g},${b})`;
    gR.fillRect(x,y,1,1);
    document.getElementById('sum-val').textContent = ((r+g+b)/3|0);

    if(++x > W-3){ x=0; y++; }
    if(y > H-3) stop();
  }

  /* ---------- fast animation loop -------------------- */
  const sel   = document.getElementById('kernel-select');
  const playB = document.getElementById('play');
  const stepB = document.getElementById('step');
  const speed = document.getElementById('speed');   // patches / frame

  function frameLoop(){
    const n = +speed.value;
    for(let i=0;i<n && y<=H-3;i++) convolveStep();
    highlight(x,y);                                // once per frame
    if(playing) rafID = requestAnimationFrame(frameLoop);
  }

  function start(){
    if(playing) return;
    playing = true;
    playB.textContent = 'Pause ❚❚';
    rafID = requestAnimationFrame(frameLoop);
  }
  function stop(){
    playing = false;
    cancelAnimationFrame(rafID);
    playB.textContent = 'Play ▶';
  }
  function stepOnce(){ if(!playing && y<=H-3) { convolveStep(); highlight(x,y);} }

  /* ---------- image loaded --------------------------- */
  img.onload = ()=>{
    gO.drawImage(img,0,0,W,H);
    drawKernelGrid();
    gR.clearRect(0,0,W,H);
    highlight(0,0);

    sel.onchange = e => { kernel = kernels[e.target.value]; reset(); };
    playB.onclick = ()=> playing ? stop() : start();
    stepB.onclick = stepOnce;
  };

  function reset(){
    stop();
    x=y=0;
    gR.clearRect(0,0,W,H);
    gO.drawImage(img,0,0,W,H);
    drawKernelGrid();
    highlight(0,0);
  }
});
