// Transparent client-side operator visualisation. No fitted weights are used here.
(() => {
  "use strict";
  const H = 14, W = 16, N = H * W;
  const $ = id => document.getElementById(id);
  let seed = 7, uploaded = null;
  const idx = (y, x) => y * W + x;
  const clamp = v => Math.max(0, Math.min(1, v));
  function rng(start) {
    let s = start >>> 0;
    return () => { s = (Math.imul(1664525, s) + 1013904223) >>> 0; return s / 4294967296; };
  }
  function scene() {
    const rand = rng(seed), color = [0, 1, 2].map(() => new Float32Array(N));
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const base = .14 + .16 * rand();
      for (let c = 0; c < 3; c++) color[c][idx(y,x)] = clamp(base + (rand()-.5)*.04);
    }
    for (let side = 0; side < 2; side++) {
      const x0 = side === 0 ? 1 : 9, shade = .36 + .21 * rand();
      for (let y = 2; y < 12; y++) for (let x = x0; x < x0 + 6; x++)
        for (let c = 0; c < 3; c++) color[c][idx(y,x)] = clamp(shade + (rand()-.5)*.09);
      const markY = 4 + Math.floor(rand() * 5);
      const markX = x0 + 1 + Math.floor(rand() * 3);
      const choice = $(side ? "colorRight" : "colorLeft").value;
      const rgb = choice === "red" ? [.96,.13,.13] : [.13,.18,.96];
      for (let y = markY; y < markY+2; y++) for (let x = markX; x < markX+2; x++)
        for (let c = 0; c < 3; c++) color[c][idx(y,x)] = rgb[c];
    }
    return color;
  }
  function pixel(color, y, x) {
    return [color[0][idx(y,x)], color[1][idx(y,x)], color[2][idx(y,x)]];
  }
  function etch(color) {
    const dirs = [[0,1],[1,0],[1,1],[1,-1]], beta=Number($("beta").value);
    const corruption=Number($("corrupt").value)/100, rand=rng(seed+81029);
    return dirs.map(([dy,dx]) => {
      const weights = new Float32Array(N);
      for(let y=0;y<H;y++) for(let x=0;x<W;x++){
        const yy=y+dy,xx=x+dx, p=idx(y,x);
        if(xx<0||xx>=W||yy<0||yy>=H) continue;
        const a=pixel(color,y,x),b=pixel(color,yy,xx);
        const distance=Math.sqrt(a.reduce((s,v,c)=>s+(v-b[c])**2,0)/3);
        let w=.06+.94*Math.exp(-beta*distance);
        w=clamp((1-corruption)*w+corruption*rand());
        weights[p]=Math.round(255*w)/255;
      }
      return weights;
    });
  }
  function question() {
    const a=[new Float32Array(N),new Float32Array(N)],
      right=$("side").value==="right", probe=$("probe").value;
    for(let y=0;y<H;y++)for(let x=0;x<W;x++) {
      const here=(x>=8)===right, i=idx(y,x);
      const m=probe==="global" ? 1 : probe==="impulse" ? ((x=== (right?11:4))&&(y===7)?1:0) : (here?1:0);
      a[0][i]=m;a[1][i]=m*(y/(H-1)*2-1);
    }
    return a;
  }
  function move(field,dy,dx) {
    const out= new Float32Array(N);
    for(let y=0;y<H;y++)for(let x=0;x<W;x++) {
      const yy=y+dy,xx=x+dx;
      if(yy>=0&&yy<H&&xx>=0&&xx<W)out[idx(y,x)]=field[idx(yy,xx)];
    }
    return out;
  }
  function op(w,q,axis) {
    const ds=axis==="A"?[[0,1,0],[2,1,1]]:[[1,1,0],[3,1,-1]];
    return q.map(field=>{
      const out=new Float32Array(N);
      const shifted=ds.map(([,dy,dx])=>move(field,dy,dx));
      for(let i=0;i<N;i++)out[i]=.5*(w[ds[0][0]][i]*shifted[0][i]+w[ds[1][0]][i]*shifted[1][i]);
      return out;
    });
  }
  function add(a,b,x=1,y=1) {
    return a.map((v,k)=>v.map((z,i)=>x*z+y*b[k][i]));
  }
  function scalar(field) {
    const out=new Float32Array(N);
    for(let i=0;i<N;i++)out[i]=field[0][i]+.35*field[1][i];
    return out;
  }
  function rms(fields) {
    let sum=0,n=0;
    fields.forEach(f=>f.forEach(v=>{sum+=v*v;n++}));
    return Math.sqrt(sum/n);
  }
  function draw(name,data,mode="mono") {
    const canvas=$(name), context=canvas.getContext("2d"), image=context.createImageData(W,H);
    let max=0;
    if(mode!=="rgb")for(const v of data)max=Math.max(max,Math.abs(v));
    max=Math.max(max,.000001);
    for(let i=0;i<N;i++){
      let rgb;
      if(mode==="rgb")rgb=data.map(f=>Math.round(255*clamp(f[i])));
      else if(mode==="difference"){
        const v=data[i]/max;
        rgb=v>0?[Math.round(25+230*v),Math.round(30+110*(1-v)),Math.round(48+70*(1-v))]:
                [Math.round(20+55*(1+v)),Math.round(40+150*(1+v)),Math.round(90+165*(-v))];
      } else {
        const v=Math.max(0,data[i]/max);
        rgb=[Math.round(9+80*v),Math.round(19+235*v),Math.round(33+186*v)];
      }
      image.data[4*i]=rgb[0];image.data[4*i+1]=rgb[1];
      image.data[4*i+2]=rgb[2];image.data[4*i+3]=255;
    }
    context.putImageData(image,0,0);
  }
  function update() {
    $("betaValue").textContent=$("beta").value;
    $("corruptValue").textContent=$("corrupt").value+"%";
    const c=uploaded||scene(), w=etch(c), q=question();
    const ab=op(w,op(w,q,"B"),"A"), ba=op(w,op(w,q,"A"),"B");
    const comm=add(ab,ba,1,-1),sym=add(ab,ba,.5,.5);
    draw("scene",c,"rgb");draw("edges",w[0]);
    draw("ab",scalar(ab));draw("ba",scalar(ba));
    draw("comm",scalar(comm),"difference");draw("sym",scalar(sym));
    const delta=rms(comm), avg=rms(sym), relative=delta/(avg+1e-9);
    $("commEnergy").textContent=delta.toFixed(4);
    $("symEnergy").textContent=avg.toFixed(4);
    $("ratio").textContent=relative.toFixed(3)+"×";
    $("commBar").style.width=Math.min(100,delta*900)+"%";
    $("symBar").style.width=Math.min(100,avg*900)+"%";
    $("ratioBar").style.width=Math.min(100,relative*100)+"%";
  }
  for (const id of ["colorLeft","colorRight","side","probe","beta","corrupt"]) $(id).addEventListener("input",update);
  $("newScene").addEventListener("click",()=>{
    seed++;uploaded=null;$("modeLabel").textContent="Synthetic scene #"+seed+" — all conductances recomputed locally.";update();
  });
  $("upload").addEventListener("change",e=>{
    const file=e.target.files&&e.target.files[0];if(!file)return;
    const src=URL.createObjectURL(file), img=new Image();
    img.onload=()=>{
      const c=document.createElement("canvas");c.width=W;c.height=H;
      const ctx=c.getContext("2d");ctx.drawImage(img,0,0,W,H);
      const raw=ctx.getImageData(0,0,W,H).data;
      uploaded=[0,1,2].map(k=>{
        const ch=new Float32Array(N);
        for(let i=0;i<N;i++)ch[i]=raw[4*i+k]/255;
        return ch;
      });
      $("modeLabel").textContent="Uploaded image: "+file.name+" (downsampled to 16×14 locally).";
      URL.revokeObjectURL(src);update();
    };
    img.onerror=()=>{URL.revokeObjectURL(src);$("modeLabel").textContent="Image could not be read.";};
    img.src=src;
  });
  update();
})();
