(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const clamp = (x, lo=0, hi=1) => Math.max(lo, Math.min(hi,x));
  let seed=31, uploaded=null;
  function rnd(seedValue){let s=seedValue>>>0;return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296}};
  function scene(){
    if(uploaded)return uploaded;
    const rand=rnd(seed), rgb=new Uint8ClampedArray(32*32*4), mode=$("choice").value;
    for(let y=0;y<32;y++)for(let x=0;x<32;x++){
      const k=(y*32+x)*4;
      let t=.25+rand()*.15;
      let color=[t,t,t];
      if(mode==="marked"){
        if(x>2&&x<14&&y>2&&y<29||x>18&&x<30&&y>2&&y<29)color=[.44,.44,.44];
        if(y>=13&&y<17&&x>=7&&x<11)color=[.95,.13,.13];
        if(y>=13&&y<17&&x>=22&&x<26)color=$("mark").value==="same"?[.95,.13,.13]:[.13,.18,.95];
      }
      if(mode==="ring"){
        const d=Math.hypot(x-15.5,y-15.5);
        color=(d>7&&d<12)?[.12,.19,.96]:[.48,.48,.48];
      }
      if(mode==="uniform") color=[.48,.48,.48];
      for(let c=0;c<3;c++)rgb[k+c]=Math.round(color[c]*255);
      rgb[k+3]=255;
    }
    return rgb;
  }
  const canvas=(id,image,width,height)=>{const c=$(id),ctx=c.getContext("2d");const d=ctx.createImageData(width,height);d.data.set(image);ctx.putImageData(d,0,0)};
  function scalarToImage(scalars,palette){
    let data=new Uint8ClampedArray(scalars.length*4);
    for(let i=0;i<scalars.length;i++){
      const col=palette(scalars[i]);
      data.set([...col,255],4*i);
    }
    return data;
  }
  function phaseColor(x){const t=clamp((x+1.2)/2.4);return[Math.round(35+210*t),Math.round(65+110*(1-Math.abs(2*t-1))),Math.round(45+210*(1-t))]};
  function heat(x){const v=clamp(x/1.8);return[Math.round(13+238*v),Math.round(30+178*v),Math.round(52+50*(1-v))]}
  function render(){
    $("gainOut").textContent=$("gain").value+"%";
    $("freqOut").textContent=(Number($("frequency").value)/100).toFixed(2)+"×";
    const rgb=scene();canvas("source",rgb,32,32);
    const tau=new Float64Array(64), gain=Number($("gain").value)/100, omega=Number($("frequency").value)/100;
    const rand=rnd(seed+919), mode=$("choice").value;
    for(let y=0;y<8;y++)for(let x=0;x<8;x++){
      let red=0,blue=0;
      for(let dy=0;dy<4;dy++)for(let dx=0;dx<4;dx++){
        const p=4*((y*4+dy)*32+(x*4+dx));
        red+=rgb[p]/255;blue+=rgb[p+2]/255;
      }
      let v=(red-blue)/16;
      if(mode==="noise"&&!uploaded)v=(rand()-.5)*2;
      if(mode==="uniform"&&!uploaded)v=.25;
      const analog=clamp(v*gain*1.2,-1.2,1.2);
      tau[y*8+x]=(Math.round((analog/1.2+1)*127.5)/255*2-1)*1.2;
    }
    canvas("phase",scalarToImage(tau,phaseColor),8,8);
    const real=new Float64Array(64),imag=new Float64Array(64);
    for(let i=0;i<64;i++){real[i]=Math.cos(tau[i]*omega);imag[i]=Math.sin(tau[i]*omega)}
    const fx=new Float64Array(64),fy=new Float64Array(64);
    for(let y=0;y<8;y++)for(let x=0;x<8;x++){
      const k=y*8+x,rx=y*8+(x+1)%8,ry=((y+1)%8)*8+x;
      fx[k]=Math.hypot(real[rx]-real[k],imag[rx]-imag[k]);
      fy[k]=Math.hypot(real[ry]-real[k],imag[ry]-imag[k]);
    }
    canvas("complex",scalarToImage(tau,phaseColor),8,8);
    canvas("wakeX",scalarToImage(fx,heat),8,8);
    canvas("wakeY",scalarToImage(fy,heat),8,8);
    canvas("blind",scalarToImage(new Float64Array(64).fill(1),v=>[Math.round(v*90),Math.round(v*90),Math.round(v*90)]),8,8);
    const rms=Math.sqrt((fx.reduce((a,b)=>a+b*b,0)+fy.reduce((a,b)=>a+b*b,0))/64);
    $("wakeValue").textContent=rms.toFixed(5);
    // Unit modulus and periodic shifts preserve L2 energy, even when tau
    // is varied spatially. This is a full numerical 20-step evolution.
    let ar=new Float64Array(64).fill(1/8),ai=new Float64Array(64);
    let dr=new Float64Array(64).fill(1/8),di=new Float64Array(64);
    for(let step=0;step<20;step++){
      const nr=new Float64Array(64),ni=new Float64Array(64),
            lr=new Float64Array(64),li=new Float64Array(64);
      for(let y=0;y<8;y++)for(let x=0;x<8;x++){
        const k=y*8+x,s=step%2===0?y*8+(x+1)%8:((y+1)%8)*8+x;
        const atten=.7+.15/(1+Math.exp(-tau[s]));
        nr[k]=ar[s]*real[s]-ai[s]*imag[s];
        ni[k]=ar[s]*imag[s]+ai[s]*real[s];
        lr[k]=dr[s]*atten;li[k]=di[s]*atten;
      }
      ar=nr;ai=ni;dr=lr;di=li;
    }
    let en=0,loss=0;
    for(let i=0;i<64;i++){en+=ar[i]**2+ai[i]**2;loss+=dr[i]**2+di[i]**2}
    $("unitaryValue").textContent=en.toFixed(7);
    $("lossyValue").textContent=loss.toFixed(7);
  }
  for(let el of ["choice","gain","frequency","mark"])$(el).addEventListener("input",()=>{uploaded=null;render()});
  $("reroll").addEventListener("click",()=>{seed++;uploaded=null;render()});
  $("file").addEventListener("change", e=>{
    const f=e.target.files?.[0];if(!f)return;
    const blob=URL.createObjectURL(f), img=new Image();
    img.onload=()=>{
      const c=document.createElement("canvas");c.width=32;c.height=32;
      const ctx=c.getContext("2d");ctx.drawImage(img,0,0,32,32);
      uploaded=ctx.getImageData(0,0,32,32).data;
      $("imageInfo").textContent="Local image: "+f.name+". Analytic color-difference encoder only.";
      URL.revokeObjectURL(blob);render();
    };
    img.onerror=()=>{URL.revokeObjectURL(blob);$("imageInfo").textContent="Could not read image."};
    img.src=blob;
  });
  render();
})();
