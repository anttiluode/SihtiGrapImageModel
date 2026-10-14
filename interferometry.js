"use strict";
(async () => {
  const fields=[
    ["paired_orders","Raw paired orders"],
    ["interferometric_ratio","Conjugate ratio"],
    ["demodulated_difference","Demodulated subtraction"],
    ["order_difference","Uncorrected subtraction"],
    ["direct_matched_sources","Direct matched sources"],
    ["direct_local_phase","Direct local phase"],
    ["active_phase_then_shift","One-order wave"],
    ["fixed_routes","Fixed routes"]
  ];
  const root=document.getElementById("bars");
  const tbody=document.getElementById("rows");
  let data;
  try {
    const [a,b]=await Promise.all(["seed41","seed97"].map(s=>fetch("results/interferometry/"+s+".json",{cache:"no-store"})));
    if(!a.ok||!b.ok)throw Error("One or both result receipts are missing");
    data=await Promise.all([a.json(),b.json()]);
    if(data.some(d=>!d.writer_frozen||d.test_queries!==2048||
             !d.results_by_budget["2"].test.interferometric_ratio))
      throw Error("Unexpected result receipt schema");
  } catch(err) {root.textContent="Unable to load measurements: "+err.message;
    tbody.innerHTML="<tr><td colspan='4'>Read the raw JSON links below.</td></tr>";return}
  const show=(budget)=>{
    for(const b of document.querySelectorAll("[data-budget]"))
       b.setAttribute("aria-pressed",String(b.dataset.budget===budget));
    root.replaceChildren();tbody.replaceChildren();
    for(const [key,label] of fields){
      const values=data.map(d=>d.results_by_budget[budget].test[key].accuracy*100);
      const mean=(values[0]+values[1])/2;
      const bar=document.createElement("div");
      bar.className="bar-row"+(key==="interferometric_ratio"?"":" secondary");
      const a=document.createElement("span");a.className="bar-label";a.textContent=label;
      const frame=document.createElement("div");frame.className="bar-frame";
      const fill=document.createElement("div");fill.className="bar-fill";
      fill.style.width=Math.max(0,Math.min(100,(mean-50)*2)).toFixed(2)+"%";
      frame.appendChild(fill);
      const value=document.createElement("span");value.className="bar-num";value.textContent=mean.toFixed(2)+"%";
      bar.append(a,frame,value);root.appendChild(bar);
      const row=document.createElement("tr");
      for(const field of [label,...values.map(v=>v.toFixed(2)+"%"),mean.toFixed(2)+"%"]){
        const td=document.createElement("td");td.textContent=field;row.append(td);
      }tbody.appendChild(row);
    }
  };
  for(const btn of document.querySelectorAll("[data-budget]"))
    btn.addEventListener("click",()=>show(btn.dataset.budget));
  show("2");
})();