"use strict";
(async function(){
 const arms=[
 ["direct_local_phase","Direct local phase",false],
 ["active_phase_then_shift","Question-directed wave",true],
 ["active_shift_then_phase","Reversed order",false],
 ["paired_orders","Both orders",false],
 ["order_difference","Order difference",false],
 ["fixed_routes","Question-independent wave",false],
 ["direct_local_amplitude","Direct local amplitude",false],
 ["passive_ports","Passive fixed ports",false]
 ];
 const plot=document.getElementById("plot");
 const table=document.getElementById("scoreRows");
 const allButtons=Array.from(document.querySelectorAll("[data-k]"));
 let records;
 try{
  const response=await Promise.all(["seed41","seed97"].map(id=>fetch("results/restricted_ports/"+id+".json",{cache:"no-cache"})));
  if(response.some(r=>!r.ok))throw new Error("Results are not available at this URL.");
  records=await Promise.all(response.map(r=>r.json()));
  if(records.length!==2||records.some(r=>r.writer_frozen!==true||r.test_queries!==2048))throw new Error("Unexpected receipt data.");
 }catch(error){
  plot.textContent="Could not load the measurements: "+error.message;
  table.innerHTML='<tr><td colspan="4">See linked raw JSON receipts.</td></tr>';
  return;
 }
 function show(k){
  for(const b of allButtons)b.setAttribute("aria-pressed",String(b.dataset.k===k));
  plot.replaceChildren();table.replaceChildren();
  for(const [key,title,active] of arms){
   const scores=records.map(rec=>rec.results_by_budget[k].test[key].accuracy*100);
   const mean=(scores[0]+scores[1])/2;
   const row=document.createElement("div");row.className="plot-row";
   const name=document.createElement("small");name.textContent=title;
   const bar=document.createElement("div");bar.className="bar";
   const fill=document.createElement("div");fill.className=active?"":"alt";
   fill.style.width=Math.max(0,Math.min(100,(mean-50)*2)).toFixed(2)+"%";
   bar.append(fill);
   const score=document.createElement("span");score.className="score";score.textContent=mean.toFixed(2)+"%";
   row.append(name,bar,score);plot.append(row);
   const tr=document.createElement("tr");
   for(const value of [title,scores[0].toFixed(2)+"%",scores[1].toFixed(2)+"%",mean.toFixed(2)+"%"]){
    const td=document.createElement("td");td.textContent=value;tr.append(td);
   }
   table.append(tr);
  }
 }
 for(const b of allButtons)b.addEventListener("click",()=>show(b.dataset.k));
 show("4");
})();
