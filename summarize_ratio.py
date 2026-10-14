"""Print the two-writer interferometry follow-up results and identity errors."""
import json
from pathlib import Path
folder=Path("results/interferometry")
data=[json.loads((folder/f"seed{s}.json").read_text()) for s in (41,97)]
arms=("paired_orders","order_difference","interferometric_ratio",
      "demodulated_difference","direct_local_phase",
      "active_phase_then_shift","active_shift_then_phase","fixed_routes")
print("INTERFEROMETRY V3B RECEIPTS / FROZEN 64-BYTE WRITERS")
for x in data:
    a=x["identity_verification"]
    print("seed",x["frozen_writer_checkpoint_seed"],
          "max endpoint ratio numerical error",a["max_abs_ratio_vs_endpoints_error"],
          "port phase std",a["port_phase_std_across_queries"])
    assert a["max_abs_ratio_vs_endpoints_error"]<1e-5
for k in ("2","4","8"):
    print("PHYSICAL SHOTS",k)
    for arm in arms:
        a,b=[r["results_by_budget"][k]["test"][arm] for r in data]
        print(f" {arm:30s} seed41={a['accuracy']:.5f} seed97={b['accuracy']:.5f} avg={(a['accuracy']+b['accuracy'])/2:.5f}")
    for i,row in enumerate(data):
        x=row["results_by_budget"][k]["test"]
        print("seed",row["frozen_writer_checkpoint_seed"],
              "ratio_vs_subtract_delta",x["interferometric_ratio"]["accuracy"]-x["order_difference"]["accuracy"],
              "ratio_vs_pair_delta",x["interferometric_ratio"]["accuracy"]-x["paired_orders"]["accuracy"],
              "ratio_vs_direct_delta",x["interferometric_ratio"]["accuracy"]-x["direct_local_phase"]["accuracy"])
