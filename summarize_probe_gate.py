"""Summarize both real trained frozen-writer runs. Do not hide failures."""
import json
from pathlib import Path

path=Path("results/restricted_ports")
a=json.loads((path/"seed41.json").read_text())
b=json.loads((path/"seed97.json").read_text())
assert a["writer_frozen"] and b["writer_frozen"]
assert a["test_queries"]==b["test_queries"]
arms=("active_phase_then_shift","active_shift_then_phase","paired_orders",
      "order_difference","fixed_routes","direct_local_phase",
      "direct_local_amplitude","passive_ports")
print("PRE-REGISTERED FROZEN-WRITER PORT GATE")
print("2 ports",a["fixed_ports"],"12 phase-shift steps/shot; K physical complex measurements")
for k in ("2","4","8"):
    print("SHOTS",k)
    for arm in arms:
        v=[z["results_by_budget"][k]["test"][arm]["accuracy"] for z in (a,b)]
        print("  %-27s  %6.2f%% %6.2f%%   mean %6.2f%%"%(arm,100*v[0],100*v[1],100*sum(v)/2))
    print("ACTIVE WINS BOTH BASELINES PER SEED:",[
        z["results_by_budget"][k]["test"]["active_phase_then_shift"]["accuracy"] >
        max(z["results_by_budget"][k]["test"]["fixed_routes"]["accuracy"],
            z["results_by_budget"][k]["test"]["direct_local_phase"]["accuracy"])
        for z in (a,b)])
