"""V3 frozen-writer / two-port test: no writer updates and no label leakage.

The "active" probe is QUESTION-CONDITIONED, not feedback adaptive:
paths are committed before observing samples and do not react to
intermediate measurements. All phase transport is unitary, two sensor
ports are fixed, and each arm uses exactly K complex measurement slots.
Matched classifier parameter budget per K. Strong addressable local
phase/amplitude controls are allowed extra direct access to source
coordinates, explicitly disclosed.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from speedzones import SpeedZoneModel,phase_bytes
from probe_gate import ARMS,BUDGETS,STEPS,PORTS,dataset_all_questions,expand_questions,observed_features,path_bank


class Readout(nn.Module):
    def __init__(self,budget):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2*budget+4,48),nn.GELU(),
                               nn.Linear(48,2))
    def forward(self,x):
        return self.net(x)


def build_table(writer, scenes,device):
    img,labels=scenes
    parts=[]
    writer.eval()
    with torch.no_grad():
        for start in range(0,len(img),64):
            tau=writer(img[start:start+64].to(device)).cpu()
            # quantization roundtrip verifies actual 64-byte object
            assert torch.allclose(tau,((phase_bytes(tau).float()/255)*2-1)*1.2,atol=3e-7)
            parts.append(tau)
    tau=torch.cat(parts)
    memory,q,y=expand_questions(tau,labels)
    tables={}
    for budget in BUDGETS:
        with torch.no_grad():
            f=observed_features(memory,q,budget)
        qs=F.one_hot(q,4).float()
        tables[budget]={name:torch.cat([v,qs],dim=1) for name,v in f.items()}
    return tables,y


def train_heads(training,labels,val,val_labels,budget,seed,steps,batch_size):
    torch.manual_seed(seed+budget*101)
    nets=nn.ModuleDict({arm:Readout(budget) for arm in ARMS})
    opt=torch.optim.AdamW(nets.parameters(),lr=.004,weight_decay=.0005)
    x_train=training
    best_state=None
    best_val=float("inf")
    best_step=0
    for step in range(steps):
        idx=torch.randint(0,len(labels),(min(batch_size,len(labels)),))
        loss=sum(F.cross_entropy(net(x_train[name][idx]),labels[idx])
                 for name,net in nets.items())/len(nets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step%40==39 or step==steps-1:
            nets.eval()
            with torch.no_grad():
                vloss=sum(F.cross_entropy(net(val[name]),val_labels).item()
                          for name,net in nets.items())/len(nets)
            if vloss<best_val:
                best_val=vloss
                best_step=step+1
                best_state={name:{k:v.detach().clone() for k,v in net.state_dict().items()}
                            for name,net in nets.items()}
            nets.train()
    for name,state in best_state.items():
        nets[name].load_state_dict(state)
    return nets,best_step,best_val


@torch.no_grad()
def score_all(nets,table,labels):
    result={}
    for name in ARMS:
        logits=nets[name](table[name])
        predictions=logits.argmax(1)
        result[name]={
            "accuracy":float((predictions==labels).float().mean()),
            "errors":int((predictions!=labels).sum().item()),
            "cross_entropy":float(F.cross_entropy(logits,labels).item()),
        }
    return result


def run(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    checkpoint=torch.load(f"checkpoints/phase_wake_seed{args.seed}.pt",map_location=device,weights_only=True)
    if checkpoint.get("seed")!=args.seed:raise ValueError("Checkpoint mismatch")
    frozen=SpeedZoneModel().to(device).eval()
    frozen.load_state_dict(checkpoint["model"],strict=True)
    for p in frozen.parameters():p.requires_grad_(False)
    # Every quadrant is queried for every image; splits are scene-disjoint.
    # Same splits across checkpoint seeds for a paired model comparison.
    shared_seed=83610
    train=dataset_all_questions(args.train_scenes,shared_seed+1)
    val=dataset_all_questions(args.val_scenes,shared_seed+2)
    test=dataset_all_questions(args.test_scenes,shared_seed+3)
    train_f,train_y=build_table(frozen.writer,train,device)
    val_f,val_y=build_table(frozen.writer,val,device)
    test_f,test_y=build_table(frozen.writer,test,device)
    output={
        "protocol":"frozen-port-gate-v3-pilot-v1",
        "frozen_writer_checkpoint_seed":args.seed,
        "writer_frozen":True,
        "measurement_budget":"K=number of complex-valued scalar detector readings; paired arms spend 2 readings per tested path; padding is not a reading",
        "fixed_ports":[list(p) for p in PORTS],
        "phase_payload_bytes":64,
        "grid":[8,8],"propagation_steps_per_shot":STEPS,
        "train_scenes":args.train_scenes,"val_scenes":args.val_scenes,
        "test_scenes":args.test_scenes,
        "test_queries":int(len(test_y)),
        "split_seeds":[shared_seed+1,shared_seed+2,shared_seed+3],
        "trainable_writer_parameters":0,
        "results_by_budget":{},
        "falsification_rule":(
            "Claim a question-directed restricted-port advantage only if active "
            "probes beat BOTH fixed question-independent routes and equally "
            "budgeted addressable direct-phase probes on held-out queries "
            "on both frozen checkpoints. Order sensitivity requires a "
            "separate superiority versus same-budget reversed order and "
            "raw paired-order controls. Do not confuse target routing with "
            "a commutator information gain."),
        "limits":"Only static four-mark synthetic scenes; 2 fixed complex ports; preplanned question-dependent routes, not feedback-adaptive or PDE wavefront propagation. The direct phase and amplitude arms have stronger addressable access than fixed port arms."
    }
    for budget in BUDGETS:
        nets,selected_step,val_loss=train_heads(
            train_f[budget],train_y,val_f[budget],val_y,budget,
            seed=args.seed,steps=args.steps,batch_size=args.batch_size)
        output["results_by_budget"][str(budget)]={
            "readout_parameters_per_arm":sum(p.numel() for p in next(iter(nets.values())).parameters()),
            "selected_step":selected_step,"validation_joint_loss":val_loss,
            "validation":score_all(nets,val_f[budget],val_y),
            "test":score_all(nets,test_f[budget],test_y)
        }
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print("FROZEN_PORT_GATE_RESULTS_BEGIN")
    print(json.dumps(output,indent=2))
    print("FROZEN_PORT_GATE_RESULTS_END")
    return output


def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=(41,97),default=41)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--steps",type=int,default=480)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--train-scenes",type=int,default=1024)
    p.add_argument("--val-scenes",type=int,default=256)
    p.add_argument("--test-scenes",type=int,default=512)
    p.add_argument("--out",type=str,default="results/restricted_ports/seed41.json")
    return p.parse()


if __name__=="__main__":run(parse())
