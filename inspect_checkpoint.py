"""Inspect saved v2 model weights: real 64-byte maps, controlled single-mark flips.

This does not train, alter checkpoints, or test a new architecture.
It asks whether existing weights localized task information, and whether
a counterfactual mark change changes only the correct queried answer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from speedzones import SpeedZoneModel, phase_bytes, commutator_fields, MAX_PHASE
from train_speedzones import make_examples

RED = (.96, .08, .08)
BLUE = (.08, .13, .96)
SELECT_ARMS = ("commutator", "ordered_pair", "phase_gradient",
               "coherent_mean", "direct_phase", "amplitude_code")


def flip_one_mark(images: torch.Tensor, quadrant: int):
    """Exact controlled intervention: change just one 2x2 mark color."""
    assert images.ndim == 4 and images.shape[1:] == (3,32,32)
    y0,x0 = (quadrant//2)*16,(quadrant%2)*16
    red = (images[:,0]>.85)&(images[:,2]<.2)
    blue = (images[:,2]>.85)&(images[:,0]<.2)
    allowed = torch.zeros_like(red)
    allowed[:, y0:y0+16, x0:x0+16] = True
    marker_red, marker_blue = red & allowed, blue & allowed
    found = (marker_red | marker_blue).flatten(1).sum(1)
    if not torch.all(found == 4):
        raise ValueError(f"Expected exactly four colored mark pixels in quadrant {quadrant}, found {found.tolist()}")
    old_class = marker_blue.flatten(1).any(1).to(torch.long)
    changed = images.clone()
    mask = marker_red | marker_blue
    for channel in range(3):
        replacement = torch.where(marker_red, BLUE[channel], RED[channel])
        changed[:,channel] = torch.where(mask, replacement, images[:,channel])
    n_changed = (changed-images).abs().amax(1).gt(0).flatten(1).sum(1)
    assert torch.all(n_changed == 4)
    return changed, old_class, 1-old_class


@torch.no_grad()
def predict(net, tau, quadrant):
    q = torch.full((len(tau),), quadrant, dtype=torch.long)
    return {arm: logits.argmax(1).cpu()
            for arm, logits in net.read(tau, q).items() if arm in SELECT_ARMS}


def heatmap_html(values, scheme="tau", size=31):
    # Generated small rectangles; SVG is not a reconstruction of a wave.
    if scheme == "tau":
        def rgb(v):
            c=max(0.,min(1.,(v/MAX_PHASE+1)/2))
            return f"rgb({int(30+225*c)},{int(50+120*(1-abs(2*c-1)))},{int(40+210*(1-c))})"
    elif scheme == "delta":
        peak=max(max(row) for row in values) or 1.
        def rgb(v):
            c=max(0,min(1,v/peak))
            return f"rgb({int(12+240*c)},{int(25+175*c)},{int(43+30*(1-c))})"
    else:
        peak=max(max(row) for row in values) or 1.
        def rgb(v):
            c=max(0,min(1,v/peak))
            return f"rgb({int(15+155*c)},{int(40+210*c)},{int(65+130*c)})"
    rectangles=[]
    for y,row in enumerate(values):
        for x,value in enumerate(row):
            rectangles.append(f'<rect x="{x*size}" y="{y*size}" width="{size}" height="{size}" fill="{rgb(float(value))}"><title>{float(value):.4f}</title></rect>')
    side=size*8
    return f'<svg role="img" viewBox="0 0 {side} {side}" xmlns="http://www.w3.org/2000/svg">{"".join(rectangles)}</svg>'


@torch.no_grad()
def inspect_model(path, images, seed, outdir):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    net=SpeedZoneModel().cpu().eval()
    net.load_state_dict(raw["model"], strict=True)
    assert raw.get("seed")==seed
    tau=net.write(images)
    codes=phase_bytes(tau).float()
    n=len(images)
    flat=codes.flatten()
    hist=np.bincount(flat.to(torch.uint8).cpu().numpy().astype(np.int64), minlength=256)
    pmf=hist[hist>0]/sum(hist)
    entropy=float(-(pmf*np.log2(pmf)).sum())
    std_cell=codes[:,0].std(dim=0,unbiased=False)
    saturation=( (flat<=1) | (flat>=254) ).float().mean().item()
    report={
        "seed":seed,"checkpoint":str(path),"sample_count":n,
        "bytes_per_scene":64,"task":"four independently colored 2x2 marks in 32x32 scenes",
        "phase_statistics":{
            "min_rad":float(tau.min()),"max_rad":float(tau.max()),
            "mean_rad":float(tau.mean()),"std_rad":float(tau.std(unbiased=False)),
            "unique_byte_values_across_all_scenes":int((hist>0).sum()),
            "pooled_byte_entropy_bits":entropy,
            "fraction_bytes_near_0_or_255":float(saturation),
            "mean_between_scene_byte_std_per_cell":float(std_cell.mean()),
            "max_between_scene_byte_std_per_cell":float(std_cell.max())
        },
        "counterfactual_per_quadrant":{},
        "example":{},
        "notes":[
            "An 8x8 field derived by local convolutions has spatial receptive-field restrictions; localization by itself is not proof of a learned geometric principle.",
            "The phase bytes are a supervised visual code, not calibrated physical delays or material speeds.",
            "Original and perturbed examples are identical except for four RGB pixels in one mark.",
            "No data used in these inspections were used for training (fixed new common draw seed 20261014)."
        ]
    }
    records={}
    for quad in range(4):
        perturbed,original_label,new_label=flip_one_mark(images,quad)
        counter=net.write(perturbed)
        code_new=phase_bytes(counter).float()
        absdiff=(code_new-codes).abs()[:,0]
        region=torch.zeros(8,8,dtype=torch.bool)
        y0,x0=(quad//2)*4,(quad%2)*4
        region[y0:y0+4,x0:x0+4]=True
        inside=absdiff[:,region].sum(dim=1)
        outside=absdiff[:,~region].sum(dim=1)
        total=inside+outside
        fraction=torch.where(total>0,inside/total,torch.zeros_like(total))
        base_pred=predict(net,tau,quad)
        flip_pred=predict(net,counter,quad)
        accuracy_original={a:float((p==original_label).float().mean()) for a,p in base_pred.items()}
        accuracy_flipped={a:float((p==new_label).float().mean()) for a,p in flip_pred.items()}
        # Check whether asking about other quadrants after the flip retains
        # their original answer predictions; not a full causal independence proof.
        other_kept={a:[] for a in SELECT_ARMS}
        for q in range(4):
            if q==quad:continue
            before=predict(net,tau,q); after=predict(net,counter,q)
            for a in SELECT_ARMS:other_kept[a].append(float((before[a]==after[a]).float().mean()))
        other_kept={a:float(np.mean(v)) for a,v in other_kept.items()}
        report["counterfactual_per_quadrant"][str(quad)]={
            "source_changed_pixels":4,
            "mean_changed_phase_bytes":float((absdiff>0).flatten(1).sum(1).float().mean()),
            "mean_absolute_byte_delta":float(absdiff.mean()),
            "fraction_change_inside_target_quadrant_mean":float(fraction.mean()),
            "fraction_change_outside_target_quadrant_mean":float(torch.where(total>0,outside/total,torch.zeros_like(total)).mean()),
            "number_of_zero_effect_samples":int((total==0).sum()),
            "original_accuracy_by_reader":accuracy_original,
            "flipped_mark_accuracy_by_reader":accuracy_flipped,
            "unchanged_other_question_predictions_by_reader":other_kept
        }
        if quad in (0,3):
            first=0
            wake_x,wake_y=commutator_fields(tau[first:first+1])
            gradient=(wake_x.abs().square()+wake_y.abs().square()).sqrt()[0,0]
            report["example"][f"quadrant_{quad}"]={
                "original_is_blue":bool(original_label[first]),
                "original_8x8_bytes":codes[first,0].to(torch.int).tolist(),
                "counterfactual_8x8_bytes":code_new[first,0].to(torch.int).tolist(),
                "absolute_byte_difference_8x8":absdiff[first].to(torch.int).tolist(),
                "original_tau_radians":np.round(tau[first,0].numpy(),4).tolist(),
                "wake_magnitude_original":np.round(gradient.numpy(),4).tolist()
            }
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/f"seed{seed}.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    images_html=[]
    for quad,ex in report["example"].items():
        orig=heatmap_html(ex["original_tau_radians"])
        diff=heatmap_html(ex["absolute_byte_difference_8x8"],"delta")
        wake=heatmap_html(ex["wake_magnitude_original"],"wake")
        images_html.append(f'<section><h2>One held-out image: mark flip in {quad}</h2><div class="tiles"><div><h3>Learned phase τ</h3>{orig}</div><div><h3>Byte changes after color flip</h3>{diff}</div><div><h3>Uniform-query commutator magnitude</h3>{wake}</div></div></section>')
    rows="".join(f'<tr><td>{q}</td><td>{v["mean_changed_phase_bytes"]:.2f}</td><td>{v["fraction_change_inside_target_quadrant_mean"]:.1%}</td><td>{v["flipped_mark_accuracy_by_reader"]["commutator"]:.1%}</td><td>{v["flipped_mark_accuracy_by_reader"]["direct_phase"]:.1%}</td><td>{v["unchanged_other_question_predictions_by_reader"]["direct_phase"]:.1%}</td></tr>' for q,v in report["counterfactual_per_quadrant"].items())
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learned phase maps - seed {seed}</title>
<style>body{{background:#081321;color:#e6f1ff;font:15px/1.65 system-ui;margin:0 auto;padding:32px;max-width:1100px}}a{{color:#6cf0d9}}h1{{font-size:40px}}h2{{font-size:23px}}.tiles{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.tiles div{{min-width:0}}svg{{width:100%;max-width:260px;border:1px solid #3d596e;image-rendering:pixelated;border-radius:7px}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #355}}.tag{{color:#6cf0d9;letter-spacing:.15em;font-size:11px}}.note{{color:#a4b8cc;max-width:820px}}@media(max-width:640px){{.tiles{{grid-template-columns:1fr}}}}</style>
<p class="tag">ACTUAL PYTORCH CHECKPOINT / NO ANALYTIC SUBSTITUTE</p><h1>What seed {seed} stored</h1><p class="note">{n} fresh synthetic scenes, strict 64-byte quantized phase maps. The color flip changes exactly four input pixels and no others. The visualizations are retrieved from the trained weights.</p>
<p>Phase σ = {report["phase_statistics"]["std_rad"]:.5f} rad · Pooled byte entropy = {entropy:.3f} bits/value · Unique byte values observed = {int((hist>0).sum())}</p>
<h2>Counterfactual locality and answer tests</h2><table><thead><tr><th>Quadrant</th><th>Changed bytes (mean)</th><th>Change within region</th><th>Flip correct: commutator</th><th>Flip correct: direct phase</th><th>Other questions unchanged</th></tr></thead><tbody>{rows}</tbody></table>
{"".join(images_html)}
<p class="note"><strong>Important:</strong> the writer is a convolutional network with finite receptive fields; local changes are architecturally encouraged. No optical wave propagation, lipid effects, or eikonal solution is inferred. Wake magnitude is a deterministic function of the learned phase map.</p>
<p><a href="seed{seed}.json">Download full per-scene-summary JSON, including exact example byte maps</a></p></html>'''
    (outdir/f"seed{seed}.html").write_text(page,encoding="utf-8")
    return report, tau


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--limit",type=int,default=256)
    p.add_argument("--out",default="checkpoint_inspection")
    p.add_argument("--print-summary",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(),4))
    # Fixed independent evaluation split shared across seeds.
    images,_,_=make_examples(args.limit,20261014)
    output=Path(args.out)
    reports={}
    maps={}
    for seed in (41,97):
        path=Path(f"checkpoints/phase_wake_seed{seed}.pt")
        if not path.exists():raise FileNotFoundError(f"Missing real checkpoint: {path}")
        reports[seed],maps[seed]=inspect_model(path,images,seed,output)
    a,b=maps[41].flatten(1),maps[97].flatten(1)
    cross={}
    for i in range(len(a)):
        x=a[i].numpy();y=b[i].numpy()
        if x.std()<1e-8 or y.std()<1e-8:continue
        cross[i]=float(np.corrcoef(x,y)[0,1])
    overview={
        "protocol":"checkpoint inspection v1; frozen weights, no training",
        "common_independent_scene_seed":20261014,
        "scenes":args.limit,
        "seeds":{},
        "same_scene_cross_seed_spatial_map_correlation_mean":float(np.mean(list(cross.values()))) if cross else None,
        "same_scene_cross_seed_spatial_map_correlation_std":float(np.std(list(cross.values()))) if cross else None,
    }
    for seed,report in reports.items():
        region=report["counterfactual_per_quadrant"]
        overview["seeds"][seed]={
            "phase_statistics":report["phase_statistics"],
            "mean_changed_phase_bytes_per_mark_flip":float(np.mean([q["mean_changed_phase_bytes"] for q in region.values()])),
            "mean_fraction_effect_inside_changed_quadrant":float(np.mean([q["fraction_change_inside_target_quadrant_mean"] for q in region.values()])),
            "counterfactual_correct_after_flip":{
                name:float(np.mean([q["flipped_mark_accuracy_by_reader"][name] for q in region.values()]))
                for name in SELECT_ARMS
            },
            "other_quadrant_answers_stable":{
                name:float(np.mean([q["unchanged_other_question_predictions_by_reader"][name] for q in region.values()]))
                for name in SELECT_ARMS
            }
        }
    (output/"summary.json").write_text(json.dumps(overview,indent=2)+"\n",encoding="utf-8")
    if args.print_summary:
        print("CHECKPOINT_INSPECTION_SUMMARY_BEGIN")
        print(json.dumps(overview,indent=2))
        print("CHECKPOINT_INSPECTION_SUMMARY_END")
    else:
        print("Wrote summaries and visualizations to", output)


if __name__=="__main__":
    main()
