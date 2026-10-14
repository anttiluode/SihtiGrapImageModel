"""Frozen v2 -> v3: question-conditioned, restricted-port phase probes.

Every active read launches a unit-amplitude point pulse at a question-
selected site, applies 12 (phase multiply, spatial shift) operations, and
reads ONE complex value at one of TWO fixed detector ports.

The toroidal grid and precommitted paths are model abstractions, not an
eikonal PDE solver or evidence of a physical wave delay.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

import torch

GRID=8
STEPS=12
PORTS=((0,0),(0,4))
OFFSETS=((1,1),(3,3),(1,3),(3,1))
ARMS=(
    "active_phase_then_shift",
    "active_shift_then_phase",
    "paired_orders",
    "order_difference",
    "interferometric_ratio",
    "demodulated_difference",
    "fixed_routes",
    "direct_local_phase",
    "direct_local_amplitude",
    "passive_ports",
)
BUDGETS=(2,4,8)


def toroidal_distance(a,b):
    return sum(min((v-u)%GRID,(u-v)%GRID) for u,v in zip(a,b))


def path_from_to(start,end,steps,rng):
    """Uniform-ish random *admissible* cardinal path with exact endpoint."""
    directions=((0,1),(1,0),(0,-1),(-1,0))
    p=start
    moves=[]
    for t in range(steps):
        remaining=steps-t-1
        possible=[]
        for dy,dx in directions:
            np=((p[0]+dy)%GRID,(p[1]+dx)%GRID)
            d=toroidal_distance(np,end)
            if d<=remaining and (remaining-d)%2==0:
                possible.append((dy,dx,np))
        if not possible:
            raise RuntimeError(f"Cannot complete path from {start} to {end}")
        dy,dx,pnew=rng.choice(possible)
        moves.append((p[0]*GRID+p[1],pnew[0]*GRID+pnew[1],dy,dx))
        p=pnew
    if p!=end: raise AssertionError("Path does not terminate on fixed port")
    return moves


@lru_cache(maxsize=1)
def path_bank():
    """Pre-register geometry without examining a dataset or any labels."""
    paths=[]
    for q in range(4):
        by,bx=(q//2)*4,(q%2)*4
        route=[]
        for k in range(8):
            oy,ox=OFFSETS[k%4]
            start=(by+oy,bx+ox)
            port=PORTS[k%len(PORTS)]
            rng=random.Random(9923+q*1000+k*99)
            # Select a spatially broad, local-exploring trajectory. This
            # criterion is purely geometric, never trained against labels.
            candidates=[path_from_to(start,port,STEPS,rng) for _ in range(100)]
            def score(moves):
                points={p for p,_,_,_ in moves if (p//GRID)//4==q//2 and (p%GRID)//4==q%2}
                return 4*len(points)+sum(((p//GRID)//4==q//2 and (p%GRID)//4==q%2) for p,_,_,_ in moves)
            route.append(max(candidates,key=score))
        paths.append(route)
    pre=torch.tensor([[[m[0] for m in path] for path in route] for route in paths],dtype=torch.long)
    post=torch.tensor([[[m[1] for m in path] for path in route] for route in paths],dtype=torch.long)
    starts=pre[:,:,0]
    ends=post[:,:,-1]
    for q in range(4):
        for k in range(8):
            assert int(ends[q,k])==PORTS[k%2][0]*GRID+PORTS[k%2][1]
            cell=int(starts[q,k]);y,x=divmod(cell,GRID)
            assert y//4==q//2 and x//4==q%2
    return {"pre":pre,"post":post,"source":starts,"end":ends}


def complex_as_reals(z):
    return torch.stack([z.real,z.imag],dim=-1).flatten(start_dim=1)


def phase_at_path(tau,indices):
    """Efficient exact simulation of one-hot unitary trajectories.

    A pulse visits precisely these positions, accumulating tau phase at
    each. Its final complex amplitude at the detector is exp(i sum tau).
    Does not access a learned decoder or any off-path pixel.
    """
    if tau.ndim != 2 or tau.shape[-1] !=64:
        raise ValueError("Expected (batch, 64) phase codes as radians")
    n,k,t=indices.shape
    assert n==len(tau) and t==STEPS
    visited=torch.gather(tau[:,None,:].expand(-1,k,-1),2,indices)
    return torch.polar(torch.ones_like(visited.sum(-1)),visited.sum(-1))


def gather_query_routes(table,q):
    return table.to(q.device)[q]


@torch.no_grad()
def observed_features(tau,q,budget):
    """Outputs [N,2*budget] per arm, all with equal readout dimensions.

    For paired_orders/order_difference: budget/2 paths, two instrument
    readings per path. Zero channels on diff are NOT extra measurements.
    Direct-local controls are deliberately stronger than fixed ports
    because they are allowed addressable source-site read access;
    don't claim they are same-interface port comparators.
    """
    if budget not in BUDGETS:raise ValueError(budget)
    if tau.ndim==4: tau=tau[:,0].flatten(1)
    if tau.shape[1]!=64:raise ValueError(tau.shape)
    if q.shape!=(len(tau),):raise ValueError(q.shape)
    paths=path_bank()
    pre=gather_query_routes(paths["pre"],q)[:,:budget]
    post=gather_query_routes(paths["post"],q)[:,:budget]
    first=phase_at_path(tau,pre)
    second=phase_at_path(tau,post)
    pair_count=budget//2
    raw_pair=torch.stack([first[:,:pair_count],second[:,:pair_count]],dim=-1).flatten(1)
    delta=first[:,:pair_count]-second[:,:pair_count]
    # Paired orderings measure identical intermediate cells, except
    # phase-then-shift includes source and shift-then-phase includes port.
    # Multiplying by conj(second) demodulates the common phase.
    ratio=first[:,:pair_count]*second[:,:pair_count].conj()
    # Equivalent representation: (first-second)*conj(second) = ratio-1.
    demodulated=delta*second[:,:pair_count].conj()
    # Need equal number of returned floats but only budget/2 complex
    # derived differences. The budget pays for both measurements.
    diff_feats=torch.cat([complex_as_reals(delta),torch.zeros_like(complex_as_reals(delta))],dim=1)
    ratio_feats=torch.cat([complex_as_reals(ratio),torch.zeros_like(complex_as_reals(ratio))],dim=1)
    demod_feats=torch.cat([complex_as_reals(demodulated),torch.zeros_like(complex_as_reals(demodulated))],dim=1)

    fixed_idx=paths["pre"][torch.arange(budget)%4,torch.arange(budget)].to(tau.device)
    fixed_idx=fixed_idx[None].expand(len(tau),-1,-1)
    fixed=phase_at_path(tau,fixed_idx)
    selected_sources=gather_query_routes(paths["source"],q)[:,:budget]
    local=torch.gather(tau,1,selected_sources)
    direct=torch.polar(torch.ones_like(local),local)
    amp=(local/1.2+1)*.5

    ports=torch.tensor([PORTS[k%2][0]*GRID+PORTS[k%2][1] for k in range(budget)],device=tau.device)
    static_tau=tau.index_select(1,ports)
    static_phases=torch.polar(torch.ones_like(static_tau),static_tau)
    return {
        "active_phase_then_shift":complex_as_reals(first),
        "active_shift_then_phase":complex_as_reals(second),
        "paired_orders":complex_as_reals(raw_pair),
        "order_difference":diff_feats,
        "interferometric_ratio":ratio_feats,
        "demodulated_difference":demod_feats,
        "fixed_routes":complex_as_reals(fixed),
        "direct_local_phase":complex_as_reals(direct),
        "direct_local_amplitude":torch.stack([amp,torch.zeros_like(amp)],-1).flatten(1),
        "passive_ports":complex_as_reals(static_phases),
    }


def explicit_wave_probe(tau, q, k, shifted_first=False):
    """Slow reference evolution: one-hot wave through an 8×8 grid."""
    bank=path_bank()
    source=int(bank["source"][q,k])
    field=torch.zeros((GRID,GRID),dtype=torch.complex64)
    y,x=divmod(source,GRID)
    field[y,x]=1.
    path_pre=bank["pre"][q,k].tolist()
    path_post=bank["post"][q,k].tolist()
    for src,dst in zip(path_pre,path_post):
        if shifted_first:
            dy,dx=divmod(dst,GRID)
            fy,fx=divmod(src,GRID)
            field=torch.roll(field,(dy-fy,dx-fx),dims=(0,1))
            field=field*torch.polar(torch.ones_like(tau),tau)
        else:
            field=field*torch.polar(torch.ones_like(tau),tau)
            dy,dx=divmod(dst,GRID)
            fy,fx=divmod(src,GRID)
            field=torch.roll(field,(dy-fy,dx-fx),dims=(0,1))
    sensor=int(bank["end"][q,k])
    return field.flatten()[sensor]


def dataset_all_questions(count,seed):
    """Return EVERY quadrant answer for each held-out synthetic scene."""
    from train_speedzones import make_examples
    images,_,_=make_examples(count,seed)
    colors=[]
    for q in range(4):
        y,x=(q//2)*16,(q%2)*16
        red=images[:,0,y:y+16,x:x+16].amax(dim=(-1,-2))
        blue=images[:,2,y:y+16,x:x+16].amax(dim=(-1,-2))
        colors.append((blue>red).long())
    labels=torch.stack(colors,dim=1)
    return images,labels


def expand_questions(phase, labels):
    n=len(phase)
    assert phase.shape==(n,1,8,8) and labels.shape==(n,4)
    return phase.repeat_interleave(4,dim=0),torch.arange(4).repeat(n),labels.flatten()
