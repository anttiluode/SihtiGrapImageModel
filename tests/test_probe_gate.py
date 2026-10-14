import torch
from probe_gate import path_bank,observed_features,ARMS,BUDGETS,explicit_wave_probe,dataset_all_questions,expand_questions


def test_all_routes_arrive_at_exactly_two_fixed_ports():
    b=path_bank()
    assert b["pre"].shape==(4,8,12)
    assert b["post"].shape==(4,8,12)
    assert b["source"].shape==(4,8)
    assert set(b["end"].flatten().tolist())=={0,4}


def test_analytic_gather_equals_actual_12_step_wave_evolution():
    torch.manual_seed(17)
    tau=torch.randn(8,8)*.3
    for q in range(4):
        for k in (0,1):
            x=observed_features(tau.reshape(1,1,8,8),torch.tensor([q]),8)
            for shifted,arm in ((False,"active_phase_then_shift"),(True,"active_shift_then_phase")):
                measurement=x[arm][0].reshape(8,2)
                z=torch.complex(measurement[k,0],measurement[k,1])
                ref=explicit_wave_probe(tau,q,k,shifted_first=shifted)
                assert torch.allclose(z,ref,atol=1e-5)
                assert abs(abs(ref).item()-1)<1e-5


def test_constant_medium_commutes_for_all_fixed_routes():
    t=torch.full((2,1,8,8),.25)
    q=torch.tensor([0,3])
    for K in BUDGETS:
        r=observed_features(t,q,K)
        assert all(v.shape==(2,2*K) for v in r.values())
        assert torch.allclose(r["order_difference"],torch.zeros_like(r["order_difference"]),atol=1e-6)


def test_fixed_routes_do_not_depend_on_target_question():
    t=torch.randn(4,1,8,8)*.4
    a=observed_features(t,torch.zeros(4,dtype=torch.long),4)
    b=observed_features(t,torch.full((4,),3,dtype=torch.long),4)
    assert torch.allclose(a["fixed_routes"],b["fixed_routes"])
    assert torch.allclose(a["passive_ports"],b["passive_ports"])


def test_all_labels_for_each_scene_and_expansion():
    scenes=dataset_all_questions(6,917)
    im,y=scenes
    assert im.shape==(6,3,32,32) and y.shape==(6,4)
    t=torch.randn(6,1,8,8)
    mem,q,a=expand_questions(t,y)
    assert mem.shape==(24,1,8,8)
    assert q.tolist()==[0,1,2,3]*6
    assert a.tolist()==y.reshape(-1).tolist()
