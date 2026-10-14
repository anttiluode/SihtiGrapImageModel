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


def test_interferometric_endpoint_identity_and_clutter_invariance():
    from probe_gate import phase_at_path, path_bank
    torch.manual_seed(31415)
    bank=path_bank()
    tau=torch.randn(7,64)*.35
    for q in range(4):
        pre=bank["pre"][q,:4][None].expand(len(tau),-1,-1)
        post=bank["post"][q,:4][None].expand(len(tau),-1,-1)
        z1=phase_at_path(tau,pre)
        z2=phase_at_path(tau,post)
        ratio=z1*z2.conj()
        source=bank["source"][q,:4]
        port=bank["end"][q,:4]
        expected=torch.polar(torch.ones_like(ratio.real),tau[:,source]-tau[:,port])
        assert torch.allclose(ratio,expected,atol=3e-6)

        for path_ix in range(4):
            # Change only *intermediate* cells, never source or port.
            source_idx=int(source[path_ix])
            port_idx=int(port[path_ix])
            mask=set(pre[0,path_ix,1:].tolist())
            mask.update(post[0,path_ix,:-1].tolist())
            mask.discard(source_idx)
            mask.discard(port_idx)
            assert mask, "Need at least one genuinely intermediate site"
            mixed=tau.clone()
            for pos in mask:
                mixed[:,pos]+=torch.randn(len(tau))*.45
            w1=phase_at_path(mixed,pre[:,path_ix:path_ix+1])[:,0]
            w2=phase_at_path(mixed,post[:,path_ix:path_ix+1])[:,0]
            assert torch.allclose(
                w1*w2.conj(),ratio[:,path_ix],atol=3e-6
            )
            # A subtraction is sensitive to a common-path phase rotation.
            assert ((w1-w2)-(z1[:,path_ix]-z2[:,path_ix])).abs().mean()>.001


def test_ratio_and_demodulation_equal_up_to_one_complex_offset():
    from probe_gate import observed_features
    t=torch.randn(5,1,8,8)*.3
    q=torch.tensor([0,1,2,3,0])
    for k in (2,4,8):
        f=observed_features(t,q,k)
        r=f["interferometric_ratio"][:,:k].view(len(q),k//2,2)
        d=f["demodulated_difference"][:,:k].view(len(q),k//2,2)
        assert torch.allclose(d[:,:,0],r[:,:,0]-1,atol=2e-6)
        assert torch.allclose(d[:,:,1],r[:,:,1],atol=2e-6)
        assert f["interferometric_ratio"].shape==(len(q),2*k)
        assert torch.all(f["interferometric_ratio"][:,k:]==0)


def test_direct_matched_sources_are_same_positions_as_interferometry():
    from probe_gate import path_bank, observed_features
    bank=path_bank()
    torch.manual_seed(87)
    tau=torch.randn(6,1,8,8)*.2
    q=torch.tensor([0,1,2,3,0,2])
    for k in (2,4,8):
        f=observed_features(tau,q,k)
        count=k//2
        site=bank["source"][q,:count]
        expected_phase=torch.gather(tau[:,0].flatten(1),1,site)
        expected=torch.polar(torch.ones_like(expected_phase),expected_phase)
        received=f["direct_matched_sources"][:,:k].reshape(len(q),count,2)
        actual=torch.complex(received[:,:,0],received[:,:,1])
        assert torch.allclose(actual,expected,atol=1e-6)
        assert torch.all(f["direct_matched_sources"][:,k:]==0)
