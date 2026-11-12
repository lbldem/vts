import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from common import (load_objects, select_dilepton, match_primary, count,
                    pair_mass, delta_r, delta_phi, four_vector, find_vertex, 
                    forward_arc, phi_at, track_circle, match_track, PION_MASS, KS_MASS)

t = uproot.open("sig_a_test.root")["Delphes"]
obj, _ = load_objects(t)
gen, mother_pid, track = obj["gen"], obj["mother_pid"], obj["track"]
mask, good_jet, inv_mass, cutflow = select_dilepton(obj["lepton"], obj["jet"], obj["met"])
match = match_primary(obj["quark"], good_jet)

# ---------------------------------------------------------------------------
# 1. charged hadron track cut (논문 Sec.2)
# ---------------------------------------------------------------------------
charged = (abs(track.pid) != 11) & (abs(track.pid) != 13)
pt_cut  = track.pt > 0.95
eta_cut = abs(track.eta) < 2.4
track_kin = track[charged & pt_cut & eta_cut]            # significance cut 전 (Test 3 용)
track_sel = track_kin[track_kin.d0_sig > 2]              # 최종 후보 track

# ---------------------------------------------------------------------------
# 2. reco 반대전하 쌍 → M_ππ  (현재 봉우리 없음 — perigee phi 문제)
# ---------------------------------------------------------------------------
pair_reco = ak.combinations(track_sel, 2, fields=["a", "b"], axis=1)
pair_reco = pair_reco[pair_reco.a.charge * pair_reco.b.charge < 0]

# ---------------------------------------------------------------------------
# 3. Test 1: gen-level K0S → π+π-  (공식 검증)
# ---------------------------------------------------------------------------
from_ks = ak.fill_none(mother_pid == 310, False)
pion_gen = gen[(abs(gen.pid) == 211) & from_ks]
pair_gen = ak.combinations(pion_gen, 2, fields=["a", "b"], axis=1)
pair_gen = pair_gen[(pair_gen.a.m1 == pair_gen.b.m1) & (pair_gen.a.pid * pair_gen.b.pid < 0)]  # 같은 K0S

# Test 2: 두 pion 모두 kinematic cut 통과 → reco 에서 볼 수 있는 최대 signal
kin = ((pair_gen.a.pt > 0.95) & (pair_gen.b.pt > 0.95) &
       (abs(pair_gen.a.eta) < 2.4) & (abs(pair_gen.b.eta) < 2.4))
pair_gen_kin = pair_gen[kin]

# Test 3: 그 pion 들이 EFlowTrack 에 있는가 (ΔR 매칭률 scan)
pi_gen = ak.concatenate([pair_gen_kin.a, pair_gen_kin.b], axis=1)
m3 = ak.cartesian({"pi": pi_gen, "trk": track_sel}, axis=1, nested=True)   # pion → (track...)
dr3 = delta_r(m3.pi.eta, m3.pi.phi, m3.trk.eta, m3.trk.phi)

# Vertex Reconstruction
trk_a = match_track(pair_gen_kin.a, track_sel)
trk_b = match_track(pair_gen_kin.b, track_sel)
both = ~ak.is_none(trk_a, axis=1) & ~ak.is_none(trk_b, axis=1)
true_pair = ak.zip({"a": trk_a[both], "b": trk_b[both], "vx": pair_gen_kin.a.x[both], "vy": pair_gen_kin.a.y[both]})

## true_pair
vt = find_vertex(true_pair.a, true_pair.b)
keep_t = vt.valid
vt, true_pair = vt[keep_t], true_pair[keep_t]
s1t = forward_arc(true_pair.a, vt.v1x, vt.v1y) + forward_arc(true_pair.b, vt.v1x, vt.v1y)
s2t = forward_arc(true_pair.a, vt.v2x, vt.v2y) + forward_arc(true_pair.b, vt.v2x, vt.v2y)
pick1t = s1t < s2t
vxt = ak.where(pick1t, vt.v1x, vt.v2x)
vyt = ak.where(pick1t, vt.v1y, vt.v2y)
a_vt = ak.with_field(true_pair.a, phi_at(true_pair.a, vxt, vyt), "phi")
b_vt = ak.with_field(true_pair.b, phi_at(true_pair.b, vxt, vyt), "phi")
m_true_v = pair_mass(a_vt, b_vt, PION_MASS, PION_MASS)


## pair_reco
vr = find_vertex(pair_reco.a, pair_reco.b)
keep = vr.valid
vr, pair_reco = vr[keep], pair_reco[keep]
s1r = forward_arc(pair_reco.a, vr.v1x, vr.v1y) + forward_arc(pair_reco.b, vr.v1x, vr.v1y)
s2r = forward_arc(pair_reco.a, vr.v2x, vr.v2y) + forward_arc(pair_reco.b, vr.v2x, vr.v2y)
pick1 = s1r < s2r
vx = ak.where(pick1, vr.v1x, vr.v2x)
vy = ak.where(pick1, vr.v1y, vr.v2y)
a_v = ak.with_field(pair_reco.a, phi_at(pair_reco.a, vx, vy), "phi")
b_v = ak.with_field(pair_reco.b, phi_at(pair_reco.b, vx, vy), "phi")
m_reco_v = pair_mass(a_v, b_v, PION_MASS, PION_MASS)

# K_S candidate record packing
Ea, pxa, pya, pza = four_vector(a_v, PION_MASS); Eb, pxb, pyb, pzb = four_vector(b_v, PION_MASS)
px, py, pz = pxa + pxb, pya + pyb, pza + pzb
pt_k = np.sqrt(px**2 + py**2)
eta_k = np.arcsinh(pz / pt_k)
phi_k = np.arctan2(py, px)
lxy = np.sqrt(vx**2 + vy**2)
cos_a = (vx*px + vy*py) / (lxy*pt_k)

ks = ak.zip({
    "mass": m_reco_v, "pt": pt_k, "eta": eta_k, "phi": phi_k,
    "lxy": lxy, "cos_a": cos_a, "dca": vr.dca2d,
    "frac_a": a_v.pt/pt_k, "frac_b": b_v.pt/pt_k,
    "d0sig_a": a_v.d0_sig, "d0sig_b": b_v.d0_sig,
    "dzsig_a": a_v.dz_sig, "dzsig_b": b_v.dz_sig})

Eat, pxat, pyat, pzat = four_vector(a_vt, PION_MASS); Ebt, pxbt, pybt, pzbt = four_vector(b_vt, PION_MASS)
pxt, pyt, pzt = pxat + pxbt, pyat + pybt, pzat + pzbt
pt_kt = np.sqrt(pxt**2 + pyt**2)
eta_kt = np.arcsinh(pzt / pt_kt)
phi_kt = np.arctan2(pyt, pxt)
lxyt = np.sqrt(vxt**2 + vyt**2)
cos_at = (vxt*pxt + vyt*pyt) / (lxyt*pt_kt)

kst = ak.zip({
    "mass": m_true_v, "pt": pt_kt, "eta": eta_kt, "phi": phi_kt,
    "lxy": lxyt, "cos_a": cos_at,
    "frac_a": a_vt.pt/pt_kt, "frac_b": b_vt.pt/pt_kt})

ks_win = ks[abs(ks.mass - KS_MASS) < 0.1]
match_jet_nonempty = match.jet[~ak.is_none(match.jet, axis=1)]
ks_jet_pair = ak.cartesian({"jet": match_jet_nonempty, "ks": ks_win}, axis=1, nested=True)
dr_ks = delta_r(ks_jet_pair.jet.eta, ks_jet_pair.jet.phi, ks_jet_pair.ks.eta, ks_jet_pair.ks.phi)
close_ks = dr_ks < 0.4
pt_close_ks = ak.mask(ks_jet_pair.ks.pt, close_ks)
best_ks = ak.argmax(pt_close_ks, axis=2, keepdims=True)
ks_cand = ak.firsts(ks_jet_pair.ks[best_ks], axis=2)

kst_win = kst[abs(kst.mass - KS_MASS) < 0.1]
kst_jet_pair = ak.cartesian({"jet": match_jet_nonempty, "ks": kst_win}, axis=1, nested=True)
dr_kst = delta_r(kst_jet_pair.jet.eta, kst_jet_pair.jet.phi, kst_jet_pair.ks.eta, kst_jet_pair.ks.phi)
close_kst = dr_kst < 0.4
pt_close_kst = ak.mask(kst_jet_pair.ks.pt, close_kst)
best_kst = ak.argmax(pt_close_kst, axis=2, keepdims=True)
kst_cand = ak.firsts(kst_jet_pair.ks[best_kst], axis=2)

x_array = ak.drop_none(ks_cand.pt/match_jet_nonempty.pt)
x_array_clean = x_array[ak.num(x_array) > 0]
xt_array = ak.drop_none(kst_cand.pt/match_jet_nonempty.pt)
xt_array_clean = xt_array[ak.num(xt_array) > 0]

# plt.hist(ak.flatten(x_array_clean), bins=50, range=(0.0, 1.0), histtype="step", color='r', label="reco")
# plt.hist(ak.flatten(xt_array_clean), bins=50, range=(0.0, 1.0), histtype="step", color='b', label="true")
# plt.legend(); plt.savefig("momentum_fraction_comparison.png", dpi=300)

# print("x_reco has", len(ak.flatten(x_array_clean)), "entries.")
# print("x_reco > 0.15 cut leaves", f"{len(ak.flatten(x_array_clean[x_array_clean > 0.15]))/len(ak.flatten(x_array_clean)):.3%}")
# print("x_true has", len(ak.flatten(xt_array_clean)), "entries.")
# print("x_true > 0.15 cut leaves", f"{len(ak.flatten(xt_array_clean[xt_array_clean > 0.15]))/len(ak.flatten(xt_array_clean)):.3%}")