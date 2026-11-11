import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from common import (load_objects, select_dilepton, match_primary, count,
                    pair_mass, delta_r, delta_phi, PION_MASS, KS_MASS)

t = uproot.open("sig_a_test.root")["Delphes"]
obj = load_objects(t)
gen, mother_pid, track = obj["gen"], obj["mother_pid"], obj["track"]

mask, good_jet, inv_mass, cutflow = select_dilepton(obj["lepton"], obj["jet"], obj["met"])
print("cutflow:", cutflow)                       # 기대: lead 629, sub 221, mll 216, zveto 198, met 147, jet 119
match = match_primary(obj["quark"], good_jet)

# ---------------------------------------------------------------------------
# 1. charged hadron track cut (논문 Sec.2)
# ---------------------------------------------------------------------------
charged = (abs(track.pid) != 11) & (abs(track.pid) != 13)
pt_cut  = track.pt > 0.95
eta_cut = abs(track.eta) < 2.4

# ErrorD0 = 0 인 더미 track (pt 0, eta 1e3) 이 55개 있음 → 오차를 inf 로 두면 significance 0 → 탈락
d0_err_safe = ak.where(track.d0_err > 0, track.d0_err, np.inf)
track = ak.with_field(track, abs(track.d0) / d0_err_safe, "d0_sig")

count("charged",   track[charged])
count("+pt",       track[charged & pt_cut])
count("+eta",      track[charged & pt_cut & eta_cut])
track_kin = track[charged & pt_cut & eta_cut]            # significance cut 전 (Test 3 용)
track_sel = track_kin[track_kin.d0_sig > 2]              # 최종 후보 track
count("+d0sig",    track_sel)                            # 기대: total 4472, 4.47/event

# ---------------------------------------------------------------------------
# 2. reco 반대전하 쌍 → M_ππ  (현재 봉우리 없음 — perigee phi 문제)
# ---------------------------------------------------------------------------
pair_reco = ak.combinations(track_sel, 2, fields=["a", "b"], axis=1)
pair_reco = pair_reco[pair_reco.a.charge * pair_reco.b.charge < 0]
m_reco = pair_mass(pair_reco.a, pair_reco.b, PION_MASS, PION_MASS)
print("reco opposite-charge pairs:", int(ak.sum(ak.num(pair_reco))))   # 기대 6248

# ---------------------------------------------------------------------------
# 3. Test 1: gen-level K0S → π+π-  (공식 검증)
# ---------------------------------------------------------------------------
from_ks = ak.fill_none(mother_pid == 310, False)
pion_gen = gen[(abs(gen.pid) == 211) & from_ks]
pair_gen = ak.combinations(pion_gen, 2, fields=["a", "b"], axis=1)
pair_gen = pair_gen[(pair_gen.a.m1 == pair_gen.b.m1) & (pair_gen.a.pid * pair_gen.b.pid < 0)]  # 같은 K0S
m_gen = pair_mass(pair_gen.a, pair_gen.b, PION_MASS, PION_MASS)
print("gen K0S->pipi pairs:", int(ak.sum(ak.num(pair_gen))))           # 기대 ~5600

# Test 2: 두 pion 모두 kinematic cut 통과 → reco 에서 볼 수 있는 최대 signal
kin = ((pair_gen.a.pt > 0.95) & (pair_gen.b.pt > 0.95) &
       (abs(pair_gen.a.eta) < 2.4) & (abs(pair_gen.b.eta) < 2.4))
pair_gen_kin = pair_gen[kin]
print("gen pairs passing kinematic cuts:", int(ak.sum(ak.num(pair_gen_kin))))   # 기대 523

# Test 3: 그 pion 들이 EFlowTrack 에 있는가 (ΔR 매칭률 scan)
pi_gen = ak.concatenate([pair_gen_kin.a, pair_gen_kin.b], axis=1)
m3 = ak.cartesian({"pi": pi_gen, "trk": track_sel}, axis=1, nested=True)   # pion → (track...)
dr3 = delta_r(m3.pi.eta, m3.pi.phi, m3.trk.eta, m3.trk.phi)
n_pi = int(ak.sum(ak.num(pi_gen)))


# Vertex Reconstruction

def match_track(pion, tracks, dr_max=0.2, deta_max=0.01):
    m = ak.cartesian({"pi": pion, "trk": tracks}, axis=1, nested=True)
    dr = delta_r(m.pi.eta, m.pi.phi, m.trk.eta, m.trk.phi)
    good = (dr < dr_max) & (abs(m.pi.eta - m.trk.eta) < deta_max)
    best = ak.argmin(ak.mask(dr, good), axis=2, keepdims=True)
    return ak.firsts(m.trk[best], axis=2)

trk_a = match_track(pair_gen_kin.a, track_sel)
trk_b = match_track(pair_gen_kin.b, track_sel)
both = ~ak.is_none(trk_a, axis=1) & ~ak.is_none(trk_b, axis=1)
true_pair = ak.zip({"a": trk_a[both], "b": trk_b[both], "vx": pair_gen_kin.a.x[both], "vy": pair_gen_kin.a.y[both]})

def track_circle(trk, B=3.8):
    R = trk.pt / (0.3 * B) * 1000.0
    perx = trk.d0 * np.sin(trk.phi)
    pery = - trk.d0 * np.cos(trk.phi)
    cx = perx + trk.charge * R * np.sin(trk.phi)
    cy = pery - trk.charge * R * np.cos(trk.phi)
    return R, cx, cy

def find_vertex(a, b):
    Ra, cxa, cya = track_circle(a); Rb, cxb, cyb = track_circle(b)
    dx, dy = cxb - cxa, cyb - cya
    d_raw = np.sqrt(dx**2 + dy**2)
    valid = d_raw > 1e-6
    d = ak.where(valid, d_raw, 1.0)
    l = (d**2 + Ra**2 - Rb**2) / (2*d)          # 중심선 위의 거리 (a 와 충돌하지 않게 l 로)
    h2 = Ra**2 - l**2
    h = np.sqrt(np.maximum(h2, 0))
    mx, my = cxa + l*dx/d, cya + l*dy/d
    nx, ny = -dy/d, dx/d                         # 중심선에 수직인 단위벡터
    return ak.zip({"v1x": mx + h*nx, "v1y": my + h*ny,
                   "v2x": mx - h*nx, "v2y": my - h*ny,
                   "h2": h2, "mx": mx, "my": my, "valid": valid})

def forward_arc(trk, vx, vy):
    R, cx, cy = track_circle(trk)
    perx, pery = trk.d0 * np.sin(trk.phi), -trk.d0 * np.cos(trk.phi)
    th_p = np.arctan2(pery - cy, perx- cx)
    th_v = np.arctan2(vy - cy, vx - cx)
    fwd = ak.where(trk.charge > 0, th_p - th_v, th_v - th_p)
    return fwd % (2*np.pi)

def phi_at(trk, vx, vy):
    R, cx, cy = track_circle(trk)
    rx, ry = vx - cx, vy - cy
    return np.arctan2(-trk.charge * rx, trk.charge * ry)

# pair_reco
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

# true_pair
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

# Plotting with sideband extra
m_r = ak.to_numpy(ak.flatten(m_reco_v)); m_t = ak.to_numpy(ak.flatten(m_true_v))

plt.hist(m_r, bins=80, range=(0.3, 0.7), histtype="step", color="b", label="all reco pairs")
plt.hist(m_t, bins=80, range=(0.3, 0.7), histtype="step", color="r", label="true Kaon pairs")
plt.axvline(KS_MASS, ls="--", lw=0.8); plt.xlabel("M_pipi [GeV]"); plt.legend(); plt.show()

lo, hi = KS_MASS - 0.1, KS_MASS + 0.1
n_win = np.sum((m_r > lo) & (m_r < hi))
n_left = np.sum((m_r > lo - 0.1) & (m_r < lo))
n_right = np.sum((m_r > hi) & (m_r < hi + 0.1))
excess = n_win - (n_left + n_right)
n_true_win = np.sum((m_t > lo) & (m_t < hi))
print(f"window {n_win}, sidebands {n_left} + {n_right}, excess {excess}, true pairs in window {n_true_win}")
print(len(m_r), lo, hi)
print(np.sum((m_r > lo) & (m_r < hi)))