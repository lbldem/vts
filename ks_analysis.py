"""
ks_analysis.py — signal (a) 테스트 샘플에서 K0S 재구성 + jet 매칭

기대값 (sig_a_test.root, 1000 events)
  cutflow           lead 629, sub 221, mll 216, zveto 198, met 147, jet 119
  track_sel         4472 (4.47/event)
  pair_reco         6248 반대전하 쌍
  gen K0S→ππ        5634 → kin cut 523 → 두 track 재구성 203
  mass window       reco 719, sideband 369+269 → excess ~80 ± 37, true in window 128
  질량 해상도        σ ≈ 70 MeV (Delphes track pT 해상도 6~25% 가 원인)
  primary jet 수    ~1495, 그중 window K0S 가 붙은 jet 307 (진짜 78)
"""
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from common import (load_objects, select_dilepton, select_tracks, match_primary,
                    true_ks_pairs, count, flat, KS_MASS)
from delphes_v0 import reconstruct_v0, attach_to_jets

# 1. 읽기 + selection ==============================================================
t = uproot.open("sig_a_test.root")["Delphes"]
obj = load_objects(t)
gen, mother_pid, track = obj["gen"], obj["mother_pid"], obj["track"]

mask, good_jet, inv_mass, cutflow = select_dilepton(obj["lepton"], obj["jet"], obj["met"])
print("cutflow:", cutflow)
match = match_primary(obj["quark"], good_jet)         # 이벤트 → quark → {quark, jet|None}

track_kin, track_sel = select_tracks(track)
count("track_kin", track_kin); count("track_sel", track_sel)

# 2. reco 쌍 → K0S 후보 ==================================================================
pair_reco = ak.combinations(track_sel, 2, fields=["a", "b"], axis=1)
pair_reco = pair_reco[pair_reco.a.charge * pair_reco.b.charge < 0]
print("reco opposite-charge pairs:", int(ak.sum(ak.num(pair_reco))))

ks, pair_reco, vr = reconstruct_v0(pair_reco)         # 본 분석 경로

# 3. truth 쌍 (MC 전용) — 같은 경로 ===========================================================
pair_gen, pair_gen_kin, true_pair = true_ks_pairs(gen, mother_pid, track_sel)
print("gen K0S->pipi:", int(ak.sum(ak.num(pair_gen))),
      " kin:", int(ak.sum(ak.num(pair_gen_kin))),
      " both tracks:", int(ak.sum(ak.num(true_pair))))

kst, true_pair, vt = reconstruct_v0(true_pair)        # 검증 경로 (같은 함수)

# vertex 정확도 (선택 규칙과 무관한 최선값)
d1 = np.sqrt((vt.v1x - true_pair.vx)**2 + (vt.v1y - true_pair.vy)**2)
d2 = np.sqrt((vt.v2x - true_pair.vx)**2 + (vt.v2y - true_pair.vy)**2)
vtx_err = np.minimum(d1, d2)
print("non-intersecting true pairs:", int(ak.sum(vt.h2 < 0)), "/", int(ak.sum(ak.num(vt))))

# 4. 질량: 그림 + sideband ==============================================================
m_r, m_t = flat(ks.mass), flat(kst.mass)
plt.hist(m_r, bins=80, range=(0.3, 0.7), histtype="step", color="b", label="all reco pairs")
plt.hist(m_t, bins=80, range=(0.3, 0.7), histtype="step", color="r", label="true K0S pairs")
plt.axvline(KS_MASS, ls="--", lw=0.8); plt.xlabel("M_pipi [GeV]"); plt.legend()
plt.savefig("mass_pipi.png", dpi=150); plt.close()

lo, hi = KS_MASS - 0.1, KS_MASS + 0.1
n_win   = np.sum((m_r > lo) & (m_r < hi))
n_side  = np.sum((m_r > lo - 0.1) & (m_r < lo)) + np.sum((m_r > hi) & (m_r < hi + 0.1))
print(f"window {n_win}, sidebands {n_side}, excess {n_win - n_side} ± {np.sqrt(n_win + n_side):.0f}, "
      f"true in window {np.sum((m_t > lo) & (m_t < hi))}")

# 5. mass window → jet 에 붙이기 ==============================================================
ks_win  = ks[abs(ks.mass - KS_MASS) < 0.1]
kst_win = kst[abs(kst.mass - KS_MASS) < 0.1]

has_jet = ~ak.is_none(match.jet, axis=1)
pjet, pq = match.jet[has_jet], match.quark[has_jet]   # primary jet 과 그 quark (같은 mask)
ks_cand  = attach_to_jets(ks_win,  pjet)              # jet 마다 후보 하나 또는 None
kst_cand = attach_to_jets(kst_win, pjet)
print("primary jets:", int(ak.sum(ak.num(pjet))),
      " with K0S cand:", int(ak.sum(~ak.is_none(ks_cand, axis=1))),
      " with true K0S:", int(ak.sum(~ak.is_none(kst_cand, axis=1))))

# 6. x = pT(K0S)/pT(jet), flavor 별 ==============================================================
x, xt = ks_cand.pt / pjet.pt, kst_cand.pt / pjet.pt
is_s = abs(pq.pid) == 3

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(flat(x), bins=40, range=(0, 1), histtype="step", color="r", label="all cand")
ax[0].hist(flat(xt), bins=40, range=(0, 1), histtype="step", color="b", label="true K0S")
ax[0].set_xlabel("x = pT(K0S)/pT(jet)"); ax[0].legend()
ax[1].hist(flat(xt[is_s]),  bins=20, range=(0, 1), histtype="step", color="g", label="true K0S in s jet", density=True)
ax[1].hist(flat(xt[~is_s]), bins=20, range=(0, 1), histtype="step", color="m", label="true K0S in b jet", density=True)
ax[1].axvline(0.15, ls="--", lw=0.8); ax[1].set_xlabel("x (true K0S only)"); ax[1].legend()
plt.tight_layout(); plt.savefig("x_fraction.png", dpi=150); plt.close()

for name, arr in [("all cand", x), ("true", xt), ("true s-jet", xt[is_s]), ("true b-jet", xt[~is_s])]:
    v = flat(arr)
    if len(v):
        print(f"{name:12s} n={len(v):4d}  x>0.15: {np.mean(v > 0.15)*100:5.1f}%")
