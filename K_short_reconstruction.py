"""
K_short_reconstruction.py — K0S → π+π- 재구성 (signal (a) 테스트 샘플)

진행 상태 (2026-11-10 기준)
  [완료] track cut → 4.47 track/event, 반대전하 쌍 6248개
  [완료] Test 1: gen pion 쌍으로 M_ππ → 0.4976 에 바늘 봉우리 (~5600쌍). 공식 OK.
  [완료] Test 2: gen 쌍 중 두 pion 모두 pT>0.95, |η|<2.4 → 523쌍 (기대 signal 수)
  [완료] Test 3: gen pion ↔ EFlowTrack ΔR 매칭률 (sig cut 후)
             0.02: 147/1046, 0.05: 393, 0.1: 625, 0.2: 814
         → Delphes track 의 phi 는 perigee 기준이라 displaced track 은 B-field bending 만큼 틀어짐.
           reco M_ππ 에 봉우리가 안 보이는 원인.
  [다음] Δphi(track - gen) vs charge/pT 산점도로 bending 부호 규약 확정
         → 2D 원 교점으로 decay vertex 와 그 점의 phi 복원 → M_ππ 재계산
"""
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
for cut in [0.02, 0.05, 0.1, 0.2]:
    print(f"dR<{cut}: {int(ak.sum(ak.any(dr3 < cut, axis=2)))} / {n_pi}")

# ---------------------------------------------------------------------------
# 4. [다음 단계] bending 부호 규약: Δphi(track - gen pion) vs charge/pT
#    기대: 원점을 지나는 직선 근처 (기울기 ∝ decay radius 라 흩어지지만 부호는 일관)
#          Δeta 는 B-field 영향이 없으니 작아야 함 (독립 검증)
# ---------------------------------------------------------------------------
dr3_masked = ak.mask(dr3, dr3 < 0.2)
best = ak.argmin(dr3_masked, axis=2, keepdims=True)
trk_best = ak.firsts(m3.trk[best], axis=2)                    # pion 마다 가장 가까운 track (없으면 None)
ok = ~ak.is_none(trk_best, axis=1)
pi_m, trk_m = pi_gen[ok], trk_best[ok]

dphi = ak.flatten(delta_phi(trk_m.phi, pi_m.phi))
deta = ak.flatten(trk_m.eta - pi_m.eta)
qpt  = ak.flatten(trk_m.charge / trk_m.pt)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(ak.to_numpy(qpt), ak.to_numpy(dphi), s=4)
ax[0].set_xlabel("charge / pT [1/GeV]"); ax[0].set_ylabel("phi(track) - phi(gen pion) [rad]")
ax[1].scatter(ak.to_numpy(qpt), ak.to_numpy(deta), s=4)
ax[1].set_xlabel("charge / pT [1/GeV]"); ax[1].set_ylabel("eta(track) - eta(gen pion)")
plt.tight_layout(); plt.show()

# (참고) M_ππ 히스토그램: reco 와 gen 겹쳐 그리기
# plt.hist(ak.to_numpy(ak.flatten(m_reco)), bins=80, range=(0.3, 0.7), histtype="step", label="reco")
# plt.hist(ak.to_numpy(ak.flatten(m_gen)),  bins=80, range=(0.3, 0.7), histtype="step", label="gen")
# plt.xlabel("M_pipi [GeV]"); plt.legend(); plt.show()
