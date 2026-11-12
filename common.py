"""
common.py — |Vts| 분석 공용 함수 (Delphes ROOT → awkward)

규칙
  * "이벤트 단위" mask 는 길이 N_event 의 boolean. 적용은 호출자가 마지막에 한 번만.
  * "객체 단위" 거르기 (jet[...], track[...]) 는 이벤트 수를 유지한 채 안쪽 리스트만 줄인다.
  * 짝을 이루는 배열은 항상 같은 mask 로 같은 줄에서 같이 거른다.
  * 변수 이름: reco (jet, track, pair_reco, ks) 와 gen (gen, quark, pion_gen, pair_gen) 을 섞지 않는다.

파일 역할
  common.py      — 데이터에서도 그대로 쓰는 것 + MC truth matching 도구
  delphes_v0.py  — Delphes 전용 K0S(V0) 재구성. 실제 데이터에서는 CMS V0 collection 으로 교체.
"""
import awkward as ak
import numpy as np

# ---------------------------------------------------------------- 상수
PION_MASS = 0.13957      # GeV
KS_MASS   = 0.497611     # GeV
Z_MASS    = 91.1876      # GeV

LEAD_PT, SUB_PT, LEP_ETA = 25.0, 20.0, 2.4
MLL_MIN, Z_WINDOW, MET_MIN = 20.0, 15.0, 40.0
JET_PT, JET_ETA = 30.0, 2.4

TRK_PT, TRK_ETA, TRK_D0SIG = 0.95, 2.4, 2.0      # K0S daughter track cut (논문 Sec.2)


# ---------------------------------------------------------------- 운동학
def four_vector(obj, mass=0.0):
    """(pt, eta, phi) 레코드 → (E, px, py, pz). jagged 구조 유지."""
    px = obj.pt * np.cos(obj.phi)
    py = obj.pt * np.sin(obj.phi)
    pz = obj.pt * np.sinh(obj.eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return E, px, py, pz


def pair_mass(a, b, mass_a=0.0, mass_b=0.0):
    """두 레코드의 invariant mass. 4-vector 를 먼저 더한 뒤 제곱."""
    Ea, pxa, pya, pza = four_vector(a, mass_a)
    Eb, pxb, pyb, pzb = four_vector(b, mass_b)
    return np.sqrt((Ea + Eb)**2 - (pxa + pxb)**2 - (pya + pyb)**2 - (pza + pzb)**2)


def delta_phi(phi1, phi2):
    """phi 차이를 [-pi, pi] 로 접음."""
    dphi = phi1 - phi2
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)


# ---------------------------------------------------------------- 객체 읽기
def load_objects(tree):
    """Delphes tree → dict of awkward 레코드 배열 (전부 길이 N_event).

    keys: ev, electron, muon, lepton, jet, met, gen, mother_pid, quark, track
    """
    ev = tree.arrays([
        "Electron.PT", "Electron.Eta", "Electron.Phi", "Electron.Charge",
        "Muon.PT", "Muon.Eta", "Muon.Phi", "Muon.Charge",
        "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass", "Jet.BTag",
        "MissingET.MET",
        "Particle.PID", "Particle.Status", "Particle.PT", "Particle.Eta", "Particle.Phi",
        "Particle.M1", "Particle.X", "Particle.Y", "Particle.Z",
        "EFlowTrack.PID", "EFlowTrack.Charge", "EFlowTrack.PT", "EFlowTrack.Eta",
        "EFlowTrack.Phi", "EFlowTrack.D0", "EFlowTrack.ErrorD0",
        "EFlowTrack.DZ", "EFlowTrack.ErrorDZ",
        "EFlowTrack.X", "EFlowTrack.Y", "EFlowTrack.Z",
    ])

    # --- lepton --------------------------------------------------------
    electron = ak.zip({
        "pt": ev["Electron.PT"], "eta": ev["Electron.Eta"], "phi": ev["Electron.Phi"],
        "q": ev["Electron.Charge"], "flavor": ak.ones_like(ev["Electron.PT"]) * 11})
    muon = ak.zip({
        "pt": ev["Muon.PT"], "eta": ev["Muon.Eta"], "phi": ev["Muon.Phi"],
        "q": ev["Muon.Charge"], "flavor": ak.ones_like(ev["Muon.PT"]) * 13})
    lepton = ak.concatenate([electron, muon], axis=1)

    # --- reco jet / MET ------------------------------------------------
    jet = ak.zip({
        "pt": ev["Jet.PT"], "eta": ev["Jet.Eta"], "phi": ev["Jet.Phi"],
        "mass": ev["Jet.Mass"], "b_tag": ev["Jet.BTag"]})
    met = ev["MissingET.MET"][:, 0]                 # 이벤트당 1개 → 평평한 배열

    # --- gen particle --------------------------------------------------
    gen = ak.zip({
        "pid": ev["Particle.PID"], "status": ev["Particle.Status"],
        "pt": ev["Particle.PT"], "eta": ev["Particle.Eta"], "phi": ev["Particle.Phi"],
        "m1": ev["Particle.M1"],
        "x": ev["Particle.X"], "y": ev["Particle.Y"], "z": ev["Particle.Z"]})

    # mother PID. M1 = -1 이면 mother 없음 (python [-1] 은 마지막 원소이므로 0 으로 바꾼 뒤 mask)
    has_mother = gen.m1 >= 0
    mother_pid = ak.mask(gen.pid[ak.where(has_mother, gen.m1, 0)], has_mother)

    # top decay quark: |pid| in {3,5}, mother = top, status 23 (첫 copy)
    from_top = ak.fill_none(abs(mother_pid) == 6, False)
    quark = gen[((abs(gen.pid) == 3) | (abs(gen.pid) == 5)) & from_top]
    quark = quark[quark.status == 23]

    # --- PF track ------------------------------------------------------
    # ErrorD0 = 0 인 더미 track (pt 0, eta 1e3) 이 있음 → 오차 inf 로 두면 significance 0 → cut 탈락
    d0, d0e = ev["EFlowTrack.D0"], ev["EFlowTrack.ErrorD0"]
    dz, dze = ev["EFlowTrack.DZ"], ev["EFlowTrack.ErrorDZ"]
    track = ak.zip({
        "pid": ev["EFlowTrack.PID"], "charge": ev["EFlowTrack.Charge"],
        "pt": ev["EFlowTrack.PT"], "eta": ev["EFlowTrack.Eta"], "phi": ev["EFlowTrack.Phi"],
        "d0": d0, "d0_err": d0e, "d0_sig": abs(d0) / ak.where(d0e > 0, d0e, np.inf),
        "dz": dz, "dz_err": dze, "dz_sig": abs(dz) / ak.where(dze > 0, dze, np.inf),
        # X, Y, Z 는 track 을 만든 입자의 "진짜" 생성 위치 (MC 전용, 분석에 쓰지 말 것)
        "x": ev["EFlowTrack.X"], "y": ev["EFlowTrack.Y"], "z": ev["EFlowTrack.Z"]})

    return {"ev": ev,
            "electron": electron, "muon": muon, "lepton": lepton,
            "jet": jet, "met": met,
            "gen": gen, "mother_pid": mother_pid, "quark": quark,
            "track": track}


# ---------------------------------------------------------------- selection
def select_dilepton(lepton, jet, met):
    """논문 Sec.2 dilepton selection.

    returns: mask(길이 N_event bool), good_jet(이벤트 수 유지, jet 단위 거름),
             inv_mass(길이 N_event, None 가능), cutflow(dict)
    """
    lepton = lepton[ak.argsort(lepton.pt, axis=1, ascending=False)]
    lepton = ak.pad_none(lepton, 2, axis=1)

    lead_ok = ak.fill_none((lepton.pt[:, 0] > LEAD_PT) & (abs(lepton.eta[:, 0]) < LEP_ETA), False)
    sub_ok  = ak.fill_none((lepton.pt[:, 1] > SUB_PT)  & (abs(lepton.eta[:, 1]) < LEP_ETA), False)

    inv_mass = pair_mass(lepton[:, 0], lepton[:, 1])
    mll_ok = ak.fill_none(inv_mass > MLL_MIN, False)

    # Z veto 는 same-flavour 쌍에만 (저자 확인 목록)
    same_flav = ak.fill_none(lepton.flavor[:, 0] == lepton.flavor[:, 1], False)
    zveto_ok = ~same_flav | ak.fill_none(abs(inv_mass - Z_MASS) > Z_WINDOW, False)

    met_ok = met > MET_MIN

    good_jet = jet[(jet.pt > JET_PT) & (abs(jet.eta) < JET_ETA)]
    jet_ok = ak.num(good_jet) >= 2

    cutflow, acc = {}, ak.ones_like(lead_ok, dtype=bool)
    for name, m in [("lead", lead_ok), ("sub", sub_ok), ("mll", mll_ok),
                    ("zveto", zveto_ok), ("met", met_ok), ("jet", jet_ok)]:
        acc = acc & m
        cutflow[name] = int(ak.sum(acc))
    return acc, good_jet, inv_mass, cutflow


def select_tracks(track):
    """K0S daughter 후보 track (논문 Sec.2). 객체 단위 거르기 → 이벤트 수 유지.

    returns: track_kin (significance cut 전, 진단용), track_sel (최종)
    """
    charged = (abs(track.pid) != 11) & (abs(track.pid) != 13)
    kin = (track.pt > TRK_PT) & (abs(track.eta) < TRK_ETA)
    track_kin = track[charged & kin]
    track_sel = track_kin[track_kin.d0_sig > TRK_D0SIG]
    return track_kin, track_sel


# ---------------------------------------------------------------- matching
def match_primary(quark, good_jet, dr_max=0.4):
    """quark 마다 ΔR<dr_max 인 jet 중 pT 최대 → primary jet.

    returns: 레코드 (이벤트 → quark → {quark, jet}); jet 은 매칭 실패 시 None.
    """
    pairs = ak.cartesian({"quark": quark, "jet": good_jet}, axis=1, nested=True)
    dr = delta_r(pairs.jet.eta, pairs.jet.phi, pairs.quark.eta, pairs.quark.phi)
    pt_close = ak.mask(pairs.jet.pt, dr < dr_max)
    best = ak.argmax(pt_close, axis=2, keepdims=True)
    return ak.zip({"quark": quark, "jet": ak.firsts(pairs.jet[best], axis=2)})


def match_track(pion, tracks, dr_max=0.2, deta_max=0.01):
    """[MC 전용] gen pion 각각에 가장 가까운 reco track. 없으면 None. 구조는 pion 과 같음.

    Δeta 조건은 displaced track 의 phi 가 bending 으로 틀어지는 것을 고려한 것
    (eta 는 자기장 영향이 없으므로 깨끗한 매칭 기준).
    """
    m = ak.cartesian({"pi": pion, "trk": tracks}, axis=1, nested=True)
    dr = delta_r(m.pi.eta, m.pi.phi, m.trk.eta, m.trk.phi)
    good = (dr < dr_max) & (abs(m.pi.eta - m.trk.eta) < deta_max)
    best = ak.argmin(ak.mask(dr, good), axis=2, keepdims=True)
    return ak.firsts(m.trk[best], axis=2)


def true_ks_pairs(gen, mother_pid, track_sel):
    """[MC 전용] gen K0S→π+π- 중 두 pion 모두 kinematic cut 통과 + 두 track 모두 재구성된 것.

    returns: pair_gen(모든 gen 쌍), pair_gen_kin(kin cut 후), true_pair({a, b, vx, vy})
             a, b 는 reco track, (vx, vy) 는 truth decay vertex [mm]
    """
    from_ks = ak.fill_none(mother_pid == 310, False)
    pion_gen = gen[(abs(gen.pid) == 211) & from_ks]
    pair_gen = ak.combinations(pion_gen, 2, fields=["a", "b"], axis=1)
    pair_gen = pair_gen[(pair_gen.a.m1 == pair_gen.b.m1) & (pair_gen.a.pid * pair_gen.b.pid < 0)]

    kin = ((pair_gen.a.pt > TRK_PT) & (pair_gen.b.pt > TRK_PT) &
           (abs(pair_gen.a.eta) < TRK_ETA) & (abs(pair_gen.b.eta) < TRK_ETA))
    pair_gen_kin = pair_gen[kin]

    trk_a = match_track(pair_gen_kin.a, track_sel)
    trk_b = match_track(pair_gen_kin.b, track_sel)
    both = ~ak.is_none(trk_a, axis=1) & ~ak.is_none(trk_b, axis=1)
    trk_a, trk_b, pg = trk_a[both], trk_b[both], pair_gen_kin[both]
    # 두 pion 이 같은 track 에 매칭된 경우 (collinear) 제거
    distinct = ~((trk_a.pt == trk_b.pt) & (trk_a.phi == trk_b.phi))
    true_pair = ak.zip({"a": trk_a[distinct], "b": trk_b[distinct],
                        "vx": pg.a.x[distinct], "vy": pg.a.y[distinct]})
    return pair_gen, pair_gen_kin, true_pair


# ---------------------------------------------------------------- 유틸
def count(name, x):
    """이벤트당 객체 수 요약 출력."""
    n = ak.num(x, axis=1)
    print(f"{name:14s} total={int(ak.sum(n)):7d}  mean/event={float(ak.mean(n)):7.2f}  max={int(ak.max(n))}")


def flat(x):
    """awkward jagged → numpy 1D (히스토그램용). None 은 제거."""
    return ak.to_numpy(ak.flatten(ak.drop_none(x)))
