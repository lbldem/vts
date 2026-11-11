"""
common.py — |Vts| 분석 공용 함수 (Delphes ROOT → awkward)

지금까지 합의된 규칙
  * 모든 "이벤트 단위" mask는 길이 N_event 의 boolean. 적용은 호출자가 마지막에 한 번만.
  * "객체 단위" 거르기 (jet[...], track[...]) 는 이벤트 수를 유지한 채 안쪽 리스트만 줄인다.
  * 변수 이름: reco 객체(jet, track)와 gen 객체(gen, quark, pion)는 섞어 쓰지 않는다.
"""
import awkward as ak
import numpy as np

PION_MASS = 0.13957      # GeV
KS_MASS   = 0.497611     # GeV
Z_MASS    = 91.1876      # GeV

LEAD_PT, SUB_PT, LEP_ETA = 25.0, 20.0, 2.4
MLL_MIN, Z_WINDOW, MET_MIN = 20.0, 15.0, 40.0
JET_PT, JET_ETA = 30.0, 2.4


def four_vector(obj, mass=0.0):
    """(pt, eta, phi) 레코드 → (E, px, py, pz). jagged 구조 유지."""
    px = obj.pt * np.cos(obj.phi)
    py = obj.pt * np.sin(obj.phi)
    pz = obj.pt * np.sinh(obj.eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return E, px, py, pz


def pair_mass(a, b, mass_a=0.0, mass_b=0.0):
    """두 객체(레코드)의 invariant mass. 4-vector를 먼저 더한 뒤 제곱."""
    Ea, pxa, pya, pza = four_vector(a, mass_a)
    Eb, pxb, pyb, pzb = four_vector(b, mass_b)
    m2 = (Ea + Eb)**2 - (pxa + pxb)**2 - (pya + pyb)**2 - (pza + pzb)**2
    return np.sqrt(m2)


def delta_phi(phi1, phi2):
    """phi 차이를 [-pi, pi] 로 접음."""
    dphi = phi1 - phi2
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)


def load_objects(tree):
    """Delphes tree → dict of awkward 레코드 배열 (전부 길이 N_event).

    keys: electron, muon, lepton, jet, met, gen, mother_pid, quark, track
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
        "EFlowTrack.Xd", "EFlowTrack.Yd", "EFlowTrack.Zd",
    ])

    # --- lepton ---------------------------------------------------------
    electron = ak.zip({
        "pt": ev["Electron.PT"], "eta": ev["Electron.Eta"], "phi": ev["Electron.Phi"],
        "q": ev["Electron.Charge"],
        "flavor": ak.ones_like(ev["Electron.PT"]) * 11})
    muon = ak.zip({
        "pt": ev["Muon.PT"], "eta": ev["Muon.Eta"], "phi": ev["Muon.Phi"],
        "q": ev["Muon.Charge"],
        "flavor": ak.ones_like(ev["Muon.PT"]) * 13})
    lepton = ak.concatenate([electron, muon], axis=1)

    # --- reco jet / MET -------------------------------------------------
    jet = ak.zip({
        "pt": ev["Jet.PT"], "eta": ev["Jet.Eta"], "phi": ev["Jet.Phi"],
        "mass": ev["Jet.Mass"], "b_tag": ev["Jet.BTag"]})
    met = ev["MissingET.MET"][:, 0]          # 이벤트당 1개 → 평평한 배열

    # --- gen particle ---------------------------------------------------
    gen = ak.zip({
        "pid": ev["Particle.PID"], "status": ev["Particle.Status"],
        "pt": ev["Particle.PT"], "eta": ev["Particle.Eta"], "phi": ev["Particle.Phi"],
        "m1": ev["Particle.M1"],
        "x": ev["Particle.X"], "y": ev["Particle.Y"], "z": ev["Particle.Z"]})

    # mother PID: M1 은 같은 이벤트 안의 index. -1 이면 mother 없음.
    # 주의: python 의 [-1] 은 "마지막 원소" 이므로 먼저 0 으로 바꾼 뒤 mask.
    has_mother = gen.m1 >= 0
    safe_m1 = ak.where(has_mother, gen.m1, 0)
    mother_pid = ak.mask(gen.pid[safe_m1], has_mother)     # None = mother 없음

    # top decay quark: |pid| in {3,5}, mother = top, status 23 (첫 copy)
    #   from_top 만으로도 첫 copy 만 잡히지만 (이후 copy 의 M1 은 quark 자신) 명시적으로 둠.
    from_top = ak.fill_none(abs(mother_pid) == 6, False)
    quark = gen[((abs(gen.pid) == 3) | (abs(gen.pid) == 5)) & from_top]
    quark = quark[quark.status == 23]

    # --- PF track (K0S 재구성용) ---------------------------------------
    #   d0_err = ErrorD0 (오차). significance 는 나중에 d0_sig 로 따로 붙인다.
    track = ak.zip({
        "pid": ev["EFlowTrack.PID"], "charge": ev["EFlowTrack.Charge"],
        "pt": ev["EFlowTrack.PT"], "eta": ev["EFlowTrack.Eta"], "phi": ev["EFlowTrack.Phi"],
        "d0": ev["EFlowTrack.D0"], "d0_err": ev["EFlowTrack.ErrorD0"],
        "dz": ev["EFlowTrack.DZ"], "dz_err": ev["EFlowTrack.ErrorDZ"],
        "x": ev["EFlowTrack.X"], "y": ev["EFlowTrack.Y"], "z": ev["EFlowTrack.Z"],
        "xd": ev["EFlowTrack.Xd"], "yd": ev["EFlowTrack.Yd"], "zd": ev["EFlowTrack.Zd"]})

    return {"electron": electron, "muon": muon, "lepton": lepton,
            "jet": jet, "met": met,
            "gen": gen, "mother_pid": mother_pid, "quark": quark,
            "track": track}


def select_dilepton(lepton, jet, met):
    """논문 Sec.2 dilepton selection.

    returns: mask(길이 N_event bool), good_jet(이벤트 수 유지, jet 단위 거름),
             inv_mass(길이 N_event, None 가능), cutflow(dict)
    """
    order = ak.argsort(lepton.pt, axis=1, ascending=False)
    lepton = lepton[order]
    lepton = ak.pad_none(lepton, 2, axis=1)              # lepton < 2 인 이벤트는 None 으로 채움

    lead_ok = ak.fill_none((lepton.pt[:, 0] > LEAD_PT) & (abs(lepton.eta[:, 0]) < LEP_ETA), False)
    sub_ok  = ak.fill_none((lepton.pt[:, 1] > SUB_PT)  & (abs(lepton.eta[:, 1]) < LEP_ETA), False)

    inv_mass = pair_mass(lepton[:, 0], lepton[:, 1])      # lepton 질량 무시 (효과 < MeV)
    mll_ok = ak.fill_none(inv_mass > MLL_MIN, False)

    # Z veto 는 same-flavour 쌍에만 (eμ 에는 Z peak 가 없음). 저자 확인 목록 항목.
    same_flav = ak.fill_none(lepton.flavor[:, 0] == lepton.flavor[:, 1], False)
    zveto_ok = ~same_flav | ak.fill_none(abs(inv_mass - Z_MASS) > Z_WINDOW, False)

    met_ok = met > MET_MIN

    # jet 단위 거르기 → 이벤트 단위 조건
    good_jet = jet[(jet.pt > JET_PT) & (abs(jet.eta) < JET_ETA)]
    jet_ok = ak.num(good_jet) >= 2

    cutflow = {}
    acc = ak.ones_like(lead_ok, dtype=bool)
    for name, m in [("lead", lead_ok), ("sub", sub_ok), ("mll", mll_ok),
                    ("zveto", zveto_ok), ("met", met_ok), ("jet", jet_ok)]:
        acc = acc & m
        cutflow[name] = int(ak.sum(acc))
    mask = acc
    return mask, good_jet, inv_mass, cutflow


def match_primary(quark, good_jet, dr_max=0.4):
    """quark 마다 ΔR<dr_max 인 jet 중 pT 최대 → primary jet.

    returns: match 레코드 (이벤트 → quark → {quark, jet}), jet 은 매칭 실패 시 None.
    """
    pairs = ak.cartesian({"quark": quark, "jet": good_jet}, axis=1, nested=True)  # quark → (jet...)
    dr = delta_r(pairs.jet.eta, pairs.jet.phi, pairs.quark.eta, pairs.quark.phi)
    close = dr < dr_max
    pt_close = ak.mask(pairs.jet.pt, close)                   # 멀면 None (구조 유지)
    best = ak.argmax(pt_close, axis=2, keepdims=True)         # quark 당 [index] 또는 [None]
    primary = ak.firsts(pairs.jet[best], axis=2)              # [jet] → jet
    return ak.zip({"quark": quark, "jet": primary})


def count(name, x):
    """이벤트당 객체 수 요약 출력."""
    n = ak.num(x, axis=1)
    print(f"{name:14s} total={int(ak.sum(n)):7d}  mean/event={float(ak.mean(n)):7.2f}  max={int(ak.max(n))}")
