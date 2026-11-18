"""
skim.py  <입력 Delphes ROOT>  <출력 skim ROOT>

Delphes 출력에서 DiSaJa 분석에 필요한 것만 골라 NanoAOD 식 이름으로 저장한다.
  - Events tree: 느슨한 사전 선택(lepton >= 2, jet >= 2)을 통과한 이벤트
  - Runs tree  : skim 전 전체 이벤트 수와 weight 합 (normalization 용)
"""
import sys
import uproot
import awkward as ak
import numpy as np

# Delphes 파일에서 읽을 branch 이름 목록
BRANCHES = [
    # jet
    "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass", "Jet.BTag", "Jet.Charge", "Jet.Constituents",
    # jet constituent: track
    "EFlowTrack.fUniqueID", "EFlowTrack.PT", "EFlowTrack.Eta", "EFlowTrack.Phi",
    "EFlowTrack.D0", "EFlowTrack.DZ", "EFlowTrack.Charge", "EFlowTrack.PID",
    # jet constituent: tower (photon, neutral hadron)
    "EFlowPhoton.fUniqueID", "EFlowPhoton.ET", "EFlowPhoton.Eta", "EFlowPhoton.Phi",
    "EFlowPhoton.Eem", "EFlowPhoton.Ehad",
    "EFlowNeutralHadron.fUniqueID", "EFlowNeutralHadron.ET", "EFlowNeutralHadron.Eta",
    "EFlowNeutralHadron.Phi", "EFlowNeutralHadron.Eem", "EFlowNeutralHadron.Ehad",
    # lepton
    "Electron.PT", "Electron.Eta", "Electron.Phi", "Electron.Charge", "Electron.IsolationVar",
    "Muon.PT", "Muon.Eta", "Muon.Phi", "Muon.Charge", "Muon.IsolationVar",
    # MET
    "MissingET.MET", "MissingET.Phi",
    # gen
    "Particle.PID", "Particle.M1", "Particle.PT", "Particle.Eta", "Particle.Phi",
    # event weight
    "Event.Weight",
]


def read_delphes(path):
    """Delphes ROOT 파일 하나를 열어 BRANCHES에 적힌 branch만 읽어서 돌려준다."""
    tree = uproot.open(path)["Delphes"]
    return tree.arrays(BRANCHES)


def find_gen_quarks(d):
    """이벤트마다 top의 딸인 quark을 찾아 pid, pt, eta, phi를 묶어서 돌려준다."""
    pid = d["Particle.PID"]
    m1 = d["Particle.M1"]

    mother_pid = pid[m1]                               # 각 입자의 mother의 PID
    is_quark = abs(pid) <= 5                           # d, u, s, c, b 까지
    from_top = (m1 >= 0) & (abs(mother_pid) == 6)      # m1 = -1 (mother 없음)은 제외
    mask = is_quark & from_top

    return ak.zip({
        "pid": pid[mask],
        "pt":  d["Particle.PT"][mask],
        "eta": d["Particle.Eta"][mask],
        "phi": d["Particle.Phi"][mask],
    })


def preselect(d):
    """느슨한 사전 선택. lepton 2개 이상, jet 2개 이상인 이벤트를 True로 표시한 mask를 돌려준다."""
    good_el = (d["Electron.PT"] > 20) & (abs(d["Electron.Eta"]) < 2.4)
    good_mu = (d["Muon.PT"] > 20) & (abs(d["Muon.Eta"]) < 2.4)
    good_jet = (d["Jet.PT"] > 30) & (abs(d["Jet.Eta"]) < 2.4)

    n_lep = ak.sum(good_el, axis=1) + ak.sum(good_mu, axis=1)
    n_jet = ak.sum(good_jet, axis=1)

    return (n_lep >= 2) & (n_jet >= 2)


def jet_index(obj_uid, jet_refs):
    """객체마다 자신이 속한 jet의 번호를 돌려준다. 어느 jet에도 없으면 -1."""
    out = []
    for uids, refs_per_jet in zip(obj_uid.tolist(), jet_refs.tolist()):   # 이벤트 하나씩
        owner = {}                                    # 사전: fUniqueID -> 그 객체가 속한 jet의 번호
        for j, refs in enumerate(refs_per_jet):       # j번째 jet과 그 명단
            for r in refs:
                owner[r] = j
        out.append([owner.get(u, -1) for u in uids])
    return ak.Array(out)


def build_output(d, q):
    """저장할 내용을 '객체 이름 -> array' 형태의 dictionary로 묶어 돌려준다."""
    jet_refs = d["Jet.Constituents"]["refs"]

    def tower(name):
        # photon과 neutral hadron을 이벤트별로 이어 붙여 하나의 tower 목록으로 만든다
        return ak.concatenate(
            [d["EFlowPhoton." + name], d["EFlowNeutralHadron." + name]], axis=1)

    # tower의 jet 번호도 같은 순서(photon 먼저, neutral hadron 다음)로 이어 붙인다
    tower_jet_idx = ak.concatenate(
        [jet_index(d["EFlowPhoton.fUniqueID"], jet_refs),
         jet_index(d["EFlowNeutralHadron.fUniqueID"], jet_refs)], axis=1)

    return {
        "Jet": ak.zip({
            "pt": d["Jet.PT"], "eta": d["Jet.Eta"], "phi": d["Jet.Phi"],
            "mass": d["Jet.Mass"], "btag": d["Jet.BTag"], "charge": d["Jet.Charge"],
        }),
        "Track": ak.zip({
            "pt": d["EFlowTrack.PT"], "eta": d["EFlowTrack.Eta"], "phi": d["EFlowTrack.Phi"],
            "d0": d["EFlowTrack.D0"], "dz": d["EFlowTrack.DZ"],
            "charge": d["EFlowTrack.Charge"], "pid": d["EFlowTrack.PID"],
            "jetIdx": jet_index(d["EFlowTrack.fUniqueID"], jet_refs),
        }),
        "Tower": ak.zip({
            "et": tower("ET"), "eta": tower("Eta"), "phi": tower("Phi"),
            "eem": tower("Eem"), "ehad": tower("Ehad"),
            "jetIdx": tower_jet_idx,
        }),
        "Electron": ak.zip({
            "pt": d["Electron.PT"], "eta": d["Electron.Eta"], "phi": d["Electron.Phi"],
            "charge": d["Electron.Charge"], "iso": d["Electron.IsolationVar"],
        }),
        "Muon": ak.zip({
            "pt": d["Muon.PT"], "eta": d["Muon.Eta"], "phi": d["Muon.Phi"],
            "charge": d["Muon.Charge"], "iso": d["Muon.IsolationVar"],
        }),
        "GenQuark": q,
        "MET_pt": d["MissingET.MET"][:, 0],
        "MET_phi": d["MissingET.Phi"][:, 0],
        "genWeight": d["Event.Weight"][:, 0],
    }


def write_root(out, runs, path):
    """Events tree(이벤트별 내용)와 Runs tree(skim 전 전체 통계)를 저장한다."""
    with uproot.recreate(path) as f:
        f["Events"] = out
        f["Runs"] = runs


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    d = read_delphes(in_path)
    q = find_gen_quarks(d)
    keep = preselect(d)

    # skim 전 전체 통계: 반드시 이벤트를 걸러내기 전에 계산
    n_total = len(d)
    sum_w = float(ak.sum(d["Event.Weight"][:, 0]))
    runs = {
        "genEventCount": np.array([n_total]),
        "genEventSumw": np.array([sum_w]),
    }

    d = d[keep]
    q = q[keep]
    write_root(build_output(d, q), runs, out_path)
    print(f"skim: kept {len(d)} of {n_total} events -> {out_path}")

    # 방금 쓴 파일을 다시 읽어서 확인
    f = uproot.open(out_path)
    ev = f["Events"]
    r = f["Runs"]
    w = ev["genWeight"].array()
    print("entries:", ev.num_entries)
    print("genEventCount:", r["genEventCount"].array()[0], " genEventSumw:", r["genEventSumw"].array()[0])
    print("kept: positive", ak.sum(w > 0), " negative", ak.sum(w < 0), " sum", ak.sum(w))
    trk = ev["Track_jetIdx"].array()
    tow = ev["Tower_jetIdx"].array()
    print("event 0, jet 0 constituents:", ak.sum(trk[0] == 0) + ak.sum(tow[0] == 0))
