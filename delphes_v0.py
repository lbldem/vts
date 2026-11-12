"""
delphes_v0.py — Delphes 전용 K0S (V0) 재구성

Delphes 는 track 을 perigee 기준 helix parameter 로만 주고 vertex fit 을 하지 않으므로,
decay vertex 와 "그 점에서의 운동량 방향" 을 여기서 직접 만든다. 실제 CMS 데이터에서는
공식 V0 collection (vertex fit 포함) 을 쓰므로 이 파일은 통째로 교체된다.

검증된 규약 (sig_a_test.root, truth 비교로 확정)
  * 양전하 track 은 B_z > 0 에서 시계방향 (원 중심이 진행 방향의 오른쪽)
  * Delphes d0 부호: d0 = (x·py − y·px)/pT  →  perigee P = d0·(sin φ, −cos φ)
  * phi_at(perigee) == track.phi  (검산 1e-8)
  * forward_arc 규칙의 vertex 선택 정답률 96%

흐름 (쌍 단위, 구조 이벤트 → 쌍):
  pair ──find_vertex──► v1, v2, h2, dca2d
       ──choose_vertex─► (vx, vy)
       ──build_ks──────► K0S 후보 레코드 (mass, pt, eta, phi, lxy, cos_a, dca, daughter 변수)
  ks + jet ──attach_to_jets──► jet 마다 대표 후보 하나
"""
import awkward as ak
import numpy as np
from common import four_vector, pair_mass, delta_r, PION_MASS

B_FIELD = 3.8          # Tesla (Delphes CMS card 의 Bz)


# ---------------------------------------------------------------- track → 원
def track_circle(trk, B=B_FIELD):
    """transverse plane 에서 track 의 원: 반경 R [mm], 중심 (cx, cy) [mm]."""
    R = trk.pt / (0.3 * B) * 1000.0
    perx = trk.d0 * np.sin(trk.phi)                 # perigee 점
    pery = -trk.d0 * np.cos(trk.phi)
    cx = perx + trk.charge * R * np.sin(trk.phi)    # 진행 방향의 오른쪽(q>0)/왼쪽(q<0)
    cy = pery - trk.charge * R * np.cos(trk.phi)
    return R, cx, cy


def perigee(trk):
    return trk.d0 * np.sin(trk.phi), -trk.d0 * np.cos(trk.phi)


# ---------------------------------------------------------------- vertex
def find_vertex(a, b):
    """두 track 의 원이 만나는 점 (2개) 또는 안 만나면 최근접점의 중점.

    returns 레코드 {v1x, v1y, v2x, v2y, h2, dca2d, valid}
      h2 >= 0 : 만남, v1 ≠ v2      h2 < 0 : 안 만남, v1 == v2 == 최근접점 중점
      dca2d   : 2D 에서 두 원 사이의 틈 (만나면 0)
      valid   : d ≈ 0 (같은 track) 이면 False
    """
    Ra, cxa, cya = track_circle(a)
    Rb, cxb, cyb = track_circle(b)
    dx, dy = cxb - cxa, cyb - cya
    d_raw = np.sqrt(dx**2 + dy**2)
    valid = d_raw > 1e-6
    d = ak.where(valid, d_raw, 1.0)                 # 0 나누기 방지; 결과는 valid 로 거른다
    ux, uy = dx / d, dy / d                         # 중심선 방향
    nx, ny = -uy, ux                                # 중심선에 수직

    # Case 1: 만남 — 공통현의 발 M 과 반현 h
    l = (d**2 + Ra**2 - Rb**2) / (2 * d)
    h2 = Ra**2 - l**2
    h = np.sqrt(np.maximum(h2, 0))
    mx, my = cxa + l * ux, cya + l * uy

    # Case 2/3: 안 만남 — 최근접점 Pa, Pb 와 그 중점
    inter = h2 >= 0
    outside = d > Ra + Rb
    sa = ak.where(outside,  1, ak.where(Ra > Rb,  1, -1))
    sb = ak.where(outside, -1, ak.where(Ra > Rb,  1, -1))
    pax, pay = cxa + sa * Ra * ux, cya + sa * Ra * uy
    pbx, pby = cxb + sb * Rb * ux, cyb + sb * Rb * uy
    mcx, mcy = (pax + pbx) / 2, (pay + pby) / 2
    dca2d = ak.where(inter, 0.0, np.sqrt((pax - pbx)**2 + (pay - pby)**2))

    return ak.zip({
        "v1x": ak.where(inter, mx + h * nx, mcx), "v1y": ak.where(inter, my + h * ny, mcy),
        "v2x": ak.where(inter, mx - h * nx, mcx), "v2y": ak.where(inter, my - h * ny, mcy),
        "h2": h2, "dca2d": dca2d, "valid": valid})


def forward_arc(trk, vx, vy):
    """perigee 에서 (vx, vy) 까지 운동 방향으로 전진한 호의 각 [0, 2π)."""
    _, cx, cy = track_circle(trk)
    perx, pery = perigee(trk)
    th_p = np.arctan2(pery - cy, perx - cx)
    th_v = np.arctan2(vy - cy, vx - cx)
    fwd = ak.where(trk.charge > 0, th_p - th_v, th_v - th_p)   # q>0: 시계방향 = 각 감소
    return fwd % (2 * np.pi)


def choose_vertex(pair, v):
    """v1, v2 중 두 track 모두 perigee 에서 '앞으로' 조금만 가면 닿는 쪽을 고른다."""
    s1 = forward_arc(pair.a, v.v1x, v.v1y) + forward_arc(pair.b, v.v1x, v.v1y)
    s2 = forward_arc(pair.a, v.v2x, v.v2y) + forward_arc(pair.b, v.v2x, v.v2y)
    pick1 = s1 < s2
    return ak.where(pick1, v.v1x, v.v2x), ak.where(pick1, v.v1y, v.v2y), pick1


def phi_at(trk, vx, vy):
    """원 위의 점 (vx, vy) 에서의 접선(운동량) 방향 φ. perigee 를 넣으면 trk.phi 가 나와야 함."""
    _, cx, cy = track_circle(trk)
    rx, ry = vx - cx, vy - cy
    return np.arctan2(-trk.charge * rx, trk.charge * ry)


# ---------------------------------------------------------------- K0S 후보
def build_ks(pair, vx, vy, dca):
    """track 쌍 + vertex → K0S 후보 레코드 (쌍과 같은 구조).

    daughter 운동량은 vertex 에서의 φ 로 다시 읽은 것 (perigee φ 가 아님).
    """
    a = ak.with_field(pair.a, phi_at(pair.a, vx, vy), "phi")
    b = ak.with_field(pair.b, phi_at(pair.b, vx, vy), "phi")
    Ea, pxa, pya, pza = four_vector(a, PION_MASS)
    Eb, pxb, pyb, pzb = four_vector(b, PION_MASS)
    px, py, pz = pxa + pxb, pya + pyb, pza + pzb
    pt = np.sqrt(px**2 + py**2)
    lxy = np.sqrt(vx**2 + vy**2)
    return ak.zip({
        "mass": pair_mass(a, b, PION_MASS, PION_MASS),
        "pt": pt, "eta": np.arcsinh(pz / pt), "phi": np.arctan2(py, px),
        "vx": vx, "vy": vy, "lxy": lxy,
        "cos_a": (vx * px + vy * py) / (lxy * pt),          # pointing angle
        "dca": dca,
        "frac_a": a.pt / pt, "frac_b": b.pt / pt,           # daughter pT fraction
        "d0sig_a": a.d0_sig, "d0sig_b": b.d0_sig,
        "dzsig_a": a.dz_sig, "dzsig_b": b.dz_sig,
        # 어느 track 쌍이었는지 되찾기 위한 꼬리표 (truth label 붙일 때 사용)
        "pt_a": pair.a.pt, "phi_a": pair.a.phi, "pt_b": pair.b.pt, "phi_b": pair.b.phi})


def reconstruct_v0(pair):
    """pair → (ks, pair_valid, vertex). 같은 함수를 pair_reco 와 true_pair 양쪽에 쓴다."""
    v = find_vertex(pair.a, pair.b)
    keep = v.valid
    v, pair = v[keep], pair[keep]
    vx, vy, _ = choose_vertex(pair, v)
    return build_ks(pair, vx, vy, v.dca2d), pair, v


def attach_to_jets(ks, jets, dr_max=0.4):
    """jet 마다 ΔR<dr_max 안의 후보 중 pT 최대 하나 (없으면 None). 구조는 jets 와 같음."""
    kj = ak.cartesian({"jet": jets, "ks": ks}, axis=1, nested=True)
    dr = delta_r(kj.jet.eta, kj.jet.phi, kj.ks.eta, kj.ks.phi)
    pt_close = ak.mask(kj.ks.pt, dr < dr_max)
    best = ak.argmax(pt_close, axis=2, keepdims=True)
    return ak.firsts(kj.ks[best], axis=2)
