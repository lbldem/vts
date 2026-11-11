import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

t = uproot.open("sig_a_test.root")["Delphes"]

ev = t.arrays(["Electron.PT", "Electron.Eta", "Electron.Phi", "Electron.Charge",
    "Muon.PT", "Muon.Eta", "Muon.Phi", "Muon.Charge",
    "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.Mass", "GenJet.BTag",
    "MissingET.MET"])

e_flav = ak.ones_like(ev["Electron.PT"]) * 11
mu_flav = ak.ones_like(ev["Muon.PT"]) * 13

electron = ak.zip({
    "pt": ev["Electron.PT"],
    "eta": ev["Electron.Eta"],
    "phi": ev["Electron.Phi"],
    "q": ev["Electron.Charge"],
    "flavor": e_flav})

muon = ak.zip({
    "pt": ev["Muon.PT"],
    "eta": ev["Muon.Eta"],
    "phi": ev["Muon.Phi"],
    "q": ev["Muon.Charge"],
    "flavor": mu_flav})

Genjet = ak.zip({
    "pt": ev["GenJet.PT"],
    "eta": ev["GenJet.Eta"],
    "phi": ev["GenJet.Phi"],
    "mass": ev["GenJet.Mass"],
    "b_tag": ev["GenJet.BTag"]})

met = ev["MissingET.MET"][:, 0]                 # Do not make a record here!

lepton = ak.concatenate([electron, muon], axis=1)
order = ak.argsort(lepton.pt, axis=1, ascending=False)
lepton = lepton[order]

lepton = ak.pad_none(lepton, 2, axis=1)         # This fills up 1 lepton events with a None

lead_ok = ak.fill_none((lepton.pt[:, 0] > 25) & (abs(lepton.eta[:, 0]) < 2.4), False)     # still a mask
print("lead_ok", ak.sum(lead_ok))
sub_ok = ak.fill_none((lepton.pt[:, 1] > 20) & (abs(lepton.eta[:, 1]) < 2.4), False)      # still a mask
print("sub_ok", ak.sum(lead_ok & sub_ok))

def four_vector(lepton):
    px = lepton.pt * np.cos(lepton.phi)
    py = lepton.pt * np.sin(lepton.phi)
    pz = lepton.pt * np.sinh(lepton.eta)
    E = np.sqrt(px**2 + py**2 + pz**2)
    return E, px, py, pz

def delta_r(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    return np.sqrt((eta1- eta2)**2 + dphi**2)


E1, px1, py1, pz1 = four_vector(lepton[:, 0])
E2, px2, py2, pz2 = four_vector(lepton[:, 1])
inv_mass = np.sqrt((E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)
mll_ok = ak.fill_none(inv_mass > 20, False)                                             # still a mask
print("mll_ok", ak.sum(lead_ok & sub_ok & mll_ok))

same_flav = ak.fill_none(lepton.flavor[:, 0] == lepton.flavor[:, 1], False)
zveto_ok = ~same_flav | ak.fill_none(abs(inv_mass - 91.19) > 15, False)
# zveto_ok = ak.fill_none(abs(inv_mass - 91.19) > 15, False)                              # still a mask
print("zveto_ok", ak.sum(lead_ok & sub_ok & mll_ok & zveto_ok))
met_ok = ak.fill_none(met > 40, False)
print("met_ok", ak.sum(lead_ok & sub_ok & mll_ok & zveto_ok & met_ok))

good_jet = Genjet[(Genjet.pt > 30) & (abs(Genjet.eta) < 2.4)]
lj = ak.cartesian({"jet": good_jet, "lep": lepton[:, :2]}, nested=True)   # jet → (lep0, lep1)
dr_lj = delta_r(lj.jet.eta, lj.jet.phi, lj.lep.eta, lj.lep.phi)
far = ak.all(dr_lj > 0.4, axis=2)      # jet마다 "두 lepton 모두에서 멀다"
good_jet = good_jet[far]
jet_ok = ak.num(good_jet) >= 2
print("jet_ok", ak.sum(lead_ok & sub_ok & mll_ok & zveto_ok & met_ok & jet_ok))
mask = lead_ok & sub_ok & mll_ok & zveto_ok & met_ok & jet_ok
lepton = lepton[mask]

particle = t.arrays(["Particle.PID", "Particle.Status", "Particle.PT", "Particle.Eta", "Particle.Phi"])

gen = ak.zip({"pid": particle["Particle.PID"],
              "status": particle["Particle.Status"],
              "pt": particle["Particle.PT"],
              "eta": particle["Particle.Eta"],
              "phi": particle["Particle.Phi"]})

quark = gen[(abs(gen.pid) == 3) | (abs(gen.pid) == 5)]
quark = quark[quark.status == 23]

pairs = ak.cartesian({"quark": quark, "jet": good_jet}, axis=1, nested=True)    # Connecting quarks and jets >> quark --> (jet i, jet j, ...) from nested=True
dr = delta_r(pairs.jet.eta, pairs.jet.phi, pairs.quark.eta, pairs.quark.phi)    # dr cone of a jet and a quark
close = dr < 0.4                                                                # Closeness T/F list of event --> quark --> jet structure
pt_close = ak.mask(pairs.jet.pt, close)                                         # Masking jet_pT with dr < 0.4, leaving Falses as none.
best = ak.argmax(pt_close, axis=2, keepdims=True)                               # Taking indices of max pT among jets close to a quark. None's ignored.
primary = ak.firsts(pairs.jet[best], axis=2)                                    # Taking the best jet's record out. >> The primary jet.
n_primary = ak.sum(~ak.is_none(primary, axis=1), axis=1)                        # Counting not None elements of primary array.

print(Genjet[0][0], Genjet[0][1])
print(np.unique(ak.to_numpy(n_primary), return_counts=True))
print(np.unique(ak.to_numpy(n_primary[mask]), return_counts=True))