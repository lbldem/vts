import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

t = uproot.open("sig_a_test.root")["Delphes"]

ev = t.arrays(["Electron.PT", "Electron.Eta", "Electron.Phi", "Electron.Charge",
    "Muon.PT", "Muon.Eta", "Muon.Phi", "Muon.Charge",
    "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass", "Jet.BTag",
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

jet = ak.zip({
    "pt": ev["Jet.PT"],
    "eta": ev["Jet.Eta"],
    "phi": ev["Jet.Phi"],
    "mass": ev["Jet.Mass"],
    "b_tag": ev["Jet.BTag"]})

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

good_jet = jet[(jet.pt > 30) & (abs(jet.eta) < 2.4)]
jet_ok = ak.num(good_jet) >= 2
print("jet_ok", ak.sum(lead_ok & sub_ok & mll_ok & zveto_ok & met_ok & jet_ok))
mask = lead_ok & sub_ok & mll_ok & zveto_ok & met_ok & jet_ok
lepton = lepton[mask]

def delta_r(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    return np.sqrt((eta1- eta2)**2 + dphi**2)

particle = t.arrays(["Particle.PID", "Particle.Status", "Particle.PT", "Particle.Eta", "Particle.Phi"])

gen = ak.zip({"pid": particle["Particle.PID"],
              "status": particle["Particle.Status"],
              "pt": particle["Particle.PT"],
              "eta": particle["Particle.Eta"],
              "phi": particle["Particle.Phi"]})

quark = gen[(abs(gen.pid) == 3) | (abs(gen.pid) == 5)]
quark = quark[quark.status == 23]


pairs = ak.cartesian({"quark": quark, "jet": good_jet}, axis=1, nested=True)
dr = delta_r(pairs.jet.eta, pairs.jet.phi, pairs.quark.eta, pairs.quark.phi)
print(dr)