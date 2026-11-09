import awkward as ak
import uproot
import numpy as np

t = uproot.open("sig_a_test.root")["Delphes"]
jet = t.arrays(["Particle.PID", "Particle.Status", "Particle.PT", "Particle.Eta", "Particle.Phi"])

gen = ak.zip({"pid": jet["Particle.PID"],
              "status": jet["Particle.Status"],
              "pt": jet["Particle.PT"],
              "eta": jet["Particle.Eta"],
              "phi": jet["Particle.Phi"]})


quark = gen[(abs(gen.pid) == 3) | (abs(gen.pid) == 5)]
quark = quark[quark.status == 23]

def delta_r(eta1, phi1, eta2, phi2):
    dphi = phi1- phi2
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    return np.sqrt((eta1 - eta2)**2 + dphi**2)

pairs = ak.cartesian({"quark": quark, "jet": gen}, axis=1, nested=True)
print(pairs)

dr = delta_r(pairs.jet.eta, pairs.jet.phi, pairs.quark.eta, pairs.quark.phi)
