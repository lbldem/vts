import sys
import time
import uproot
import awkward as ak
import numpy as np

BRANCHES = [
    # Jet
    "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass", "Jet.BTag", "Jet.Charge", "Jet.Constituents",
    # Jet constituent: track
    "EFlowTrack.fUniqueID", "EFlowTrack.PT", "EFlowTrack.Eta", "EFlowTrack.Phi", "EFlowTrack.D0",
    "EFlowTrack.DZ", "EFlowTrack.Charge", "EFlowTrack.PID",  
    # Jet constituent: tower (photon, neutral hadron)
    "EFlowPhoton.fUniqueID", "EFlowPhoton.ET", "EFlowPhoton.Eta", "EFlowPhoton.Phi",
    "EFlowPhoton.Eem", "EFlowPhoton.Ehad",
    "EFlowNeutralHadron.fUniqueID", "EFlowNeutralHadron.ET", "EFlowNeutralHadron.Eta",
    "EFlowNeutralHadron.Phi", "EFlowNeutralHadron.Eem", "EFlowNeutralHadron.Ehad",
    # Lepton
    "Electron.PT", "Electron.Eta", "Electron.Phi", "Electron.Charge", "Electron.IsolationVar",
    "Muon.PT", "Muon.Eta", "Muon.Phi", "Muon.Charge", "Muon.IsolationVar",
    # MET
    "MissingET.MET", "MissingET.Phi",
    # gen
    "Particle.PID", "Particle.M1", "Particle.PT", "Particle.Eta", "Particle.Phi"]

def read_delphes(path):
    tree = uproot.open(path)["Delphes"]
    return tree.arrays(BRANCHES)

if __name__ == "__main__":
    path = sys.argv[1]
    start = time.time()
    d = read_delphes(path)
    