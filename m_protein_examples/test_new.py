from os import sys
sys.path.append("/beagle3/gavoth/cwaltmann/code/chroma")
from chroma import api
api.register_key("900b9938893b4912a03c0e694e046dd0")
import torch
from chroma import Chroma, conditioners, Protein
import numpy as np
path = "/beagle3/gavoth/cwaltmann/code/chroma_weights/"
chroma = Chroma(weights_backbone=path + "chroma_backbone_v1.0.pt", weights_design=path + "chroma_design_v1.0.pt")
import yaml
import copy as cp

device="cpu"

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = "cuda"

def load_targets(fname):
    f = open(fname, "r")
    data = f.readlines()[8:48]
    targets = []
    for line in data:
        s = line.split()
        pos = [float(s[6]), float(s[7]), float(s[8])]
        targets.append(pos)
    return np.array(targets)



def get_map(yname, pname):
    with open(yname, "r") as stream:
        data_loaded = yaml.safe_load(stream)
    indexes = []
    count = 1
    for pair in data_loaded['system'][0]['sites']:
        indexes.append(np.add(data_loaded['site-types'][pair[0]]['index'], pair[1]))
    resids = []
    f = open(pname, "r")
    data = f.readlines()
    for bead in indexes:
        bead_list = []
        for aa in bead:
            s = data[aa + 8].split()
            num = int(s[5]) + 222 * int(s[4] == "B")
            bead_list.append(num)
        resids.append(bead_list)
    return resids



# get 11 samples
for index in range(11):
    # protein is already aligned to the CG targets
    protein = Protein("long_aligned_compact.pdb", device=device)
    mapp = get_map("M_mapping.yaml", "long_aligned_compact.pdb")
    #print(mapp)

    target = load_targets("cg_targets_compact.pdb")
    #print(target)
    #this is the shift factor of 2 Angstroms to stop it from over optimizing for the CG targets during the soft constraint
    allowed = 2
    #this weights the consraint energy relative to the sampling energy
    weight = 10

    # first step with soft constraint
    protein = chroma.cg_sample(mapp, target, allowed, weight, protein_init=protein, steps=1000, initialize_noise=False, fixed=False, sde_func="reverse_sde", noise_range=[0,3])
    #output intermediate structure
    protein.to("cg_backmapped_long2compact" +str(index)+  "_soft.pdb")

    # hard constraint
    protein = chroma.cg_sample(mapp, target, allowed, weight, protein_init=protein, steps=1000, initialize_noise=False, fixed=True, sde_func="reverse_sde", noise_range=[3,4])
    protein.to("cg_backmapped_long2compact" + str(index) + "_hard.pdb")

    #output
    n_protein = Protein("cg_backmapped_long2compact" + str(index) + "_hard.pdb", device=device)

    # this part just adds heavy side-chain atoms, but it's really bad at it
    n_protein = chroma.pack(n_protein, clamped=True)
    n_protein.to("cg_backmapped_long2compact_" + str(index) + "hard_packed.pdb")
