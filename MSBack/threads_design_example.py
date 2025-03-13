import threading 
import os
import time
import torch

import numpy as np
import mstool
import sys
sys.path.append("/home/cwaltmann/PycharmProjects/MSBack/MSBack")
from MSToolProtein import MSToolProtein
from MSToolProtein import MSToolProteinComplex
from MSToolProtein import AAProtein
import copy as cp
from Simulation import *

mstp_start = MSToolProtein("protein.pdb")

chains = mstp_start.u.atoms.chain.values
chains = list(set(chains))
chains.sort()
prefix = "temp2/"

mstp_start.init_atom_map()
n_mer = 7

if not os.path.exists(prefix):
        os.makedirs(prefix)

begin = time.time()


def launch_pool_in_batches(pool, n_threads):
    n_batches = int(np.ceil(len(pool) / n_threads))
    for batch in range(n_batches):
        start = batch * n_threads
        end = np.min( [(batch + 1) * n_threads, len(pool)])
        subpool = pool[start:end]
        for thread in subpool:
            thread.start()
        for thread in subpool:
            thread.join()

def flow(mstp_start, new_sequence, flowback, device, chain, prefix="./temp2/"):

    pro = cp.deepcopy(mstp_start)
    pro.u.atoms = pro.u.atoms[pro.u.atoms['name']=='CA']
    pro.u.atoms.loc[pro.u.atoms['chain']  == chain, 'resname'] = new_sequence
    pro.add_backbone_sidechains()
    start = time.time()
    flo = pro.flowback_protein(flowback=flowback, device=device )
    end = time.time()
    flo.add_C_terminal_O()
    print("finished chain " + str(chain) + ". flowback took " + str(end-start))
    flo.write(prefix + "design_reassembled" + str(chain) + "_capped.pdb")

n_threads = torch.cuda.device_count()

flowbacks = [mstp_start.load_flowback(device="cuda:" + str(i)) for i in range(n_threads)]
chromas = [mstp_start.load_chroma(device="cuda:" + str(i)) for i in range(n_threads)]
devices = ["cuda:" + str(i) for i in range(n_threads)]
print("hello")

seqs = []
pool = [ threading.Thread(target=design_worker, args=(mstp_start, c, chromas[i%n_threads][0], chromas[i%n_threads][1], seqs))  for i, c in enumerate(chains)]


launch_pool_in_batches(pool, n_threads)
print(time.time()-begin)
print("hello again")

pool = [threading.Thread(target=flow, args=(mstp_start, seqs[i], flowbacks[i%n_threads][0], flowbacks[i%n_threads][1], c)) for i,c in enumerate(chains)]
launch_pool_in_batches(pool, n_threads)

print(time.time()-begin)
print("hello again")


#protein_names = [prefix + "design_reassembled" + str(chains[i]) + "_capped.pdb" for i in range(n_mer)]

#ols = [mstp_start.three_to_one_letter_map[str(resn)] for resn in seqs]

#pool = [threading.Thread(target=sim_worker, args=(protein_names[i], i%n_threads, i, ols[i] )) for i in range(n_mer)]
#launch_pool_in_batches(pool, n_threads)


#"""
print(time.time()-begin)
print("hello again")



