import numpy as np
import mstool
import sys
sys.path.append("/home/cwaltmann/PycharmProjects/MSBack/MSBack")
from MSToolProtein import MSToolProtein
from MSToolProtein import MSToolProteinComplex
from MSToolProtein import AAProtein
import time




mstp = AAProtein("M_AA.pdb", yaml = "M_mapping.yaml")
cg_protein = MSToolProtein("M_long_CG.pdb")
rmsd, aligned_AA = mstp.get_protein_aligned_with_cg_group(cg_protein)
print(rmsd)
aligned_AA.write("compact_aligned_long.pdb")

