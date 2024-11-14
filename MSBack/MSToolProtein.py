import math as m
import numpy as np
import copy as cp
import yaml
import mstool
import time


class MSToolProtein(object):

    def __init__(self, pdbfile, trajectory=None):

        self.u = mstool.Universe(pdbfile)

        self.map = None

        self.name = pdbfile[:-4]
        print(self.name)



    def get_map(self, mapping_file):

        map = None
        print(mapping_file, mapping_file[-5:])
        if mapping_file[-5:] == ".yaml":
            map = self.read_cg_yaml(mapping_file)
            print("map", map)
        return map

    def reindex_chains(self, number):
        i=0
        length = self.u.atoms.name.count()

        while i * number < length:
            self.u.atoms.loc[i*number:(i+1)*number, 'chain'] = get_segid(i)
            i+=1

    def count_chain_lengths(self):

        length = self.u.atoms.name.count()
        current = 0
        past = 0
        lengths = []
        i =0
        num =0
        resids = self.u.atoms.resid
        #print(resids)
        #print(resids[0])
        while i < length:
            past = current
            current = resids[i]
            #print(current)
            if current < past:
                print(num)
                lengths.append(num)
                num = 0
            i += 1
            num += 1
        lengths.append(num)
        return lengths
    
    def reindex_chains_with_list(self, chain_lengths):

        current = 0
        for i in range(len(chain_lengths)):
            length = chain_lengths[i]
            self.u.atoms.loc[current:current + length, 'chain'] = get_segid(i)
            current += length

    def shift(self, vector):

        values = self.u.atoms[['x', 'y', 'z']].values
        values = np.add(values, vector)
        self.u.atoms[['x', 'y', 'z']] = values

    def shift_center_to(self, position):

        values = self.u.atoms[['x', 'y', 'z']].values

        center = np.average(values, axis=0)

        to_shift = np.subtract(position, center)

        self.shift(to_shift)

        #print(self.u.atoms[['x', 'y', 'z']].values)
        #quit()

    def get_positions(self):

        return self.u.atoms[['x', 'y', 'z']].values

    def write(self, name):

        self.u.write(name)

    def dump_CA_only(self, name=None):

        self.u.atoms = self.u.atoms.query('name == "CA"')
        if name is None:
            name = self.name + "_CA_only.dms"
        self.write(name=name)
        
    def rmsd(self, ref_protein):

        uref = ref_protein.u
        new_mobile_xyz, rmsd = mstool._fit_to(self.u.atoms[['x', 'y', 'z']].values, uref.atoms[['x', 'y', 'z']].values)

        return rmsd


class AAProtein(MSToolProtein):

    def __init__(self, pdbfile, yaml=None):
        super(AAProtein, self).__init__(pdbfile=pdbfile)
        self.map = None

        if yaml is not None:
            self.map = self.read_cg_yaml(yaml)

    def read_cg_yaml(self, yname):

        with open(yname, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        indexes = []

        for pair in data_loaded['system'][0]['sites']:
            indexes.append(np.add(data_loaded['site-types'][pair[0]]['index'], pair[1]))

        return indexes

    def get_cg_protein(self):

        positions = self.u.atoms[['x', 'y', 'z']].values
        to_return = cp.deepcopy(self)
        to_return.u.atoms = to_return.u.atoms[:len(self.map)]
        new_positions = [np.average(positions[np.subtract(bead, 1)], axis=0) for bead in self.map]
        to_return.u.atoms[['x', 'y', 'z']] = new_positions
        to_return.u.atoms['name'] = 'BB'
        to_return.u.atoms['resname'] = ["P" + str(i+1) for i in range(len(to_return.u.atoms))]
        to_return.u.atoms['resid'] = list(range(1, len(to_return.u.atoms) + 1))
        to_return.u.atoms['chain'] = 'A'
        return to_return

    def get_protein_aligned_with_cg_group(self, ref_protein):

        uref = ref_protein.u

        cg_protein  = self.get_cg_protein()

        mob_ref_atoms = cg_protein.u.atoms
        umob = cp.deepcopy(self)
        new_mobile_xyz, rmsd = mstool._fit_to(mob_ref_atoms[['x', 'y', 'z']].values, uref.atoms[['x', 'y', 'z']].values)


        dr_mob_xyz = np.average(mob_ref_atoms[['x', 'y', 'z']].values, axis=0)
        dr_ref_xyz = np.average(uref.atoms[['x', 'y', 'z']].values, axis=0)
        R, rmsd = mstool.rotation_matrix(mob_ref_atoms[['x', 'y', 'z']].values - dr_mob_xyz,
                                         uref.atoms[['x', 'y', 'z']].values - dr_ref_xyz)

        new_mob_xyz = dr_ref_xyz + (R @ (umob.u.atoms[['x', 'y', 'z']].values - dr_mob_xyz).T).T

        umob.u.atoms[['x', 'y', 'z']] = new_mob_xyz

        return rmsd, umob

    def diffuse_to_CG(self, cg_protein, output_soft=True, pack_sidechains=True, output_name=None):

        import sys
        import os
        sys.path.append("/beagle3/gavoth/cwaltmann/code/chroma")
        from chroma import api
        api.register_key("900b9938893b4912a03c0e694e046dd0")
        import torch
        from chroma import Chroma, conditioners, Protein
        path = "/beagle3/gavoth/cwaltmann/code/chroma_weights/"
        chroma = Chroma(weights_backbone=path + "chroma_backbone_v1.0.pt", weights_design=path + "chroma_design_v1.0.pt")

        device = "cpu"

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = "cuda"

        mapp = self.map
        f = MSToolProtein(cg_protein)
        target = np.array(f.get_positions())

        protein_name = self.name + ".pdb"
        if not os.exists(protein_name):
            protein_name = "temp.pdb"
            self.write(protein_name)
        name = protein_name
        if name is not None:
            name = output_name
        protein = Protein(protein_name, device=device)
        allowed =2
        weight=10
        protein = chroma.cg_sample(mapp, target, allowed, weight, protein_init=protein, steps=1000,
                                   initialize_noise=False, fixed=False, sde_func="reverse_sde", noise_range=[0, 3])
        if output_soft:
            protein.to("soft_" + name + ".pdb")

        protein = chroma.cg_sample(mapp, target, allowed, weight, protein_init=protein, steps=1000,
                                   initialize_noise=False, fixed=True, sde_func="reverse_sde", noise_range=[3, 4])
        protein.to("hard_" + name + ".pdb")
        
        if pack_sidechains:
            n_protein = Protein("hard_" + name + ".pdb", device=device)
            n_protein = chroma.pack(n_protein, clamped=True)
            n_protein.to("hard_" + name + "_packed.pdb")
            
        last_name = "hard_" + name
        if pack_sidechains:
            last_name += "_packed"
        last_name += ".pdb"
        
        return MSToolProtein(last_name)


class MSToolProteinComplex(object):

    def __init__(self, mstps):

        self.proteins = mstps
        self.name = "complex"
        self.chain_indices = [get_segid(i) for i in range(len(self.proteins))]



    def get_universe(self):

        for index, protein in enumerate(self.proteins):
            protein.u.atoms['chain'] = get_segid(index)

        return self.list_merge([protein.u for protein in self.proteins])

    def binary_merge(self, lizt):

        odd = len(lizt) % 2 != 0
        half = int(len(lizt) / 2)
        save = ""
        if odd:
            save = lizt[-1]
        lizt = [mstool.Merge(lizt[i].atoms, lizt[i + half].atoms) for i in range(half)]
        if odd:
            lizt.append(save)
        while len(lizt) != 1:
            odd = len(lizt) % 2 != 0
            if odd:
                save = lizt[-1]
            half = int(len(lizt) / 2)
            now = time.time()
            lizt = [mstool.Merge(lizt[i].atoms, lizt[i + half].atoms) for i in range(half)]
            later = time.time()
            print(later - now)
            if odd:
                lizt.append(save)
        return lizt[0]

    def list_merge(self, lizt):

        now = time.time()
        for i in range(1,len(lizt)):
            print(i)
            lizt[0] = mstool.Merge(lizt[0].atoms, lizt[i].atoms)
        later = time.time()
        print(later - now)
        return lizt[0]

    def write(self,name=None, pdb=False, reindex=False, reindex_list=None,
              reindex_count=False):

        if name is None:
            name = self.name + ".dms"

        total = self.get_universe()
        if reindex:
            i = 0
            length = total.atoms.name.count()

            while i * reindex < length:
                total.atoms.loc[i * reindex:(i + 1) * reindex, 'chain'] = get_segid(i)
                i += 1
        elif reindex_count:
            length = total.atoms.name.count()
            current = 0
            past = 0
            reindex_list = []
            i = 0
            num = 0
            resids = total.atoms.resid
            while i < length:
                past = current
                current = resids[i]
                # print(current)
                if current < past:
                    print(num)
                    reindex_list.append(num)
                    num = 0
                i += 1
                num += 1
            reindex_list.append(num)
        if reindex_list is not None:
            current = 0
            for i in range(len(reindex_list)):
                length = reindex_list[i]
                total.atoms.loc[current:current + length, 'chain'] = get_segid(i)
                current += length

        total.write(name)

        if pdb:
            try:
                total.write(name + ".pdb")
            except Exception as e:
                print("Probably have an illegal chain name for PDB")

def get_segid(index, include_numbers=True):

        num = index // 26
        if num == 0:
            num = ""
        out = str(chr(index % 26 + 97)).upper()
        if include_numbers:
            out = out + str(num)

        return  out
