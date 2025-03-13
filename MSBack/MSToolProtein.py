import math as m
import numpy as np
import copy as cp
import yaml
import mstool
import time
import pandas as pd
import sys

sys.path.append("../Utils")
from Utils import *

class MSToolProtein(object):

    def __init__(self, pdbfile, trajectory=None):

        self.u = mstool.Universe(pdbfile)

        self.map = None

        self.name = pdbfile.split("/")[-1][:-4]
        print(self.name)

        self.chroma = None
        self.chroma_device = None
        self.flowback = None
        self.flowback_device = None
        self.atom_map_14 = None
        self.three_to_one_letter_map = None



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

    def reindex_chains_by_residues(self, chain_lengths):

        current_resid = 1
        for i in range(len(chain_lengths)):
            previous_resid = current_resid
            current_resid += chain_lengths[i]
            #print("a")
            #print((self.u.atoms.resid < current_resid)[0])
            condition1 = (self.u.atoms.resid < current_resid)
            condition2 = (self.u.atoms.resid >= previous_resid)
            self.u.atoms.loc[condition1 & condition2, 'chain'] = get_segid(i)
            #print("b")
            current_resids = self.u.atoms[condition1 & condition2].resid.values
            #print(current_resids[0])
            new_resids = np.subtract(current_resids, current_resid - chain_lengths[i] - 1)
            #print(new_resids[0])
            self.u.atoms.loc[condition1 & condition2, 'resid'] = new_resids
            #print(i)
            #print(self.u.atoms.chain)


    def shift(self, vector):

        values = self.u.atoms[['x', 'y', 'z']].values
        values = np.add(values, vector)
        self.u.atoms[['x', 'y', 'z']] = values

    def center(self):

        values = self.u.atoms[['x', 'y', 'z']].values
        vector = np.multiply(-1, np.average(values, axis=0))
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
        print(self.u.atoms[['x', 'y', 'z']].values.shape)
        print(uref.atoms[['x', 'y', 'z']].values.shape)
        new_mobile_xyz, rmsd = mstool._fit_to(self.u.atoms[['x', 'y', 'z']].values, uref.atoms[['x', 'y', 'z']].values)

        return rmsd

    def load_chroma(self, device=None):

        if self.chroma is None:
            import sys
            import os
            sys.path.append("/beagle3/gavoth/cwaltmann/code/chroma")
            from chroma import api
            import torch
            api.register_key("900b9938893b4912a03c0e694e046dd0")
            from chroma import Chroma, Protein
            print("imported Protein")
            path = "/beagle3/gavoth/cwaltmann/code/chroma_weights/"
            if device is None:
                device = "cpu"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    device = "cuda"
            self.chroma = Chroma(weights_backbone=path + "chroma_backbone_v1.0.pt",
                                 weights_design=path + "chroma_design_v1.0.pt", device=device)
            self.chroma_device = device
        else:
            if device is not None:
                if self.chroma_device != device:
                    import sys
                    import os
                    sys.path.append("/beagle3/gavoth/cwaltmann/code/chroma")
                    from chroma import api
                    import torch
                    api.register_key("900b9938893b4912a03c0e694e046dd0")
                    from chroma import Chroma, Protein
                    print("imported Protein")
                    path = "/beagle3/gavoth/cwaltmann/code/chroma_weights/"
                    self.chroma = Chroma(weights_backbone=path + "chroma_backbone_v1.0.pt",
                                         weights_design=path + "chroma_design_v1.0.pt", device=device)
                    self.chroma_device = device

        print("chroma device", self.chroma_device)
        return self.chroma, self.chroma_device



    def design_protein_sequence(self, pack_sidechains=True, output_name=None, chroma=None, device=None, p=5):
        from chroma import Protein

        if chroma is None or device is None:
            if self.chroma is None:
                self.load_chroma(device=device)

            chroma = self.chroma
            device = self.chroma_device


        protein_name =  self.name + ".pdb"
        if not os.path.exists(protein_name):
            protein_name = "temp.pdb"
            self.write(protein_name)

        if output_name is None:
            output_name = "design_" +self.name + ".pdb"

        protein = Protein(protein_name, device=device)

        mask = np.random.randint(100, size=221)
        p = p
        select = torch.Tensor([[thing < p for thing in mask]])
        select = torch.Tensor(select).to(device)
        print(select.shape)
        print(protein.to_XCS()[0].shape)

        n_protein = chroma.design(protein=protein, design_selection=select)
        if pack_sidechains:
            n_protein = chroma.pack(protein=n_protein, clamped=True)
        n_protein.to(output_name)
        if os.path.exists("temp.pdb"):
            os.remove("temp.pdb")
        del select
        del n_protein
        del protein
        torch.cuda.empty_cache()
        return MSToolProtein(output_name)

    def get_sequence(self):
        return self.u.atoms[self.u.atoms.name == "CA"].resname.values

    def get_one_letter_sequence(self):

        self.init_atom_map()
        three = self.get_sequence()
        return [self.three_to_one_letter_map[resn] for resn in three]


    def prep_flowback(self):

        b1 = ["H" != name[0] and "OT2" != name[0] for name in self.u.atoms.name]

        self.u.atoms = self.u.atoms[b1]
        b2 = self.u.atoms.name != "CA"
        others = self.u.atoms[b2]
        n_atoms = others.shape[0]
        values = [[0.0,0.0,0.0]  for _ in range(n_atoms)]
        self.u.atoms.loc[b2, ["x", "y", "z"]] = values

    def init_atom_map(self):

        if self.three_to_one_letter_map is None:
            from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, SC_BUILD_INFO
            from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP

            self.three_to_one_letter_map = {y: x for x, y in ONE_TO_THREE_LETTER_MAP.items()}

            self.atom_map_14 = {}
            for one_letter in ONE_TO_THREE_LETTER_MAP.keys():
                self.atom_map_14[one_letter] = ["N", "CA", "C", "O"] + list(
                    SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
                #self.atom_map_14[one_letter].extend(["PAD"] * (14 - len(self[one_letter])))

    def add_backbone_sidechains(self):

        proteinu = self.u
        data = {'name': [],
                'resname': [],
                'resid': [],
                'chain': [],
                'segname': [],
                'x': [],
                'y': [],
                'z': [],
                'bfactor': 0.0}

        fields = ['resname', 'chain', 'segname', 'resid']
        things = proteinu.atoms[fields].values
        resname = things[:, 0]
        chain = things[:, 1]
        segname = things[:, 2]
        resid = things[:, 3]
        self.init_atom_map()
        for ind, resn in enumerate(resname):
            for name in self.atom_map_14[self.three_to_one_letter_map[resn]]:
                if name != "CA":
                    #print(name, resn, segname[ind], resid[ind], chain[ind] )
                    data['name'].extend([name])
                    data['resname'].extend([resn])
                    data['segname'].extend([segname[ind]])
                    data['resid'].extend([resid[ind]])
                    data['chain'].extend([chain[ind]])
                    data['x'].extend([0.0])
                    data['y'].extend([0.0])
                    data['z'].extend([0.0])
        new_proteinu_atoms = pd.concat([proteinu.atoms, pd.DataFrame(data)], ignore_index=True)
        new_proteinu_atoms.sort_values(by=['chain', 'resid'], inplace=True)
        self.u.atoms = new_proteinu_atoms

    def remove_duplicate_atoms(self, keep_first=True):

        chains = list(set(self.u.atoms.chain.values))
        chains.sort()
        #print("chains", chains)
        to_remove = []
        for chain in chains:
            sub1 = self.u.atoms[self.u.atoms.chain==chain]
            resids = list(set(sub1.resid.values))
            resids.sort()
            #print(resids)
            for resid in resids:
                sub2 = sub1[sub1.resid==resid]
                sub2_indexes = sub2.index.values
                sub2_names = sub2.name.values
                #print(resid, sub2_names, sub2_indexes)
                seen_names = []
                for index, name in enumerate(sub2_names):
                    if name in seen_names:
                        to_remove.append(sub2_indexes[index])
                    else:
                        seen_names.append(name)
            #print(to_remove)
            #print(len(to_remove))
        for i in range(len(to_remove) - 1, -1, -1):
            #print("removing at index ", to_remove[i])
            self.u.atoms.drop(index=to_remove[i], inplace=True)

    def add_C_terminal_O(self):

        proteinu = self.u
        data = {'name': [],
                'resname': [],
                'resid': [],
                'chain': [],
                'segname': [],
                'x': [],
                'y': [],
                'z': [],
                'bfactor': 0.0}

        chains = list(dict.fromkeys(proteinu.atoms['chain']))
        for chain in chains:
            bA1 = proteinu.atoms.chain == chain
            Ntermresid = proteinu.atoms[bA1]['resid'].min()
            Ctermresid = proteinu.atoms[bA1]['resid'].max()

            bA2 = proteinu.atoms.resid == Ctermresid
            bA3 = proteinu.atoms['name'] == 'C'
            bA4 = proteinu.atoms['name'].isin(['OT1', 'O', 'O1'])
            bA5 = proteinu.atoms['name'] == 'CA'
            bA6 = proteinu.atoms['name'] == 'N'

            resn, segn, Cx, Cy, Cz = proteinu.atoms[bA1 & bA2 & bA3][['resname', 'segname', 'x', 'y', 'z']].values[0]
            Ox, Oy, Oz = proteinu.atoms[bA1 & bA2 & bA4][['x', 'y', 'z']].values[0]
            CAx, CAy, CAz = proteinu.atoms[bA1 & bA2 & bA5][['x', 'y', 'z']].values[0]
            Nx, Ny, Nz = proteinu.atoms[bA1 & bA2 & bA6][['x', 'y', 'z']].values[0]

            dr1 = np.array([CAx, CAy, CAz]) - np.array([Nx, Ny, Nz])
            dr2 = np.array([Ox, Oy, Oz]) - np.array([Cx, Cy, Cz])
            dr3 = np.array([Cx, Cy, Cz]) - np.array([CAx, CAy, CAz])

            dr1 /= np.linalg.norm(dr1)
            dr2 /= np.linalg.norm(dr2)
            dr3 /= np.linalg.norm(dr3)

            posOT2 = np.array([Cx, Cy, Cz]) + (dr3 - dr2) / np.linalg.norm(dr3 - dr2) * 1.25
            data['name'].extend(['OT2'])
            data['resname'].extend([resn])
            data['segname'].extend([segn])
            data['resid'].extend([Ctermresid])
            data['chain'].extend([chain])
            data['x'].extend([posOT2[0]])
            data['y'].extend([posOT2[1]])
            data['z'].extend([posOT2[2]])

        new_proteinu_atoms = pd.concat([proteinu.atoms, pd.DataFrame(data)], ignore_index=True)
        new_proteinu_atoms.sort_values(by=['chain', 'resid'], inplace=True)
        self.u.atoms = new_proteinu_atoms

    def load_flowback(self, device=None):

        if self.flowback is None:
            import sys
            import os
            import torch
            path_to_flowback = './Flow-Back/'
            if path_to_flowback not in sys.path:
                sys.path.append(path_to_flowback)
                sys.path.append(path_to_flowback + 'scripts/utils/')
                sys.path.append( path_to_flowback + 'scripts/')
            if device is None:
                device = "cpu"
                if torch.cuda.is_available():
                    device = "cuda"
            ckp = 14
            model_path = path_to_flowback + "/models/Pro_pretrained"
            self.flowback = load_model(model_path, ckp, device)
            self.flowback_device = device
        else:
            if device is not None:
                if self.flowback_device != device:
                    import sys
                    import os
                    import torch
                    path_to_flowback = './Flow-Back/'
                    if path_to_flowback not in sys.path:
                        sys.path.append(path_to_flowback)
                        sys.path.append(path_to_flowback + 'scripts/utils/')
                        sys.path.append(path_to_flowback + 'scripts/')
                    if device is None:
                        device = "cpu"
                        if torch.cuda.is_available():
                            device = "cuda"
                    ckp = 14
                    model_path = path_to_flowback + "/models/Pro_pretrained"
                    self.flowback = load_model(model_path, ckp, device)
                    self.flowback_device = device

        return self.flowback, self.flowback_device



    def remove_hydrogens(self):

        b1 = ["H" != name[0] and "OT2" != name[0] for name in self.u.atoms.name]
        self.u.atoms = self.u.atoms[b1]

    def reindex_resids(self):

        chains = list(set(self.u.atoms.chain.values))
        chains.sort()
        #print("chains", chains)
        for chain in chains:
            #print(chain, " is the chain")
            values = self.u.atoms[self.u.atoms.chain == chain].resid.values
            #print(values)
            values = np.subtract(values, int(min(values) - 1))
            #print(values)
            indexes = self.u.atoms[self.u.atoms.chain==chain].index.values
            self.u.atoms.loc[indexes[0]:indexes[-1],  'resid'] = values


    def get_aligned_protein(self, ref_protein):

        uref = ref_protein.u

        mob_ref_atoms = self.u.atoms
        umob = cp.deepcopy(self)

        print(mob_ref_atoms[['x', 'y', 'z']].values.shape, uref.atoms[['x', 'y', 'z']].values.shape)

        new_mobile_xyz, rmsd = mstool._fit_to(mob_ref_atoms[['x', 'y', 'z']].values, uref.atoms[['x', 'y', 'z']].values)
        umob.u.atoms[['x', 'y', 'z']] = new_mobile_xyz

        umob.name = "aligned_" + umob.name

        return rmsd, umob

    def get_aligned_protein_resids(self, ref_protein, list_o_resids, ref_resids=None, CA_only=False):

        uref = cp.deepcopy(ref_protein.u)
        if ref_resids is None:
            ref_resids = list_o_resids
        print(ref_resids)
        b1 = [resid in ref_resids for resid in list(uref.atoms.resid.values)]

        uref.atoms = uref.atoms[(b1)]
        b2 = [name != "OT2" for name in uref.atoms.name.values]
        uref.atoms = uref.atoms[(b2)]

        print(list_o_resids)
        b1 = [resid in list_o_resids for resid in list(self.u.atoms.resid.values)]
        mob_ref_atoms = self.u.atoms[b1]
        b2 = [name != "OT2" for name in list(mob_ref_atoms.name.values)]
        mob_ref_atoms = mob_ref_atoms[b2]

        print(mob_ref_atoms.shape)
        print(uref.atoms.shape)
        print(mob_ref_atoms.name)
        print(uref.atoms.name)
        umob = cp.deepcopy(self)
        new_mobile_xyz, rmsd = mstool._fit_to(mob_ref_atoms[['x', 'y', 'z']].values, uref.atoms[['x', 'y', 'z']].values)

        dr_mob_xyz = np.average(mob_ref_atoms[['x', 'y', 'z']].values, axis=0)
        dr_ref_xyz = np.average(uref.atoms[['x', 'y', 'z']].values, axis=0)
        R, rmsd = mstool.rotation_matrix(mob_ref_atoms[['x', 'y', 'z']].values - dr_mob_xyz,
                                         uref.atoms[['x', 'y', 'z']].values - dr_ref_xyz)

        new_mob_xyz = dr_ref_xyz + (R @ (umob.u.atoms[['x', 'y', 'z']].values - dr_mob_xyz).T).T

        umob.u.atoms[['x', 'y', 'z']] = new_mob_xyz

        return rmsd, umob

    def flowback_protein(self, output_name=None, flowback=None, device=None):

        model = flowback
        if model is None or device is None:
            if self.flowback is None:
                self.load_flowback(device=device)

            model = self.flowback
            device = self.flowback_device

        flow_name = self.name + ".pdb"
        flowtein = cp.deepcopy(self)
        flowtein.prep_flowback()
        if not os.path.exists(flow_name):
            flowtein.write(flow_name)


        res_list = flowtein.u.atoms.resname.values
        allatom_list = flowtein.u.atoms.name.values
        res_ohe = pro_res_to_ohe(res_list)
        atom_ohe = pro_allatom_to_ohe(allatom_list)
        #atom_ohe = pro_atom_to_ohe(allatom_list)
        xyz = np.multiply(flowtein.u.atoms[['x', 'y', 'z']].values, .1)

        temp = flowtein.u.atoms.name == "CA"

        mask_idxs = np.array([ind for ind, answer in enumerate(temp) if answer])
        n_atoms = flowtein.u.atoms.shape[0]

        #mask_idxs = top.select('name CA')
        mask = np.ones(len(res_ohe))
        mask[mask_idxs] = 0

        #aa_to_cg = get_aa_to_cg(top, mask_idxs)
        #def get_aa_to_cg(top, msk):
         #   '''Mapping between AA and CG
         #      Assign to Ca positions for now with mask, but will need to generalize this'''

        #    aa_to_cg = []
        #    for atom_idx, atom in enumerate(top.atoms):
        #        res_idx = atom.residue.index
        #        aa_to_cg.append(msk[res_idx])

        #    return np.array(aa_to_cg)
        #characteristics = [(atom.resid, atom.chain) for atom in flowtein.u.atoms]
        characteristics = zip(flowtein.u.atoms.resid.values, flowtein.u.atoms.chain.values)

        old_indexes = list(flowtein.u.atoms.index.values)
        aa_to_cg = np.array([old_indexes.index(int(flowtein.u.atoms.index[(flowtein.u.atoms.name=="CA") &
            (flowtein.u.atoms.resid == c[0]) & (flowtein.u.atoms.chain==c[1])][0])) for c in characteristics])

        #res_ohe, atom_ohe, xyz, aa_to_cg, mask, n_atoms, top = load_features_pro(trj)
        xyz_gen = []

        xyz_test_real = [xyz] #[xyz[i] for i in test_idxs]
        map_test = [aa_to_cg] #[aa_to_cg] * n_test
        mask_test = [mask]# * n_test
        res_test = [res_ohe]# * n_test
        atom_test = [atom_ohe]# * n_test

        CG_noise = 0.003
        nsteps =100

        model_wrpd = ModelWrapper(model=model,
                                  feats=torch.tensor(np.array(res_test)).int().to(device),
                                  mask=torch.tensor(np.array(mask_test)).bool().to(device).to(device),
                                  atom_feats=torch.tensor(np.array(atom_test)).to(device))

        # apply noise -- only masked values need to be filled here
        xyz_test_prior = np.array(get_prior_mask(xyz_test_real, map_test, scale=CG_noise, masks=mask_test))


        with torch.no_grad():
            ode_traj = euler_integrator(model_wrpd,
                                        torch.tensor(xyz_test_prior, dtype=torch.float32).to(device),
                                        nsteps=nsteps)

        # save trj -- optionally save ODE integration not just last structure -- only for one gen
        xyz_gen.append(ode_traj[-1])
        #print(np.shape(xyz_gen))
        xyz_gen = np.concatenate(xyz_gen)
        #print(xyz_gen.shape)

        flowtein.u.atoms[['x', 'y', 'z']] = np.multiply(xyz_gen[0], 10)

        if output_name is not None:
            flowtein.write(output_name)
        return flowtein

    def get_ss(self):

        import pydssp
        coords = []
        c = list(set(self.u.atoms.chain.values))
        c.sort()
        for chain in c:
            r = self.u.atoms[self.u.atoms.chain == chain].resid.values
            r = list(set(r))
            r.sort()
            for resid in r:
                data = self.u.atoms[(self.u.atoms.resid == resid) & (self.u.atoms.chain == chain)]
                cca = data.name == "CA"
                cc = data.name == "C"
                cn = data.name == "N"
                co = data.name == "O"
                thing = []
                thing.append(np.array(data[(cn)][['x', 'y', 'z']].values)[0])
                thing.append(np.array(data[(cca)][['x', 'y', 'z']].values)[0])
                thing.append(np.array(data[(cc)][['x', 'y', 'z']].values)[0])
                thing.append(np.array(data[(co)][['x', 'y', 'z']].values)[0])
                coords.append(np.array(thing))
                #print(chain ,resid, thing)
        #print(coords)
        coords = torch.Tensor(np.array(coords))

        #hbond_mat = pydssp.get_hbond_map(coords) > 0.5

        return pydssp.assign(coords, out_type="index")




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

    def alpha_c_only_map(self, max=None):

        if max is None:
            max=float('inf')

        names = self.u.atoms.name.values
        map = cp.deepcopy(self.map)
        for ind1, bead in enumerate(map):
            to_delete = []
            for ind, atom in enumerate(bead):
                if atom >= len(names):
                    to_delete.append(ind)
                elif names[atom] != "CA":
                    to_delete.append(ind)
            for i in range(len(to_delete)):
                index = to_delete.pop()
                map[ind1] = np.delete(map[ind1], index)

        return self.cap_map(map, max)

    def remove_empties(self, map):

        if len(map[-1]) == 0:
            map.remove(map[-1])
            self.remove_empties(map)
        return map
    def cap_map(self, map, max):

        new_map = []
        count = 1
        for ind1, bead in enumerate(map):
            new_map.append([])
            for ind, atom in enumerate(bead):
                new_map[ind1].append(atom)
                count += 1
                if count > max:
                    return new_map
        return self.remove_empties(new_map)
    def res_index_map(self, max=None):

        if max is None:
            max = float('inf')
        ca_map = self.alpha_c_only_map()
        new_map =[]
        count = 1
        for ind1, bead in enumerate(ca_map):
            new_map.append([])
            for ind, atom in enumerate(bead):
                new_map[ind1].append(count)
                count += 1
                if count > max:
                    return new_map
        return  new_map

    def get_cg_protein(self, CA_only=False):

        the_map = self.map
        if CA_only:
            the_map = self.alpha_c_only_map()
        positions = self.u.atoms[['x', 'y', 'z']].values
        to_return = cp.deepcopy(self)
        to_return.u.atoms = to_return.u.atoms[:len(the_map)]
        new_positions = [np.average(positions[bead], axis=0) for bead in the_map]
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
        print("start protein, monomer", mob_ref_atoms[['x', 'y', 'z']].values.shape)
        print("end protein, conf0", uref.atoms[['x', 'y', 'z']].values.shape)
        print(uref.atoms[['x', 'y', 'z']].values)
        new_mobile_xyz, rmsd = mstool._fit_to(mob_ref_atoms[['x', 'y', 'z']].values, uref.atoms[['x', 'y', 'z']].values)


        dr_mob_xyz = np.average(mob_ref_atoms[['x', 'y', 'z']].values, axis=0)
        dr_ref_xyz = np.average(uref.atoms[['x', 'y', 'z']].values, axis=0)
        R, rmsd = mstool.rotation_matrix(mob_ref_atoms[['x', 'y', 'z']].values - dr_mob_xyz,
                                         uref.atoms[['x', 'y', 'z']].values - dr_ref_xyz)

        new_mob_xyz = dr_ref_xyz + (R @ (umob.u.atoms[['x', 'y', 'z']].values - dr_mob_xyz).T).T

        umob.u.atoms[['x', 'y', 'z']] = new_mob_xyz
        umob.name = "aligned_" + umob.name

        return rmsd, umob

    def diffuse_to_CG(self, cg_protein, output_soft=True, pack_sidechains=True,
                      output_name=None, chroma=None, device=None):

        from chroma import Protein

        if chroma is None or device is None:
            if self.chroma is None:
                self.load_chroma(device=device)

            chroma = self.chroma
            device = self.chroma_device

        mapp = self.map
        target = cg_protein.u.atoms[['x', 'y', 'z']].values
        #print("mapp")
        #print(mapp)
        #print(target)
        #quit()

        protein_name = self.name + ".pdb"
        if not os.path.exists(protein_name):
            #protein_name = "temp.pdb"
            self.write(protein_name)

        protein = Protein(protein_name, device=device)
        allowed =2
        weight=10
        protein = chroma.cg_sample(mapp, target, allowed, weight, protein_init=protein, steps=1000,
                                   initialize_noise=False, fixed=False, sde_func="reverse_sde", noise_range=[0, 3])
        if output_soft:
            protein.to("soft_" + self.name + ".pdb")

        protein = chroma.cg_sample(mapp, target, allowed, weight, protein_init=protein, steps=1000,
                                   initialize_noise=False, fixed=True, sde_func="reverse_sde", noise_range=[3, 4])

        short_name = self.name
        if output_name is not None:
            short_name = output_name.split(".")[0]
        else:
            short_name = "hard_" + self.name

        protein.to(short_name + ".pdb")

        last_name = short_name
        
        if pack_sidechains:
            n_protein = Protein(last_name + ".pdb", device=device)
            n_protein = chroma.pack(n_protein, clamped=True)
            last_name = last_name + "_packed"
            n_protein.to(last_name + ".pdb")
        
        return MSToolProtein(last_name + ".pdb")


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
                if current != past and current != past+1:
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
