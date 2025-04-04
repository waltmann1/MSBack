import pickle as pkl
from egnn_utils import *
from time import time
import numpy as np

def timer(func):
    def time_wrapper(*args, **kwargs):
        now = time()
        result = func(*args, **kwargs)
        end = time()
        total = end - now
        print(f"The total time was {total} seconds.")
        return result

    return time_wrapper

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

def pro_res_to_ohe(string_list):

    amino_acids = {
    "ALA": 1,  # Alanine
    "ARG": 2,  # Arginine
    "ASN": 3,  # Asparagine
    "ASP": 4,  # Aspartic acid
    "CYS": 5,  # Cysteine
    "GLU": 6,  # Glutamic acid
    "GLN": 7,  # Glutamine
    "GLY": 8,  # Glycine
    "HIS": 9,  # Histidine
    "ILE": 10, # Isoleucine
    "LEU": 11, # Leucine
    "LYS": 12, # Lysine
    "MET": 13, # Methionine
    "PHE": 14, # Phenylalanine
    "PRO": 15, # Proline
    "SER": 16, # Serine
    "THR": 17, # Threonine
    "TRP": 18, # Tryptophan
    "TYR": 19, # Tyrosine
    "VAL": 20  # Valine
    }

    indices = [amino_acids[string] for string in string_list]
    return np.array(indices)

def pro_allatom_to_ohe(string_list):
    '''List all possible atom names'''

    atom_types = {
        'C':1,
        'CA':2,
        'CB':3,
        'CD':4,
        'CD1':5,
        'CD2':6,
        'CE':7,
        'CE1':8,
        'CE2':9,
        'CE3':10,
        'CG':11,
        'CG1':12,
        'CG2':13,
        'CH2':14,
        'CZ':15,
        'CZ2':16,
        'CZ3':17,
        'N':18,
        'ND1':19,
        'ND2':20,
        'NE':21,
        'NE1':22,
        'NE2':23,
        'NH1':24,
        'NH2':25,
        'NZ':26,
        'O':27,
        'OT2':27,
        'OD1':28,
        'OD2':29,
        'OE1':30,
        'OE2':31,
        'OG':32,
        'OG1':33,
        'OH':34,
        'SD':35,
        'SG':36
    }

    indices = [atom_types[string] for string in string_list]
    return np.array(indices)

def pro_atom_to_ohe(string_list):

    atoms = {
            'C': 1,
            'O': 2,
            'N': 3,
            'S': 4,
        }
    indices = [atoms[string[0]] for string in string_list]
    return np.array(indices)

def load_model(model_path, ckp, device): #, sym='e3', pos_cos=None, seq_feats=0, seq_decay=100):
    '''Load model from a given path to device'''

    # load hyperparams associated with specific model
    model_path = os.path.dirname(os.path.abspath(__file__)) + "/" +  model_path
    model_params = pkl.load(open(f'{model_path}/params.pkl', 'rb'))

    ## TODO -- get rid or these inputs and always set pos_cos, num_positions to None
    pos_cos = model_params['pos_cos']

    # don't use normal pos encoding if using sin/cos
    if pos_cos > 0.001:
        num_positions = None
    else:
        pos_cos = None
        num_positions = None  # change this back ifthe
        #num_positions = model_params['max_atoms']

    model = EGNN_Network_time(
        num_tokens = model_params['res_dim'],
        atom_dim = model_params['atom_dim'],
        num_positions = num_positions, #model_params['max_atoms'],
        dim = model_params['dim'],
        depth = model_params['depth'],
        num_nearest_neighbors = model_params['num_nearest_neighbors'],
        global_linear_attn_every = 0,
        coor_weights_clamp_value = 2.,
        m_dim=model_params['mdim'],
        fourier_features = 4,
        time_dim=0,
        res_dim=model_params['res_dim'],
        sym=model_params['sym'],
        emb_cos_scale=pos_cos,
        seq_feats=model_params['seq_feats'],
        seq_decay=model_params['seq_decay'],
        act=model_params['act'],
    ).to(device)

    # load model
    state_dict_path = f'{model_path}/state-{ckp}.pth'
    model.load_state_dict(torch.load(state_dict_path))

    return model
