

# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils_Geo as utils
import argparse
from qm9 import dataset
from qm9.models import get_model, get_autoencoder, get_latent_diffusion
import os
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import sample
from qm9.analyze import analyze_stability_for_molecules, analyze_node_distribution
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
import qm9.losses as losses

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

def main():
    with open(join('GeoLDM_Drugs', 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)
    
    # Retrieve GEOM-Drugs dataloaders 
    # dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    print(args.dataset)
    # Load model
    generative_model, nodes_dist, prop_dist = get_autoencoder(args, device, dataset_info)
    
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy' 
    flow_state_dict = torch.load(join('GeoLDM_Drugs', fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict, strict=False)
    print(generative_model.latent_node_nf)
    
    return generative_model
    
    

if __name__=="__main__":
    main()
