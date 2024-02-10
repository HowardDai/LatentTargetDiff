# Latent Target Diffusion


This repository is the implementation of Latent TargetDiff, for my CPSC 483 final project. The corresponding paper can be found [here](https://drive.google.com/file/d/1P9IocrFOaITxkJaY4Xg5F1EzRMhZmK1m/view?usp=sharing).

This repository is based off of the following two repositories:

https://github.com/guanjq/targetdiff 

https://github.com/MinkaiXu/GeoLDM

## Script for Producing Main Results:
**[summary.ipynb](summary.ipynb)**

To run the notebook, download all baselines from the folder linked below (in "sampling_results").

### Links for GeoLDM autoencoder and baseline samples:
Files to download are located [here](https://drive.google.com/drive/folders/1lQLC9Nztl6-x-z1zRpBNVNWTA8-UtGYK?usp=sharing).

## Environment

### Dependency

The code has been tested in the following environment:

Create Mamba environment
```bash
mamba env create -f environment.yaml
conda activate latenttargetdiff  # note: one still needs to use `conda` to (de)activate environments
```
-----

# Target-Aware Molecule Generation
## Data
The data used for training / evaluating the model are organized in the [data](https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK?usp=share_link) Google Drive folder.  

To train the model from scratch, you need to download the preprocessed lmdb file and split file:
* `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`
* `crossdocked_pocket10_pose_split.pt`

To evaluate the model on the test set, you need to download _and_ unzip the `test_set.zip`. It includes the original PDB files that will be used in Vina Docking.

## Training
### Training from scratch
```bash
python scripts/train_diffusion.py configs/training.yml
```

## Sampling
### Sampling for pockets in the testset
```bash
python scripts/sample_diffusion.py configs/sampling.yml --data_id {i} # Replace {i} with the index of the data. i should be between 0 and 99 for the testset.
```

## Evaluation
### Evaluation from sampling results
```bash
python scripts/evaluate_diffusion.py {OUTPUT_DIR} --docking_mode vina_score --protein_root data/test_set
```
The docking mode can be chosen from {qvina, vina_score, vina_dock, none}

Note: It will take some time to prepare pqdqt and pqr files when you run the evaluation code with vina_score/vina_dock docking mode for the first time.




```
