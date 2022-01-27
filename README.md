# hmm_encoding
Scripts for running Hidden Markov Models and Encoding (Ridge/Banded Ridge Regression) Models


1) *hmm_functions.py* - frequently used functions for fitting an HMM on individaul/group-based fMRI data from a region of interest (subcortical or cortical) and outputting the probability function (and converting to entropy), model fit (log likelihood), voxelwise spatial patterns of the best-fitting model, and predicted moments of transition from the best-fitting model 

2) *hmm.ipynd* - notebook that uses hmm_functions.py to identity brain regions sensitive to emotion transitions in movies and music

3) *ridge_wholebrain_hab.py* - script for running encoding models with ridge regression (predicting voxelwise/parcel activity in the brain from a set of features extracted from the stimulus) on a high-performance computing cluster with SLURM [outer CV loops for validation and inner CV loops for optimzation of the model are set as inputs]

4) *banded_ridge_full_hab.py* - script for running encoding models with banded ridge regression (optimizing alphas separately for each set of features) on a high-performance computing cluster with SLURM [outer CV loops for validation and inner CV loops for optimzation of the model are set as inputs]

5) *apply_srm.ipynb* - jupyter notebook that applies a method of functional alignment (shared response model that tries to learn a common representational space of lower dimensionality) to retains features of brain data that are common across participants; the script fits the model in a particular region of the brain and applies it to held-out data to map individual participant data into the shared functional space


