#!/usr/bin/env python3

import os
import sys 
import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
import deepdish as dd
import brainiak
from brainiak.utils import utils

np.random.seed(1337)sc
np.set_printoptions(precision=4, suppress=True)

from scipy.stats import zscore
from matplotlib import pyplot as plt

import h5py
from tikreg import models, utils as tikutils
from tikreg import spatial_priors, temporal_priors
from scipy import stats
from datetime import datetime

startTime = datetime.now()
print(startTime)

##############################################
# Path info, Load features and data
tr = 0.8
nTR = 750
trs = np.arange(nTR)
path = '/rigel/psych/users/ms5924/hbn/'
#path = '/home/msachs/hbn'
surfpath = '%s/preprocessed_surfdata/' %path
subcortpath = '%s/preproc_rois_subcort/' %path
#set networks to look at and subcortical rois to look at 

featpaths = ['emos_on_sums.npy','seg_feats.npy','visfeats_tr_class4.npy']
f1 = np.load('%s/%s' %(path,featpaths[0]))
f2 = np.load('%s/%s' %(path,featpaths[1]))
f3 = np.load('%s/%s' %(path,featpaths[2]))
#feats1 = np.hstack([f1,f2])
feats1_conv = brainiak.utils.fmrisim.convolve_hrf(f1,tr_duration = tr,temporal_resolution = 1/tr, scale_function = True)
feats2_conv = brainiak.utils.fmrisim.convolve_hrf(f2,tr_duration = tr,temporal_resolution = 1/tr, scale_function = True)
feats3_conv = brainiak.utils.fmrisim.convolve_hrf(f3,tr_duration = tr,temporal_resolution = 1/tr, scale_function = True)

X1 = stats.zscore(feats1_conv, axis = 0)
X2 = stats.zscore(feats2_conv, axis = 0)
X3 = stats.zscore(feats3_conv, axis = 0)
 

nfeats = X1.shape[1] + X2.shape[1] + X3.shape[1] #CHANGE HERE
print('Features:',X1.shape, X2.shape,X3.shape, nfeats)

##############################################
# solving for multiple of hyperparameters using polar search

# Sampling in terms of ratios and scalings
alphas = np.logspace(0,4,11)[1:9] #changed 09-28-21
ratios = np.logspace(-2,2,9) #change 09-28-21
n_splits = 3 #change from 6 to 2 on 09-28-21
cv = KFold(n_splits=n_splits)

#new people
nsub = 150

#set networks to look at and subcortical rois to look at 
allnetworks = ['Default','Limbic','SalVentAttn','TempPar','Aud']
subcortrois = ['lcaud','lput','lhippo','lamyg','lpal','rcaud','rput','rhippo','ramyg','rpal','nac']

##FIRST MAKE OVERLAP MAPS FOR EACH ROI
subcortrois = ['lcaud','lput','lhippo','lamyg','lnac','lpal','rcaud','rput','rhippo','ramyg','rnac','rpal']
allsurfrois = np.load(path + '/allsurfrois.npy').tolist()
allrois = allsurfrois + subcortrois
allsigs = ['LH_LimbicB_OFC_1.h5','LH_DefaultC_PHC_1.h5','RH_LimbicB_OFC_1.h5',\
		   'RH_SalVentAttnA_ParOper_1.h5','lamyg','lpal','rcaud']

s = int(sys.argv[1])
for roinum,roiname in enumerate(allrois):
# 	roioutdic = '%s/encoding/encode_banded_%s_%dss.h5' %(path,roiname[:-3],nsub)
# 	if os.path.exists(roioutdic):
# 		roidic = dd.io.load(roioutdic)
# 		if len(roidic.keys()) == nsub:
# 			print(roiname, 'is completed. moving on....')
# 		else:
# 			surfdata = dd.io.load(infname)
# 			sublist = np.array([elem for elem in surfdata['sublist'] if elem not in roidic.keys()]).astype('object')
# 			roidic = dd.io.load(roioutdic)
# 	else:
		#roidic = {}
	roistarttime = datetime.now()
	if roiname in allsurfrois:
		infname = '%s%s_15over_%dss.h5' %(surfpath,roiname[:-3],nsub)
	else:
		infname = '%sDM_%s_15over_zscore_nocompcor.h5' %(subcortpath,roiname)
	surfdata = dd.io.load(infname)
	sublist = surfdata['sublist']
	sub = sublist[s]
	subidx = np.where(np.array(surfdata['sublist']) == sub)[0][0]
	if roiname in allsurfrois:
		suboutdic = '%s/encoding/encode_banded_3feat_%s_%s.h5' %(path,roiname[:-3],sub)
	else:
		suboutdic = '%s/encoding/encode_banded_3feat_%s_%s.h5' %(path,roiname,sub)
	if os.path.exists(suboutdic):
		#subdic = dd.io.load(suboutdic)
		print(s,sub,suboutdic,'exists')
		#continue
	braindata = surfdata['data'][:,:,subidx] #changed 11-10
	nvox = braindata.shape[1]
	print(roinum, 'of', len(allrois),roiname,braindata.shape,'working on',s,subidx,sub, suboutdic)
	subdic = {n:np.zeros((nvox,n_splits)) for n in ['corr','alphas','lamda1','lamda2','lamda3']}
	subdic['kweights'] = np.zeros((int(nTR - (nTR/n_splits)),nvox,n_splits))
	subdic['weights'] = np.zeros((nfeats,nvox,n_splits))
	subdic['pred'] = np.zeros((int(nTR/n_splits),nvox,n_splits))
	scount = -1 #this is actually the cv count
	#for training and testing of CV spits, the 0 (1st) split is the first time points (1/6) in testing
	for train, test in cv.split(X=trs):
		scount +=1
		print(scount,'TRAIN:',train[0],train[train.shape[0]-1],'TEST:',test[0],test[test.shape[0]-1])
		X1train = X1[train]
		X1test = X1[test]
		X2train = X2[train]
		X2test = X2[test]
		X3train = X3[train]
		X3test = X3[test]
		Ytrain = braindata[train]
		Ytest = braindata[test]
		temporal_prior = temporal_priors.SphericalPrior(delays=[0]) # no delays - could have hrf as temporal priors in which case don't want to convolve with hrf

		nfeatures1 = X1train.shape[1]
		nfeatures2 = X2train.shape[1]
		nfeatures3 = X3train.shape[1]
		X1_prior = spatial_priors.SphericalPrior(nfeatures1, hyparams=[1.0])
		X2_prior = spatial_priors.SphericalPrior(nfeatures2, hyparams=ratios)
		X3_prior = spatial_priors.SphericalPrior(nfeatures3, hyparams=ratios)
		#population_optimal=False, if troo indivdual response CV values are not kept
		fit_banded_polar = models.estimate_stem_wmvnp([X1train, X2train,X3train], Ytrain,
													  [X1test, X2test,X3test],Ytest,
													  feature_priors=[X1_prior, X2_prior,X3_prior],
													  temporal_prior=temporal_prior,
													  ridges=alphas,           # Solution for all alphas
													  normalize_hyparams=True, # Normalizes the ratios
													  folds=(1,5),
													  performance=True,
													  weights=True,
													  predictions=True,
													  verbosity=True)
		subdic['corr'][:,scount] = np.nan_to_num(fit_banded_polar['performance'].squeeze())
		subdic['alphas'][:,scount] = fit_banded_polar['optima'][:,-1]
		subdic['lamda1'][:,scount] = fit_banded_polar['optima'][:,1]
		subdic['lamda2'][:,scount] = fit_banded_polar['optima'][:,2]
		subdic['lamda3'][:,scount] = fit_banded_polar['optima'][:,3]
		for i in range(0,5):
			print('innercv',i,fit_banded_polar['cvresults'][i,:,:,:,0])

		subdic['kweights'][:,:,scount] = fit_banded_polar['weights'] #TR (of train) x Vox
		subdic['pred'][:,:,scount] = fit_banded_polar['predictions'] #TR (of test) x vox

		# Next, we compute the model weights for each voxel separately. To achieve this, we first find the optimal set of hyperparameters
		#stored the hyperparameters into separate vectors,
		subalphas = subdic['alphas'][:,scount]
		lambda_ones = subdic['lamda1'][:,scount]
		lambda_twos = subdic['lamda2'][:,scount]
		lambda_thre = subdic['lamda3'][:,scount]

		#use matrix multiplication to convert the estimated kernel weights into primal weights
		weights_x1 = np.linalg.multi_dot([X1train.T, fit_banded_polar['weights'], np.diag(subalphas), np.diag(lambda_ones**-2)])
		weights_x2 = np.linalg.multi_dot([X2train.T, fit_banded_polar['weights'], np.diag(subalphas), np.diag(lambda_twos**-2)])
		weights_x3 = np.linalg.multi_dot([X3train.T, fit_banded_polar['weights'], np.diag(subalphas), np.diag(lambda_thre**-2)])

		weights_joint = np.vstack([weights_x1, weights_x2,weights_x3])
		subdic['weights'][:,:,scount] = weights_joint #this is feature x voxel x CV
		print(subdic['alphas'][:,scount].mean(),subdic['lamda1'][:,scount].mean(),subdic['lamda2'][:,scount].mean())
		print('Saving out',suboutdic)
		dd.io.save(suboutdic,dd.io.ForcePickle(subdic))
		print('Time taken for one ROI:',datetime.now() - roistarttime)
		#get preference mapping 
# 		for f in range(subdic['weights'].shape[0]):
# 			vweight = subdic['weights'][f,:,scount]
		#roidic[sub] = subdic
print('Time taken:',datetime.now() - startTime)
# try:
# 	dd.io.save(roioutdic,dd.io.ForcePickle(roidic))
# except OverflowError:
# 	dd.io.save(roioutdic,roidic)
# print('Time taken after saving:',datetime.now() - startTime)
# 	with open(roioutdic, 'wb') as fp:
# 		pickle.dump(roidic, fp)