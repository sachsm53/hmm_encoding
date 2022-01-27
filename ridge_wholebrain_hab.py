#!/usr/bin/env python3

import os
import sys 
import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import deepdish as dd
import brainiak
from brainiak.utils import utils
import math

np.random.seed(1337)
np.set_printoptions(precision=4, suppress=True)

from scipy.stats import zscore
from matplotlib import pyplot as plt

import h5py
from scipy.stats import norm, zscore, pearsonr, ttest_1samp, linregress,ttest_ind
from datetime import datetime

startTime = datetime.now()

##############################################
# Path info, Load features and data
tr = 0.8
nTR = 750
trs = np.arange(nTR)
path = '/rigel/psych/users/ms5924/hbn/'
#path = '/home/msachs/hbn/'
#set networks to look at and subcortical rois to look at 

featpaths = ['emos_on_sums.npy','seg_feats.npy','visfeats_tr_class4.npy']
f1 = np.load('%s/%s' %(path,featpaths[0]))
f2 = np.load('%s/%s' %(path,featpaths[1]))
f3 = np.load('%s/%s' %(path,featpaths[2]))
#feats1 = np.hstack([f1,f2])
feats1_conv = brainiak.utils.fmrisim.convolve_hrf(f1,tr_duration = tr,temporal_resolution = 1/tr, scale_function = True)
feats2_conv = brainiak.utils.fmrisim.convolve_hrf(f2,tr_duration = tr,temporal_resolution = 1/tr, scale_function = True)
feats3_conv = brainiak.utils.fmrisim.convolve_hrf(f3,tr_duration = tr,temporal_resolution = 1/tr, scale_function = True)

X1 = zscore(feats1_conv, axis = 0)
X2 = zscore(feats2_conv, axis = 0)
X3 = zscore(feats3_conv, axis = 0)

#X1 = sklearn.preprocessing.normalize(X1)
#X2 = sklearn.preprocessing.normalize(X2)
#X3 = sklearn.preprocessing.normalize(X3)

nfeats = X1.shape[1] + X2.shape[1] + X3.shape[1] #CHANGE HERE
feats = np.concatenate([X1,X2,X3],axis = 1)
print('Features:',X1.shape, X2.shape,X3.shape, nfeats)


#############################################
#permutation set up
nPerm = 1000
blocksize = int(10) #[from previous paper]
blockarray = np.load('%s/blockshuffle_%dsize_%dperm.npy' %(path,blocksize,nPerm))

##############################################
# Ridge parameters

#new people
nsub = 150
task = 'DM'

# Set up ridge model
outsplit = 6
insplit = 5
cv = KFold(n_splits=outsplit)
innercv = KFold(n_splits=insplit)
alpha_grid = np.geomspace(10.0,10000.0, num=20)

#Set up subject data and output
inp = int(sys.argv[1])
s = math.floor(inp/10)
permset = inp % 10
permranges = permset*100
permrangee = (permset+1)*100
if permrangee == nPerm:
	permrangee += 1

subfolders = [elem for elem in os.listdir('%sparcel_means/' %path) if 'sub' in elem]
subfolder = subfolders[s]
sub = subfolder.split('_')[0]

surfdata = dd.io.load(path + 'parcel_means/' + subfolder)
allrois = surfdata[task]['roinames']
braindata = surfdata[task]['roimean']
nvox = braindata.shape[1]
subdic = {t:np.zeros((outsplit,nvox,nPerm+1)) for t in ['r2','r','alphas']}
subdic['voxpred_cv'] = np.zeros((int(nTR/outsplit),nvox,nPerm+1))
subdic['voxpred_full'] = np.zeros((nTR,nvox,nPerm+1))
subdic['inner_cv'] = {}
subdic['r_alltr'] = np.zeros((nvox,nPerm+1))
subdic['r_foldavg'] = np.zeros((nvox,nPerm+1))
#subdic['zstat_foldavg'] = np.zeros((nvox))
#subdic['pval_foldavg'] = np.zeros((nvox,2))
#subdic['zstat_alltr'] = np.zeros((nvox))
#subdic['pval_alltr'] = np.zeros((nvox,2))
#subdic['r_null'] = np.zeros((outsplit,nPerm+1,nvox))

# CHANGE on 01-13 for whole brain
parcelnum = '400'
suboutdic = '%sencoding_%sparc/encode_banded_3feat_%dperm_%s_%d2%d.h5' %(path,parcelnum,nPerm,sub,permranges,permrangee)
if os.path.exists(suboutdic):
	print(s,subfolder,suboutdic,'exists')
	#sys.exit()

print('Starting on',sub,permranges,'to',permrangee)
for p in range(permranges,permrangee):
	pstartTime = datetime.now()
	outercv = -1
	#apply blockwise shuffling if not first perm
	if p == 0: 
		braindata = braindata
	else:
		braindata = braindata[blockarray[:,p-1]]
	
	for train, test in cv.split(X=trs):
		outercv += 1
		#print('\t',p,sub,'loop',outercv)
		dtrain = braindata[train,:]
		dtest = braindata[test,:]
		#only do interloop if first perm
		#if p == 0:
			#print('\t',p,sub,'entering inner loop',outercv)
		inner_scores = {t:np.zeros((5,nvox,len(alpha_grid))) for t in ['r2','r']}
		innerc = -1
		for ntrain,ntest in innercv.split(X=trs[train]):
			cdtrain = dtrain[ntrain,:]
			cdtest = dtrain[ntest,:] 
			innerc += 1
			for a,alpha in enumerate(alpha_grid):
				npred = Ridge(alpha=alpha).fit(feats[ntrain], cdtrain).predict(feats[ntest])
				inner_scores['r2'][innerc,:,a] = (r2_score(cdtest, npred,multioutput='raw_values'))
				 #also calculate the pearson correlation
				for v in range(nvox):
					inner_scores['r'][innerc,v,a] = pearsonr(cdtest[:,v],npred[:,v])[0]
		subdic['inner_cv'] = inner_scores
		#get best alpha (average of the best for each loop) for each voxel 
		for v in range(nvox):
			max_index_row = np.argmax(inner_scores['r'][:,v,:], axis=1)
			vbest_alpha = 10 ** np.mean(np.log10(alpha_grid[max_index_row]))
			prediction = Ridge(alpha=vbest_alpha).fit(feats[train], dtrain[:,v]).predict(feats[test])
			subdic['r'][outercv,v,p] = pearsonr(dtest[:,v],prediction)[0]
			subdic['alphas'][outercv,v,p] = vbest_alpha
			subdic['voxpred_cv'][:,v,p] = prediction
			subdic['r2'][outercv,v,p] = 1 - sum((prediction - dtest[:,v])**2)/sum((np.mean(dtrain[:,v]) - dtest[:,v])**2)
			subdic['voxpred_full'][test,v,p] = prediction
	#correlate entire time series for each voxel 
	for v in range(nvox):
		subdic['r_alltr'][v,p] = pearsonr(subdic['voxpred_full'][:,v,p],braindata[:,v])[0]
		subdic['r_foldavg'][v,p] = subdic['r'][:,v,p].mean()
	
	print(s,sub,'Max voxel correlation is:', allrois[np.where(subdic['r_alltr'][:,p] == subdic['r_alltr'][:,p].max())][0], subdic['r_alltr'][:,p][np.where(subdic['r_alltr'][:,p] == subdic['r_alltr'][:,p].max())][0])
	print(s,sub,'Max voxel fold avg is:', allrois[np.where(subdic['r_foldavg'][:,p] == subdic['r_foldavg'][:,p].max())][0], subdic['r_foldavg'][:,p][np.where(subdic['r_foldavg'][:,p] == subdic['r_foldavg'][:,p].max())][0])
	
	print('Time taken for perm:',p,datetime.now() - pstartTime)
	
	#Apply blockwise to braindata
	#braindata = braindata[blockarray[:,p-1]]
	
	#save out if multiple of 100
	#if not p % 100:
		#print(p)
	print('Saving out',suboutdic,'at',p, subdic.keys())
	dd.io.save(suboutdic,subdic)

#get the z-stat and pvals for each 
#dd.io.save(suboutdic,subdic)

# for v in range(nvox):
# 	subdic['zstat_foldavg'][v] = (subdic['r_foldavg'][v,0] - subdic['r_foldavg'][v,1:].mean())/subdic['r_foldavg'][v,1:].std()
# 	subdic['zstat_alltr'][v] = (subdic['r_alltr'][v,0] - subdic['r_alltr'][v,1:].mean())/subdic['r_foldavg'][v,1:].std()
# 	subdic['pval_foldavg'][v,0] = norm.sf((subdic['zstat_foldavg'][v]))
# 	subdic['pval_foldavg'][v,1] = utils.p_from_null(subdic['r_foldavg'][v,0], subdic['r_foldavg'][v,1:], side = 'right',exact=False, axis=None)
# 	subdic['pval_alltr'][v,0] = norm.sf((subdic['zstat_alltr'][v]))
# 	subdic['pval_alltr'][v,1] = utils.p_from_null(subdic['r_alltr'][v,0], subdic['r_alltr'][v,1:], side = 'right',exact=False, axis=None)

# dd.io.save(suboutdic,subdic)
print('Total time:',p,datetime.now() - startTime)