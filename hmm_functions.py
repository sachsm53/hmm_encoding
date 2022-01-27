#!/usr/bin/env python3

import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os 
import glob
import time
import pickle
import argparse
import numpy as np
import brainiak
import brainiak.eventseg.event
#import brainiak.utils.utils
from brainiak.utils import utils
#import deepdish as dd
import h5py
from scipy import stats
from datetime import datetime
from scipy.stats import norm, zscore, pearsonr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json

startTime = datetime.now()

####### Set up command line
# Initiate the parser
#parser = argparse.ArgumentParser()

# Add long and short argument
#parser.add_argument("--sl", "-sl", help="set searchlight number")

# Read arguments from the command line
#args = parser.parse_args()

# #print(sys.argv[1])
# roiname = int(sys.argv[1])
# print(roiname)

# roiname = int(sys.argv[1])
# print(roiname)

# # # Check for --sl
# if type(sl) == str:
#     print("Running HMM %d" %roiname)
# else:
#     print("No ROI input given. Exiting")
#     exit()

# sl = args.sl 

######## Set up functions
def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

#old way, includes 6 motion, FD, all compcors, and all cosines
def make_conf(conffile):
		#Create an array of regression coefficients
		conf = np.genfromtxt(conffile, names=True)
		#conf = pd.read_csv(conffile,sep = '\t')
		#Check the average framewise displacement: 
		try:
			fd = np.nan_to_num(conf['FramewiseDisplacement'])
		except ValueError:
			fd = np.nan_to_num(conf['framewise_displacement'])
		avg = fd.mean()
# 		if avg > 0.5:
#  			if task == 'DM':
#  				dmcount = dmcount + 1
#  			elif task == "TP":
#  				tpcount = tpcount + 1
			#print(dmcount,tpcount,task,avg,'mm of movement for',sub,s)
			#print(avg,'mm of movement for',sub,s)
			#continue

		#check scrub motion outliers 
		try:
			scrub = np.column_stack([conf[k] for k in conf.dtype.names if 'outlier' in k])
			print(scrub.shape[1],'trs to be scrubbed')
			if scrub.shape[1] > (0.5*scrub.shape[0]):
				print(scrub.shape[1],'trs to be scrubbed')
		except ValueError:
			pass

		#6 motion parameters (want plus the differentials (12)?)
		try:
			motion = np.column_stack((conf['X'],
								  conf['Y'],
								  conf['Z'],
								  conf['RotX'],
								  conf['RotY'],
								  conf['RotZ']))
		except ValueError:
			motion = np.column_stack((conf['trans_x'],
					  conf['trans_y'],
					  conf['trans_z'],
					  conf['rot_x'],
					  conf['rot_y'],
					  conf['rot_z']))
		try:
			compcor = np.column_stack(([conf[k] for k in conf.dtype.names if 'aComp' in k]))
			cosines = np.column_stack([conf[k] for k in conf.dtype.names if 'Cosine' in k])
			csf = conf['CSF']
			wm = conf['WhiteMatter']
		except ValueError:
			compcor = np.column_stack(([conf[k] for k in conf.dtype.names if 'a_comp' in k]))
			cosines = np.column_stack([conf[k] for k in conf.dtype.names if 'cosine' in k])
			csf = conf['csf']
			wm = conf['white_matter']
		reg = np.column_stack((csf,
						   wm,
						   fd,
						   compcor,    
						   cosines,
						   motion,
							np.vstack((np.zeros((1, motion.shape[1])),
								  np.diff(motion, axis=0)))
							  ))

		#replace NAN with zeros if in first row
		np.nan_to_num(reg[0,:],copy = False)
		print('Conf file is', reg.shape)
		return reg

#take the first five components of compcor or based on variance (this isnt done yet)
model =  {'confounds':
		  ['trans_x', 'trans_y', 'trans_z',
		   'rot_x', 'rot_y', 'rot_z', 'cosine'],
		  'aCompCor': [{'n_comps': 5, 'tissue': 'CSF'},
					   {'n_comps': 5, 'tissue': 'WM'}]}
	#eventually add meta
def make_conf2(conffile,confmeta):
		#Create an array of regression coefficients
		conf = np.genfromtxt(conffile, names=True)
		conf_metadata = json.load(open(confmeta))
		
		#conf = pd.read_csv(conffile,sep = '\t')

		#Check the average framewise displacement: 
		try:
			fd = np.nan_to_num(conf['FramewiseDisplacement'])
		except ValueError:
			fd = np.nan_to_num(conf['framewise_displacement'])
		avg = fd.mean()
		if avg > 0.5:
			print(avg,'mm of movement')

		#check scrub motion outliers 
		try:
			scrub = np.column_stack([conf[k] for k in conf.dtype.names if 'outlier' in k])
			print(scrub.shape[1],'trs to be scrubbed')
			if scrub.shape[1] > (0.5*scrub.shape[0]):
				print(scrub.shape[1],'trs to be scrubbed')
		except ValueError:
			pass

		#6 motion parameters (want plus the differentials (12)?)
		try:
			motion = np.column_stack((conf['X'],
								  conf['Y'],
								  conf['Z'],
								  conf['RotX'],
								  conf['RotY'],
								  conf['RotZ']))
		except ValueError:
			motion = np.column_stack((conf['trans_x'],
					  conf['trans_y'],
					  conf['trans_z'],
					  conf['rot_x'],
					  conf['rot_y'],
					  conf['rot_z']))
			


		try:
			compcor = np.column_stack(([conf[k] for k in conf.dtype.names if 'aComp' in k]))
			cosines = np.column_stack([conf[k] for k in conf.dtype.names if 'Cosine' in k])
		except ValueError:
			compcor = np.column_stack(([conf[k] for k in conf.dtype.names if 'a_comp' in k]))
			cosines = np.column_stack([conf[k] for k in conf.dtype.names if 'cosine' in k])
		
		#First 5 components of compocor for CM and WM
		compcor_csf = {c: conf_metadata[c] for c in conf_metadata if conf_metadata[c]['Mask'] == 'CSF'}
		compcor_wm = {c: conf_metadata[c] for c in conf_metadata if conf_metadata[c]['Mask'] == 'WM'}
		n_comps = 5
		if len(compcor) >= n_comps:
			comp_selector = compcor[:n_comps]
		else:
			comp_selector = compcor
			print(f"Warning: Only {len(compcor)} "f"components available ({n_comps} requested)")

		reg = np.column_stack((conf['CSF'],
						   conf['WhiteMatter'],
						   fd,
						   comp_selector,    
						   cosines,
						   motion,
							np.vstack((np.zeros((1, motion.shape[1])),
								  np.diff(motion, axis=0)))
							  ))

		#replace NAN with zeros if in first row
		np.nan_to_num(reg[0,:],copy = False)
		print('Conf file is', reg.shape)
		return reg

def match_z(proposed_bounds, gt_bounds, num_TRs,nPerm):
    threshold = 6/tr
    np.random.seed(0)

    gt_lengths = np.diff(np.concatenate(([0],gt_bounds,[num_TRs])))
    match = np.zeros(nPerm + 1)
    for p in range(nPerm + 1):
        gt_bounds = np.cumsum(gt_lengths)[:-1]
        for b in gt_bounds:
            if np.any(np.abs(proposed_bounds - b) <= threshold):
                match[p] += 1
        match[p] /= len(gt_bounds)
        gt_lengths = np.random.permutation(gt_lengths)
    
    return match,(match[0]-np.mean(match[1:]))/np.std(match[1:])    

def entropy_p(entropy_array, gt_bounds, num_TRs, nPerm):
    np.random.seed(0)
    gt_lengths = np.diff(np.concatenate(([0],gt_bounds,[num_TRs])))
    avg_entrop = np.zeros(nPerm + 1)
    for p in range(nPerm + 1):
        gt_bounds = np.cumsum(gt_lengths)[:-1]
        for b in gt_bounds:
            avg_entrop[p] += entropy_array[b]
        avg_entrop[p] /= len(gt_bounds)
        gt_lengths = np.random.permutation(gt_lengths)
    
    return avg_entrop

#6) running the hmm and getting outputs
def runhmm(braindata,k,ntr):
    hmm = brainiak.eventseg.event.EventSegment(n_events=k,split_merge = True)
    hmm.fit(braindata)
    bestfit_events = np.argmax(hmm.segments_[0], axis=1)
    bestfit_bounds = np.where(np.diff(bestfit_events))[0]
    event_array = np.arange(0,k)
    hmm_prob = np.dot(hmm.segments_[0],event_array)
    hmm_diff = np.zeros((ntr))
    hmm_diff[1:ntr] = np.diff(hmm_prob)
    #print(s,'HMM and Behavior bounds are',len(bestfit_bounds),len(events_run))

    events_entrop = np.zeros((ntr))
    for tr in range(ntr):
        entrop = stats.entropy(hmm.segments_[0][tr,:])
        events_entrop[tr] = entrop
    
    #added to match loglikelihood
    hmm_null = brainiak.eventseg.event.EventSegment(n_events=2,split_merge = True)
    hmm_null.fit(braindata)
    
    return hmm.segments_[0],bestfit_events,bestfit_bounds,events_entrop, hmm_diff,hmm.ll_[-1][0],hmm_null.ll_[-1][0],hmm.event_pat_

# 5) Within-versus-across boundary correlations (need ground truth boundaries and ROI data averaged across people, but not voxels)
def withinacross(bounds,submean_data,nPerm,ntr,w):
#used to have a loop for window for wi,w in enumerate(win_range): # windows in range 5 - 10 sec
    np.random.seed(0)

    events = bounds2events(bounds,ntr)
    _, event_lengths = np.unique(events, return_counts=True)

    corrs = np.zeros(ntr-w)
    for t in range(ntr-w):
        corrs[t] = pearsonr(submean_data[t],submean_data[t+w])[0]

    bound_corrs = np.zeros((nPerm+1))
    for p in range(nPerm+1):
        # Test within minus across boudary pattern correlation with held-out subjects
        within_r = np.mean(corrs[events[:-w] == events[w:]])
        across_r = np.mean(corrs[events[:-w] != events[w:]])
        #print(p,wi,w,within_r -across_r)
        bound_corrs[p] = (within_r - across_r)

        #####
        # Randomize the placement of event lengths and in next loop, calculate again
        perm_lengths = np.random.permutation(event_lengths)
        events = np.zeros(ntr, dtype=np.int)
        events[np.cumsum(perm_lengths[:-1])] = 1
        events = np.cumsum(events)

    #pval = brainiak.utils.utils.p_from_null(bound_corrs[0], bound_corrs[1:nPerm+1], side = 'right',exact=False, axis=None) #side='right'
    pval = utils.p_from_null(bound_corrs[0], bound_corrs[1:nPerm+1], side = 'right',exact=False, axis=None) #side='right'
    zstat = (bound_corrs[0] - bound_corrs[1:].mean())/bound_corrs[1:].std()
    pval_from_z = stats.norm.sf((zstat)) #one-sided
    return bound_corrs,zstat,pval_from_z,pval

#3) Plotting timepoint by time point matrix with bounds
def plot_tt_similarity_matrix(ax, data_matrix, bounds, n_TRs, title_text):
    ax.imshow(np.corrcoef(data_matrix.T), cmap='viridis')
    ax.set_title(title_text)
    ax.set_xlabel('TR')
    ax.set_ylabel('TR')
    # plot the boundaries 
    bounds_aug = np.concatenate(([0],bounds,[n_TRs]))
    for i in range(len(bounds_aug)-1):
        rect = patches.Rectangle(
            (bounds_aug[i],bounds_aug[i]),
            bounds_aug[i+1]-bounds_aug[i],
            bounds_aug[i+1]-bounds_aug[i],
            linewidth=2,edgecolor='w',facecolor='none'
        )
        ax.add_patch(rect)

#Sliding window to determine which graph looks the best
def plot_tt_simmat_windows(submean_data, bounds, ntr, title_text,event_select):
	windows = np.arange(0,len(bounds))
	#f, axs = plt.subplots(50,5, figsize = (500,350))
	for i,e in enumerate(windows):
		if bounds[e] > bounds[-event_select]: 
			break
		event_part = bounds[i:i+event_select]
		submean_data_part = submean_data[event_part[0]:event_part[event_select-1],:]
		bound_part =  event_part - bounds[i]
		ntr_part = event_part[len(event_part)-1] - event_part[0]
		print(i,e,event_part,bound_part,ntr_part,submean_data_part.shape)
		title_text = '%s to %s' %(event_part[0],event_part[event_select-1])
		f, ax = plt.subplots(1,1, figsize = (6,6))
		plot_tt_similarity_matrix(ax, submean_data_part.T, bound_part, ntr_part,title_text)

# 		hmm_event_part = np.array([elem for elem in bestfit_bounds if elem in np.arange(event_part[0],(event_part[0]+ntr_part))])
# 		if len(hmm_event_part) > 0:
# 			title_text = '%s to %s' %(emo_event_part[0],emo_event_part[event_select-1])
# 			f, ax = plt.subplots(1,1, figsize = (6,6))
# 			hmm_bound_part =  hmm_event_part - hmm_event_part[0]
# 			plot_tt_similarity_matrix(ax, submean_data_part.T, hmm_bound_part, ntr_part,title_text)
# 			figname = '/home/msachs/emo_events_forrest/tr-tr-png-sliding-window/tr-tr-hmm_bounds_lipl_seg%d.png' %i
# 			f.savefig(figname,bbox_inches="tight")

# 		else:
# 			title_text = '%s to %s' %(emo_event_part[0],emo_event_part[event_select-1])
# 			f, ax = plt.subplots(1,1, figsize = (6,6))
# 			plot_tt_similarity_matrix(ax, submean_data_part.T, bound_part, ntr_part,title_text)

# 4) converting from boundary moments array to events array #UPDATED TO INCLUDE TR!
def bounds2events(bounds,ntr):
    events = np.repeat(len(bounds),ntr)
    for i,b in enumerate(bounds):
        if i == 0: #first event
            events[0:b+1] = i
        else:
            events[bounds[i-1]+1:b+1] = i
            
#         elif i != len(bounds)-1: 
#             print(i,bounds[i-1]+1,b+1)
            
#         else: #last event
#             events[b+1:ntr] = i+1
#             print(i,bounds[i-1]+1,b+1)
    
    #check results 
    test_bounds = np.where(np.diff(events))[0]
    if not np.array_equal(test_bounds,np.array(bounds)):
        print('Something went wrong with bounds2events')
    return events

# run group permutatons and recalculate hmm
def groupperm(groupdata,g1mask,g2mask,k,emo_in_event,ntr_event,nPerm):
	np.random.seed(1)
	g1nmask = np.ones(np.count_nonzero(g1mask), dtype=bool)
	g2nmask = np.zeros(np.count_nonzero(g2mask),dtype=bool)
	pentrop_lo = np.zeros((ntr_event,nPerm+1))
	pentrop_hi = np.zeros((ntr_event,nPerm+1))
	for p in range(nPerm+1):
		submean_hi = groupdata[:,:,g2mask].mean(axis = 2)
		submean_lo = groupdata[:,:,g1mask].mean(axis = 2)
		hmmseg_hi,events_hi,bounds_hi,entrop_hi,hmmdiff_hi, ll_hi, llnull_hi,pattern_hi = runhmm(groupdata[:,:,g2mask].mean(axis = 2),k+1,ntr_event)
		hmmseg_lo,events_lo,bounds_lo,entrop_lo,hmmdiff_lo, ll_lo, llnull_lo,pattern_lo = runhmm(groupdata[:,:,g1mask].mean(axis = 2),k+1,ntr_event)
		#avgentr_hi[p] = entropy_p(entrop_hi, emo_in_event, ntr_event, 1)[0]
		#avgentr_lo[p] = entropy_p(entrop_lo, emo_in_event, ntr_event, 1)[0]
		pentrop_lo[:,p] = entrop_lo
		pentrop_hi[:,p] = entrop_hi
		#print(p,entrop_lo[5],entrop_hi[5],np.count_nonzero(g1mask),np.count_nonzero(g2mask))
		
		g1mask = np.concatenate([g1nmask,g2nmask])
		np.random.shuffle(g1mask)
		g2mask = ~g1mask
		
	return pentrop_lo,pentrop_hi
		
		
		# hmm_tg.fit([submean_lo,submean_hi])
#         #eseg, ll = hmm_tg.find_events(submean_hi)
#         hmm_problo = np.dot(hmm_tg.segments_[0],np.arange(0,kcount+1)) #time points by events 0 to k minues
#         hmm_probhi = np.dot(hmm_tg.segments_[1],np.arange(0,kcount+1)) #time points by events 0 to k minues
#         hmm_groups[roiname]['aoc_hi-lo'][p] = hmm_probhi.sum() - hmm_problo.sum()
#         hmm_groups[roiname]['ll_hi-lo'][p] = hmm_tg.ll_[-1][1] - hmm_tg.ll_[-1][0]
#         if p == 0:
#             hmm_groups[roiname]['himfq_seg'] =  hmm_tg.segments_[1]
#             hmm_groups[roiname]['lomfq_seg'] =  hmm_tg.segments_[0]

# ####### Define variables 
# datapath = '/'
# slpath = '/rigel/psych/users/ms5924/studyforrest_preproc_sl/preproc_sls'
# behav_events = os.path.join(path,'emo_event_bounds_15ss_peaks_runs.pkl')
# nPerm = 1000


# ####### Load Variables 
# with open(behav_events, "rb") as fp:   # Unpickling
#     emo_event_bounds = pickle.load(fp)

# ####### Run script/loop

# #run the HMM separately for runs 

# outputname = '%s/hmm_felt_%d.h5' %(datapath,roinane) 

# def 
# if os.path.exists(outputname):
# 	with open(outputname, "rb") as fp:   # Unpickling
# 		allrun_results = pickle.load(fp)
# else:
# 	allrun_results = {}
# 	entropy_avg = np.zeros((nPerm+1))

# run_lls = np.zeros((len(run_trs),2))

# #check if already run 
# try:
# 	allrun_results['ll']
# 	print(sl,'searchlight is completed finished. Moving on...',allrun_results['ll'])
# 	exit()
# except KeyError:
# 	try:
# 		allrun_results[run]
# 		print(run,run_length, 'run is finished. Continuing where left off...',len(allrun_results[run]))
# 		continue
# 	except KeyError:
# 		allrun_results[run] = {}

# #load events for thsi run
# events_run = emo_event_bounds[run]
# k = len(events_run) + 1
# print('Number of behav boundaries:',len(events_run))

# #load the h5 searchlight data file 
# datafname = '%s/run%d_rh_quick.h5' %(slpath,run)
# if os.path.getsize(datafname) > 0:
# 	with open(datafname, "rb") as fp:   # Unpickling (saving as h5 faster)
# 		sl_data_run = pickle.load(fp)
# else:
# 	print('Data was not uploaded correctly')
# 	exit()

# #load the correct searchlight and get mean across subjects
# try:
# 	sl_data_run[str(sl)].mean(axis = 2)
# except KeyError:
# 	print(sl,'does not exist for this hemisphere')
# 	exit()

# #check how many vertices have zero 
# submean_data_run = sl_data_run[str(sl)].mean(axis = 2)
# print(datafname,submean_data_run.mean())
# zeroresult = np.all((submean_data_run == 0), axis=0)
# misvert = np.count_nonzero(zeroresult) #true where all zero, this counts the number of true, i.e. number of missing voxels
# if np.count_nonzero(~zeroresult) < 50:
# 	print(misvert,'Too few vertices for this searchlight. Aborting....')
# 	with open('%s/hmm_felt_rh_sl_%d_misvert.h5' %(datapath,sl), "wb") as fp: #save so i know it didn't work
# 		pickle.dump(allrun_results, fp,protocol=2)
# 	exit()
# elif misvert !=0:
# 	print('Removing',misvert,'vertices')
# 	submean_data_run = submean_data_run[:,~zeroresult]

# print(submean_data_run.shape)

# #run the HMM
# hmm = brainiak.eventseg.event.EventSegment(n_events=k,split_merge = True)
# print(run,'Fitting the HMM with',k,'events and timepoints:',submean_data_run.shape[0])
# hmm.fit(submean_data_run)
# bestfit_events = np.argmax(hmm.segments_[0], axis=1)
# bestfit_bounds = np.where(np.diff(bestfit_events))[0]
# print(run,'HMM and Behavior bounds are',len(bestfit_bounds),len(events_run))

# #check the fit of HMM
# hmm_null = brainiak.eventseg.event.EventSegment(n_events=2,split_merge = True)
# hmm_null.fit(submean_data_run)
# #ll_diffs = hmm.ll_[-1][0] - hmm_null.ll_[-1][0]
# run_lls[run,0] = hmm.ll_[-1][0] 
# run_lls[run,1] = hmm_null.ll_[-1][0] 
# #print(run,'real ll:',hmm.ll_[-1],hmm_null.ll_[-1])

# #get entropy at behav boundaries
# events_entrop = np.zeros((run_length))
# #print(bestfit_events[revs])
# for tr in range(run_length):
# 	entrop = scipy.stats.entropy(hmm.segments_[0][tr,:])
# 	events_entrop[tr] = entrop

# entropy_perm = entropy_p(events_entrop,events_run,run_length,nPerm)
# entropy_avg += entropy_perm

# #save all results 
# allrun_results[run]['entropy'] = events_entrop
# allrun_results[run]['entropy_perm'] = entropy_perm
# allrun_results[run]['hmm_segment'] = hmm.segments_[0]
# allrun_results[run]['k'] = k
# allrun_results[run]['ll'] = hmm.ll_[-1][0]
# allrun_results[run]['ll_null'] = hmm_null.ll_[-1][0]

# # Get the average entropy across all runs and permutations
# entropy_avg = entropy_avg/len(run_trs)
# #pval = brainiak.utils.utils.p_from_null(entropy_avg[0], entropy_avg[1:nPerm+1], side = 'right',exact=False, axis=None) #side='right'
# #allrun_results['pval'] = pval
# allrun_results['avg_entropy_perm'] = entropy_avg
# lls_diff = np.mean(run_lls[:,0]) - np.mean(run_lls[:,1])
# allrun_results['ll_diffs'] = lls_diff
# print(sl,entropy_avg[0],round(np.mean(entropy_avg[1:nPerm+1]),3),round(lls_diff,3))

# if lls_diff <  20:
# 	print('HMM Fit is not good, diff:',lls_diff)
# 	with open('%s/hmm_felt_rh_sl_%d_poorhmm.h5' %(datapath,sl), "wb") as fp: #save so i know it didn't work
# 		pickle.dump(allrun_results, fp,protocol=2)
# else:
# 	with open(outputname, "wb") as fp:   #Pickling
# 		pickle.dump(allrun_results, fp,protocol=2)
# 		print('      Saving as',outputname)

# print(datetime.now() - startTime)