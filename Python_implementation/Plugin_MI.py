#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:34:23 2024

@author: IWAS\choaak
"""

import numpy as np

#==============================================================================
# -----------------------------------------------------------------------------
# plug-in estimator of MI(ot ; pred_leakage), when observed trace(ot) and intermediate (pred_leakage) are discrete ---------
def mi_plug_in(pred_leakage, ot):
    '''
    Parameters
    ----------
    pred_leakage : A list of univariate 'discrete' predicted leakage
    ot : A list of Univariate 'discrete' Observable Traces

    Returns
    -------
    Plug-in estimate of MI between two set of discrete data

    '''
    
    uniq1 = list(set(ot))
    ot_count = np.array( [ ot.count(i) for i in uniq1]) 
    ot_prob = ot_count/np.sum(ot_count)
    entrop_trc = 0
    for i in range(len(ot_prob)):
        entrop_trc -=  np.log2(ot_prob[i])*ot_prob[i]
        
    all_cond_probs = []
    uniq_lkg = list(set(pred_leakage))
    for i in uniq_lkg:
        cond_tr = [ot[j] for j, l in enumerate(pred_leakage) if l == i]
        uniq2 = list(set(cond_tr))
        cond_tr_count = np.array( [ cond_tr.count(j) for j in uniq2] )
        all_cond_probs.append( cond_tr_count/ np.sum(cond_tr_count))
    
    N = len(all_cond_probs)
    cond_entrps = [0]*N
    for i in range(N):
        for j in range(len(all_cond_probs[i])):
            if(all_cond_probs[i][j] != 0):
                cond_entrps[i] -= np.log2(all_cond_probs[i][j])*all_cond_probs[i][j]
    
        
    pred_lkg_hist = [pred_leakage.count(i) for i in uniq_lkg]
    temp_counts = np.array(pred_lkg_hist)
    pred_lkg_prob = temp_counts/np.sum(pred_lkg_hist)
    
    cond_entrp = 0

    for i in range(N):
        cond_entrp += cond_entrps[i]*pred_lkg_prob[i]
    
    return entrop_trc - cond_entrp

# plug-in estimator of MI( {L_1,L_2} ; Y)---------------------------------------------
# Y: pred_leakage(discrete) and L_i = observed traces(discrete)-----------------------
def mi_plug_in2d(pred_leakage, ot1, ot2):
    '''
    Plug-in estimate of MI between bivariate discrete Trace and univariate discrete 
    Predicted leakage

    Parameters
    ----------
    pred_leakage : a list or array of Univariate discrete predicted leakage (e.g. intermediate) 
    
    ot1 : (ot1, ot2) is the bivariate pair of observable leakage (or traces)
    ot2 : 

    Returns
    -------
    Plug-in MI estimate 

    '''
    
    ot_biv = [[i,j] for (i,j) in zip(ot1,ot2)]
    unique_ot = np.unique( ot_biv, axis = 0)
    count_biv = [ np.count_nonzero(np.logical_and( ot1 == x[0]
                             , ot2 == x[1])) for  x in unique_ot]
    ot_prob = count_biv/np.sum(count_biv)
    entrop_trc = 0
    for i in range(len(ot_prob)):
        entrop_trc -=  np.log2(ot_prob[i])*ot_prob[i]

    all_cond_probs = []
    uniq_lkg = list(set(pred_leakage))
    for i in uniq_lkg:
        cond_tr1 = [ot1[j] for j, l in enumerate(pred_leakage) if l == i]
        cond_tr2 = [ot2[j] for j, l in enumerate(pred_leakage) if l == i]
        cond_biv =  [[i,j] for (i,j) in zip(cond_tr1,cond_tr2)]
        unique_cond = np.unique( cond_biv, axis = 0)
        cond_tr_count = [ np.count_nonzero(np.logical_and( cond_tr1 == x[0]
                                 , cond_tr2 == x[1])) for  x in unique_cond]
        all_cond_probs.append( cond_tr_count/ np.sum(cond_tr_count))
        
    N = len(all_cond_probs)
    cond_entrps = [0]*N
    for i in range(N):
        for j in range(len(all_cond_probs[i])):
            if(all_cond_probs[i][j] != 0):
                cond_entrps[i] -= np.log2(all_cond_probs[i][j])*all_cond_probs[i][j]
    
        
    pred_lkg_hist = [pred_leakage.count(i) for i in uniq_lkg]
    temp_counts = np.array(pred_lkg_hist)
    pred_lkg_prob = temp_counts/np.sum(pred_lkg_hist)
    
    cond_entrp = 0

    for i in range(N):
        cond_entrp += cond_entrps[i]*pred_lkg_prob[i]

    return entrop_trc - cond_entrp
#====================================================================================