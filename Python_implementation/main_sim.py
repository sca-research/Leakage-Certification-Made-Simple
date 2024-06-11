#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:12:08 2024

@author: IWAS\choaak
"""

import numpy as np
from  Lkg_gen import *
from tqdm import tqdm
from GKOV_MI import *


def main():
    
    
    # Converegence Analysis Experiments for different MI estimators------------
    sig =  2 # Mention the standard deviation of noise
    
    n_trial = 1 # Number of repititons of experiments (for n_samples > 1e05 take value = 1)
    
    # n_samples = np.array(range(10000, 120000, 20000))
    n_samples = np.array(range(50000, 1500000, 200000)) # for bivariate traces
    
    for i in range(len(n_samples)):
        
        print("Number of traces = {}".format(n_samples[i]))
        
        # Giving the t_n functional value in terms of number of sample size 
        t_value = round(np.log(n_samples[i]))
        # t_value = round( np.log10(q[i])**2)
        print("Nearest Neighbour value = {}". format( t_value ))
        
        b = 0
        b1 = 0
        for m in tqdm(range(n_trial)):
            
            Encrypt_key = np.random.randint(0, 256, dtype = np.uint8)
            
            
            # Simulate univariate traces-------------------------------------
            # Tr, plain_text = univariate_trace(Encrypt_key, n_samples[i], sig)
            
            # Simulate Bivariate traces------------------------------------------
            # Tr, plain_text = Bivariate_trace(Encrypt_key, n_samples[i], sig)
            
            # Simulate Multivariate traces---------------------------------------
            Tr, plain_text = Multivariate_trace(Encrypt_key, n_samples[i], sig)
            # MI estimators -------------------------------------------------
            # z = MI_mixt_gao( n_samples[i], t_value, plain_text, np.array(Tr))
            # z = MI_mixt_KSG( n_samples[i], t_value, plain_text, np.array(Tr))
            
            z = MI_gao_multi_(n_samples[i], t_value, plain_text, Tr)
            
            b += z
            b1 += z ** 2
        
        b = b / n_trial
        b1 = (b1 / n_trial) - b ** 2
        
        print("Estimated MI_GKOV = {}".format(b))
        print("Estimated Variance MI_GKOV = {}". format(b1))

if __name__== '__main__':
    main()            
            
        