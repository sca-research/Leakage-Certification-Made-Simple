#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:46:47 2024

@author: IWAS\choaak
"""
import h5py
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from GKOV_MI import MI_gao_multi, MI_mixt_gao

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset



from TI_PI_MLP import train_mlp, load_model_

Reverse_AES_Sbox = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,                                                                                                                                                                                       
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d]


MODEL_FOLDER = ''


def run_lr(N_prof, N_val, trace_prof, label_prof, trace_val, label_val):
    """ 
    Profiling and calculating the log_probability score using logistic regression 
    
    """

    len_prof = len(label_prof)
    sample_idx =  np.random.choice( len_prof , N_prof, replace = False)
    X = trace_prof[sample_idx]
    y = label_prof[sample_idx]
    
    '''Running Different Training models'''
    # clf = LogisticRegression(solver = 'newton-cholesky', n_jobs = -1)
    clf = GaussianNB()
    # clf = LDA()
    clf.fit(X, y)

    # # Generates validation dataset
    len_val = len(label_val)
    smpl_idx =  np.random.choice( len_val , N_val, replace = False)
    X_val = trace_val[smpl_idx]
    y_val = label_val[smpl_idx]
   
    # # Predicts the scores
    scores_lr = 0
    scores_lr_prof = 0
    for i in range(256):
        X_ = X[np.where(y == i)]
        x_ = X_val[np.where(y_val == i)]        
        if len(x_) > 0: 
            x_score =  clf.predict_log_proba(x_)[:, i]
            scores_lr += x_score.mean()
            pass
        if len(X_) > 0: 
            x_prof_score = clf.predict_log_proba(X_)[:, i]
            scores_lr_prof += x_prof_score.mean()
            pass
        pass      

    scores_lr = scores_lr / 256
    scores_lr_prof = scores_lr_prof / 256               

    return scores_lr, scores_lr_prof



def run_MI_model(N_prof, N_val, trace_prof, label_prof, trace_val, label_val):
    
    len_prof = len(label_prof)
    sample_idx =  np.random.choice( len_prof , N_prof, replace = False)
    X = trace_prof[sample_idx]
    y = label_prof[sample_idx]
    
    '''Running Different Training models'''
    clf = LogisticRegression(solver = 'newton-cholesky', n_jobs = -1)
    # clf = GaussianNB()
    # clf = LDA()
    clf.fit(X, y)

    
    # # Generates validation dataset
    len_val = len(label_val)
    smpl_idx =  np.random.choice( len_val , N_val, replace = False)
    X_val = trace_val[smpl_idx]
    y_val = label_val[smpl_idx]
    
    y_predict =clf.predict(X_val)
    t_value = int(np.log(N_val))
    MI_value = MI_mixt_gao( N_val, t_value, y_predict, y_val)
    
    return MI_value
    


def run_MI_MLP(N_prof, N_val, trace_prof, label_prof, trace_val, label_val):
    
    len_prof = len(label_prof)
    sample_idx =  np.random.choice( len_prof , N_prof, replace = False)
    X = trace_prof[sample_idx]
    y = label_prof[sample_idx]
    

    '''Calling MLP_model'''
    train_mlp(X,y, N_prof)
    clf = load_model_(X.shape[1], N_prof)
    # clf = mlp_task(X.shape[1])
    # file_name = 'aes_hd_model{}'.format(N_prof)
    # clf.load_weights(MODEL_FOLDER + file_name +'.h5')
    
    # # Generates validation dataset
    len_val = len(label_val)
    smpl_idx =  np.random.choice( len_val , N_val, replace = False)
    X_val = trace_val[smpl_idx]
    y_val = label_val[smpl_idx]
    
    y_pred_prob = clf.predict(X_val, verbose = 0)['x']
    y_predict = np.argmax(y_pred_prob, axis = 1)
    
    t_value = int(np.log10(N_val)) ** 3
    
    # t_value = int(np.log(N_val))
    MI_value = MI_mixt_gao( N_val, t_value, y_predict, y_val)
    
    return MI_value




def compute_PI_TI(N_prof_range, N_val, trace_prof, label_prof, trace_val, label_val):
    
    PIs_lr, TIs_lr = [], []
    for N_prof in N_prof_range:
        print(f"Number of profile traces = {N_prof}")
        PI_lr = 0
        TI_lr = 0
        n_exp = 20
        
        for _ in range(n_exp):
            score_lr, score_lr_prof = run_lr(N_prof, N_val, trace_prof, label_prof, trace_val, label_val)
            PI_lr += 8 + score_lr
            TI_lr += 8 + score_lr_prof
            
        PIs_lr.append(PI_lr/n_exp)
        TIs_lr.append(TI_lr/n_exp)
    
    PIs_lr = np.array(PIs_lr)
    TIs_lr = np.array(TIs_lr)
    
    return PIs_lr, TIs_lr

def compute_GKOV_MI(N_prof_range, N_val, trace_prof, label_prof, trace_val, label_val):
    
    gkov_mis = []
    for N_prof in N_prof_range:
        print(f"Number of profiling traces = {N_prof}")
        gkov_mi = 0
        n_exp = 10
        
        for _ in range(n_exp):
            gkov_mi += run_MI_model(N_prof, N_val, trace_prof, label_prof, trace_val, label_val)
            # gkov_mi += run_MI_MLP(N_prof, N_val, trace_prof, label_prof, trace_val, label_val)
        gkov_mis.append(gkov_mi / n_exp)
        
    return gkov_mis
    
    





def main():

              

    # load data----------------------------
    TS = np.load('aes_hd_ext.npz')
    
    Input = np.array( TS['data'][:, 16:32], dtype = np.uint8)
    
    Input_1 = np.array(Input[:, 11], dtype = np.uint16)
    Input_2 = np.array(Input[:, 7], dtype = np.uint16)
    
    key = np.array( TS['data'][:, 32:], dtype = np.uint8)[:, 11]
    
    Trace = np.array(TS['traces'], dtype = np.int16)
    
    n_trace = len(Trace)
    X = np.array([ Reverse_AES_Sbox[Input_1[i] ^ key[0]]^Input_2[i]
                  for i in np.arange(0, n_trace)], dtype = np.uint16)
    
    idx = [539, 540, 542, 543, 965, 966, 967, 968, 969, 1021]
    T_prof = Trace[: int(4.5e05) ]
    Y_prof = X[: int(4.5e05)]
    T_valid = Trace[ int(4.5e05): ]
    Y_valid = X[ int(4.5e05):]
    

   
    N_prof_range = np.logspace(3.5, 5.5, 10).astype('int32')
    # N_prof_range = np.logspace(3.5, 5.5, 10).astype('int32')
    # N_prof_range = [14677]
    n_repeats = 1
    PI_mean, TI_mean = [], []
    # GKOV_MI_mean = []
    for _ in range(n_repeats):
        PIs_lr, TIs_lr = compute_PI_TI(N_prof_range, int(1e04), T_prof[:, idx], Y_prof, T_valid[:, idx], Y_valid)
        # gkov_mis = compute_GKOV_MI(N_prof_range, int(5e04), T_prof[:, idx], Y_prof, T_valid[:, idx], Y_valid)
        PI_mean.append(PIs_lr)
        TI_mean.append(TIs_lr)
        # GKOV_MI_mean.append(gkov_mis)


    PI_mean = np.array(PI_mean).mean(axis=0)
    TI_mean = np.array(TI_mean).mean(axis=0)
    # GKOV_MI_mean =  np.array(GKOV_MI_mean).mean(axis=0)
    
    print(f" PI_values {PI_mean}")
    print(f" TI_values {TI_mean}") 
    # print(f" GKOV_MI values {GKOV_MI_mean}") 
    
if __name__ == "__main__":
    main()



def main1():
    TS = np.load('aes_hd_ext.npz')
    
    Input = np.array( TS['data'][:, 16:32], dtype = np.uint8)
    
    Input_1 = np.array(Input[:, 11], dtype = np.uint16)
    Input_2 = np.array(Input[:, 7], dtype = np.uint16)
    
    key = np.array( TS['data'][:, 32:], dtype = np.uint8)[:, 11]
    
    Trace = np.array(TS['traces'], dtype = np.int16)
    
    n_trace = len(Trace)
    X = np.array([ Reverse_AES_Sbox[Input_1[i] ^ key[0]]^Input_2[i]
                  for i in np.arange(0, n_trace)], dtype = np.uint16)
    
    idx = [539, 540, 542, 543, 965, 966, 967, 968, 969, 1021]
    
    t_n = int(np.log(n_trace))
    # MI_value = MI_gao_multi( n_trace, t_n, X, Trace[:, idx]) # 3.1807564712296204
    
    fig, ax = plt.subplots()
    TINY_SIZE = 10
    SMALL_SIZE = 10
    MEDIUM_SIZE = 10

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=TINY_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=TINY_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    

    
    
    N_prof_range_logistic, PI_mean_logistic, TI_mean_logistic = np.load('TI_PI_logistic.npy')
    N_prof_range_GT, PI_mean_GT, TI_mean_GT = np.load('TI_PI_GT.npy')
    N_prof_range_MLP, PI_mean_MLP, TI_mean_MLP = np.load('AES_HD/TI_PI_MLP.npy')
    
    
    N_prof_range_gkov =  np.logspace(3.5, 5.5, 10).astype('int32')
    
    gkov_mis_MLP = [0.66516056, 0.64705979, 0.63682915, 0.59384923, 0.74363808, 0.6977572,
    0.67144827, 0.70210949, 0.68681934, 0.71599029]
    gkov_mis_logistic =  [1.24159857, 1.17457378, 1.14770238, 1.1243134, 1.0762696, 1.04624685,
    1.00552968, 1.00916878, 0.95836143, 0.89409189]
    gkov_mis_GT = [0.79072509, 0.6591623, 0.59118787, 0.58059053, 0.51645357, 0.52850148,
    0.49307509, 0.48387059, 0.46598488, 0.46000819]
    
    ax.axhline(3.1807564712296204, color = 'red') # MI value
    p11, = ax.plot(N_prof_range_MLP, TI_mean_MLP, color = 'green')
    p12, = ax.plot(N_prof_range_MLP, PI_mean_MLP, color = 'green', linestyle = 'dashed')
    p21, = ax.plot(N_prof_range_logistic, TI_mean_logistic, color = 'blue')
    p22, = ax.plot(N_prof_range_logistic, PI_mean_logistic, color = 'blue', linestyle = 'dashed')
    p31, = ax.plot(N_prof_range_GT, TI_mean_GT, color = 'orange', label="GT")
    p32, = ax.plot(N_prof_range_GT, PI_mean_GT, color = 'orange', linestyle = 'dashed')
    p4, = ax.plot(N_prof_range_gkov, gkov_mis_MLP, color = 'green', linestyle = 'dotted')
    p5, = ax.plot(N_prof_range_gkov, gkov_mis_logistic, color = 'blue', linestyle = 'dotted')
    p6, = ax.plot(N_prof_range_gkov,  gkov_mis_GT , color = 'orange', linestyle = 'dotted')
    ax.set_xscale('log', base = 10)
    ax.grid(linestyle='--', linewidth= 0.5)
    ax.set_xlabel("Number of profiling traces")
    ax.set_ylabel("$MI$ [bits]")
    
    ax.set( ylim = (( 0.4, 3.3)) )
    
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    l = ax.legend( [ (p21, p22), (p11, p12), (p31, p32), p5, p4, p6], [ r'$(TI, PI)_{Log-lin}$', r'$(TI, PI)_{MLP}$', r'$(TI, PI)_{GT}$',  "$I_n(Y; Y|T_{M_{Log-lin}})$",
                     "$I_n(Y; Y|T_{M_{MLP}})$", "$I_n(Y; Y|T_{M_{GT}})$"], handlelength=3, numpoints= 1 , handler_map={tuple: HandlerTuple(ndivide=None)} , 
                      prop = { "size": 10 }, loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    axins = zoomed_inset_axes(ax, 3, loc= 'center right')
    
    
    
    x1, x2, y1, y2 = 30000, 50000, 2.2, 2.55
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
        
    
    # axins = ax.inset_axes([0.3, 0.3, 0.3, 0.3], xlim=(x1, x2), ylim=(y1, y2)) 
                          # xticklabels=[ 0.5, 3, 6, 9, 12], yticklabels= [0.003, 0.010, 0.016, 0.023])


    axins.plot(N_prof_range_MLP, TI_mean_MLP, color = 'green', label="MLP")
    axins.plot(N_prof_range_MLP, PI_mean_MLP, color = 'green', linestyle = 'dashed')
    axins.plot(N_prof_range_logistic, TI_mean_logistic, color = 'blue', label="LR")
    axins.plot(N_prof_range_logistic, PI_mean_logistic, color = 'blue', linestyle = 'dashed')
    axins.plot(N_prof_range_GT, TI_mean_GT, color = 'orange', label="GT")
    axins.plot(N_prof_range_GT, PI_mean_GT, color = 'orange', linestyle = 'dashed')
    # axins.set_xscale('log', base = 10)










    axins.set_yticks(np.linspace(2.2,2.56,3))
    axins.set_xticks(np.linspace(30000,50000,3))
    
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.4")
    
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    
    
    
    
    
    
    fname = 'MI_TI_PI_AES_HD_.png'
    # plt.legend()
    plt.savefig(fname, dpi = 1200)
    plt.show()
    
    
    
# if __name__ == "__main__":
#     main1()