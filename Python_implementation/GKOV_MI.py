import numpy as np
import scipy.special
from scipy.spatial import cKDTree
import math





def ends_gap_chunk1(poss, t_n,  chunk_size):
    ''' Memory efficient nearest neighbor search for multivariate data'''
    
    kdtree = cKDTree(poss)
    chunk_count = poss.shape[0] // chunk_size
    out = []
    for chunk in np.array_split(poss, chunk_count):
        out.append(np.array(kdtree.query(chunk, k = t_n + 1,
            p=float('inf'), workers=-1)[0]))
        pass
    return np.concatenate(out)

# def ends_gap_chunk2(poss, dia_max,  chunk_size):
#     ''' Memory efficient radius query search for multivariate data'''
#     kdtree = cKDTree(poss)
#     chunk_count = poss.shape[0] // chunk_size
#     out = []
#     for chunk in np.array_split(poss, chunk_count):
#         out.append( kdtree.query_ball_point(chunk, dia_max, p =float('inf'),
#             workers= - 1, return_length= True))
#         pass
#     return np.concatenate(out, dtype=np.int32)

def ends_gap_chunk3(poss, dia_max,  chunk_size):
    ''' Memory efficient radius query search for univariate data'''
    kdtree = cKDTree(poss)
    chunk_count = poss.shape[0] // chunk_size
    out = []
    for (poss_chunk, dia_max_chunk) in zip(np.array_split(
        poss, chunk_count), np.array_split(dia_max, chunk_count)):

        out.append(kdtree.query_ball_point(poss_chunk, dia_max_chunk,
            p = float('inf'), workers=-1, return_length=True))
        pass
    return np.concatenate(out, dtype=np.int32)



def MI_mixt_gao(q, t_n, pred_leakage, Tr):
    '''
    GKOV MI estimator for discrete, discrete-continuous mixture or continuous data
    (univariate leakage traces)
    Parameters
    ----------
    q : Number of Samples (or Traces )
    t_n : Nearest neighbor value (e.g. t_n = int(np.log(n)))  
    pred_leakage : A list of Univariate Predicted Leakage variable (e.g. intermediate outcome)
    Tr : A list of Univariate Observable Traces

    Returns
    -------
    MI estimate (float value)
    
    url = https://arxiv.org/abs/1709.06212
    '''
    Tr_ = np.column_stack((Tr, pred_leakage))
    tree1 = cKDTree(Tr_)
    dist_ =  tree1.query(Tr_, k=t_n + 1, p = float('inf'), workers=-1)[0]
    dist = dist_[:, t_n]
    del dist_
    
    cond_dist = np.where(dist == 0, dist, dist-2e-15)
    Tr = np.array(Tr).reshape((q,1))
    tree2 = cKDTree(Tr)
    result_m2 = tree2.query_ball_point(Tr, cond_dist + 1e-15,
            p=float('inf'), workers=-1, return_length=True)
    del Tr, tree2
    
    pred_leakage = np.array(pred_leakage).reshape((q, 1))
    tree3 = cKDTree(pred_leakage)
    result_n2 = tree3.query_ball_point(pred_leakage, cond_dist+1e-15,
            p = float('inf'), workers= -1, return_length = True)
    del pred_leakage, tree3
    
    ans = 0
    for i in np.arange(0, q):
        kp, m, n = t_n, result_m2[i], result_n2[i]
        if dist[i] == 0:
            kp = tree1.query_ball_point(Tr_[i], 1e-15, p = float('inf'),
                                        workers = -1, return_length = True) 
            pass
        ans += (scipy.special.digamma(kp) - np.log(m+1) - np.log(n+1))/q
        pass
    
    ans = ans + np.log(q)
    del dist, tree1, result_m2, result_n2
    
    result = ans * np.log2(math.e)
    return result



def MI_mixt_KSG(q, t_n, pred_leakage, Tr):
    '''
    KSG MI estimator for discrete, continuous data (univariate Tr)
    Parameters
    ----------
    q : Number of Samples (or Traces )
    t_n : Nearest neighbor value (e.g. t_n = int(np.log(n)))  
    pred_leakage : A list of Univariate Predicted Leakage variable (e.g. intermediate outcome)
    Tr : A list of Univariate Observable Traces

    Returns
    -------
    MI estimate (float value)
    
    
    url = https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138#fulltext
    '''
    Tr_ = np.column_stack((Tr, pred_leakage))
    tree1 = cKDTree(Tr_)
    dist_ =  tree1.query(Tr_, k=t_n + 1, p = float('inf'), workers=-1)[0]
    dist = dist_[:, t_n]
    del dist_
    
    cond_dist = np.where(dist == 0, dist, dist-2e-15)
    Tr = np.array(Tr).reshape((q,1))
    tree2 = cKDTree(Tr)
    result_m2 = tree2.query_ball_point(Tr, cond_dist + 1e-15,
            p=float('inf'), workers=-1, return_length=True)
    del Tr, tree2
    
    pred_leakage = np.array(pred_leakage).reshape((q, 1))
    tree3 = cKDTree(pred_leakage)
    result_n2 = tree3.query_ball_point(pred_leakage, cond_dist+1e-15,
            p = float('inf'), workers= -1, return_length = True)
    del pred_leakage, tree3
    
    m = np.sum(np.log(result_m2 + 1))
    n = np.sum(np.log(result_n2 + 1))
    del dist, result_m2, result_n2
    ans = scipy.special.digamma(t_n) + np.log(q) - ((m + n)/q)
    
    result = ans * np.log2(math.e)
    return result



def MI_mixt_gao_opt(q, t_n, pred_leakage, Tr, chunk_size): 
    ''' 
    Optimal GKOV MI evaluation with lesser memory complexity  (recommended for q > 10^7)
    
    Parameters
    ----------
    q : Number of Samples (or Traces )
    t_n : Nearest neighbor value (e.g. t_n = int(np.log(n)))  
    pred_leakage : A list of Univariate Predicted Leakage variable (e.g. intermediate outcome)
    Tr : A list of Univariate Observable Traces
    chunk_size : Size of a Chunk of large traces (minimum value should be q)
    (Ref link: https://github.com/scipy/scipy/issues/15621#issuecomment-1057141414)
     
    Returns
    -------
    MI estimate (float value)

    '''
    
    # radius tree search for combined data of Tr and pred_leakage
    # Remember here both Tr and pred_leakage are one dimensional arrays
    
    Tr_ = np.column_stack((Tr, pred_leakage))
    dist_ = ends_gap_chunk1(Tr_, t_n, chunk_size)
    dist = dist_[:,t_n]
    del dist_ 
    
    # Distance vector adjustment-------------------------------
    cond_dist = np.where( dist == 0, dist, dist - 2e-15)
    
    Tr = np.array(Tr).reshape((q,1)) # If we consider Tr is univariate
    result_m2 = ends_gap_chunk3(Tr , cond_dist + 1e-15 , chunk_size) 
    del Tr
    
    pred_leakage = np.array(pred_leakage).reshape((q,1))
    result_n2 = ends_gap_chunk3(pred_leakage, cond_dist + 1e-15, chunk_size)
    del pred_leakage
    
    treexy = cKDTree(Tr_)
    ans = 0
    for i in np.arange(0, q):
        kp, m, n = t_n, result_m2[i], result_n2[i]
        if dist[i] == 0:
            kp = treexy.query_ball_point(Tr_[i], 1e-15, p = float('inf'),
                                        workers = -1, return_length = True)
            pass
        ans += (scipy.special.digamma(kp) - np.log(m + 1) - np.log(n + 1)) / q
        pass
    ans = ans + np.log(q)
    del dist, kp, result_m2, result_n2
    
    result = ans * np.log2(math.e)
    return result





def MI_gao_multi(q, t_n, pred_leakage, Tr):
    '''Calculate GKOV MI between multivariate Tr and univariate pred_leakage'''

    Tr_ = np.column_stack((Tr, pred_leakage))
    tree1 = cKDTree(Tr_)
    
    dist_ =  tree1.query(Tr_, k = t_n + 1, p = float('inf'), workers = -1)[0]
    dist = dist_[:, t_n]
    del dist_

    # Distance vector adjustment-------------------------------
    cond_dist = np.where(dist == 0, dist, dist - 2e-15)
    tree2 = cKDTree(Tr)
    result_m2 = tree2.query_ball_point(Tr, cond_dist + 1e-15,
            p = float('inf'), workers = -1, return_length = True)
    del Tr, tree2

    # Reshaping pred_leakage for tree search------------------
    if len(np.shape(pred_leakage)) == 1:
        pred_leakage = np.array(pred_leakage).reshape((q, 1))
    tree3 = cKDTree(pred_leakage)
    result_n2 = tree3.query_ball_point(pred_leakage, cond_dist + 1e-15,
            p = float('inf'), workers= -1, return_length = True)
    del pred_leakage, tree3

    ans = 0
    for i in np.arange(0, q):
        kp, m, n = t_n, result_m2[i], result_n2[i]
        if dist[i] == 0:
            kp = tree1.query_ball_point(Tr_[i], 1e-15, p = float('inf'), 
                                        workers = -1, return_length = True)
            pass
        ans +=  (scipy.special.digamma(kp) - np.log(m+1) - np.log(n+1))/q
        # ans +=  (scipy.special.digamma(kp) - (scipy.special.digamma(m) + scipy.special.digamma(n)))/q
        pass

    ans = ans + np.log(q)
    del dist, tree1, result_m2, result_n2
    result = ans * np.log2(math.e) # adjustment for log base = 2 results

    return result 





def MI_gao_multi_opt(q, t_n, pred_leakage, Tr, chunk_size = 50000):
    '''
    GKOV MI estimator (Multivariate version) with lesser memory complexity  (recommended for q > 10^6)

    Parameters
    ----------
    q : Number of Samples (or Traces )
    t_n : Nearest neighbor value (e.g. t_n = int(np.log(n)))  
    pred_leakage : A list of uni(/multi)variate predicted leakage (e.g. intermediate outcome)
    Tr : Multidimensional Trace data (dimension >=2)
    chunk_size : Size of a chunk of large traces (it must satisfy <= q), we have considered a default value of 50000
    (Ref link: https://github.com/scipy/scipy/issues/15621#issuecomment-1057141414)
    Returns                                                                                                                                                                                                                                                                               
    -------
    MI estimate (float value)
    '''

    Tr_ = np.column_stack((Tr, pred_leakage))
    dist_ = ends_gap_chunk1(Tr_, t_n, chunk_size)
    dist = dist_.T[t_n]
    del dist_
    cond_dist = np.where(dist == 0, dist, dist - 2e-15)
    
    # ---------------------------------------------------------------------------------
    result_m2 = ends_gap_chunk3(Tr, cond_dist + 1e-15, chunk_size)
    del Tr
    # ---------------------------------------------------------------------------------
    pred_leakage = np.array(pred_leakage).reshape((q, 1))       # comment this when pred_leakage is multivariate
    
    result_n2 = ends_gap_chunk3(pred_leakage, cond_dist + 1e-15, chunk_size)
    del pred_leakage
    
    treexy = cKDTree(Tr_)
    ans = 0
    for i in np.arange(0, q):
        kp, m, n = t_n, result_m2[i], result_n2[i]
        if dist[i] == 0:
            kp = treexy.query_ball_point(Tr_[i], 1e-15, p = float('inf'),
                                        workers = -1, return_length = True)
        ans +=  (scipy.special.digamma(kp) - np.log(m+1) - np.log(n+1))/q
    ans = ans + np.log(q)
    del dist , result_m2, result_n2
    
    result = ans * np.log2(math.e)
    return result
