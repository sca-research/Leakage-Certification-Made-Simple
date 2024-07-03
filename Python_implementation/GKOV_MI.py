import numpy as np
import scipy.special
from scipy.spatial import cKDTree
import math





def ends_gap_chunk1(kdtree, poss, t_n,  chunk_size):
    ''' Memory efficient nearest neighbor search for multivariate data'''
    
    chunk_count = poss.shape[0] // chunk_size
    out = []
    for chunk in np.array_split(poss, chunk_count):
        out.append(np.array(kdtree.query(chunk, k = t_n + 1,
            p=float('inf'), workers=-1)[0]))
        pass
    return np.concatenate(out)



def ends_gap_chunk2(kdtree, poss, dia_max,  chunk_size):
    ''' Memory efficient radius query search for multivariate data'''
    
    chunk_count = poss.shape[0] // chunk_size
    out = []
    for chunk in np.array_split(poss, chunk_count):
        out.append( kdtree.query_ball_point(chunk, dia_max, p =float('inf'),
            workers= - 1, return_length= True))
        pass
    return np.concatenate(out, dtype=np.int32)

def ends_gap_chunk3(kdtree, poss, dia_max,  chunk_size):
    ''' Memory efficient radius query search for univariate data'''
    
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
    GKOV MI estimator for discrete, discrete-continuous mixture or continuous data (univariate leakage traces)
    Parameters
    ----------
    q : Number of Samples (or Traces )
    t_n : Nearest neighbor value (e.g. t_n = int(np.log(n)))  
    pred_leakage : A list of Univariate Predicted Leakage variable (e.g. intermediate outcome)
    Tr : A list of Univariate Observable Traces

    Returns
    -------
    MI estimate (float value)
    '''
    
    # Combine Tr and pred_leakage into one array
    Tr_ = np.column_stack((Tr, pred_leakage))
    
    # Build KD-trees
    tree1 = cKDTree(Tr_)
    tree2 = cKDTree(np.array(Tr).reshape((q, 1)))
    tree3 = cKDTree(np.array(pred_leakage).reshape((q, 1)))
    
    # Find distances to the t_n-th nearest neighbor
    dist = tree1.query(Tr_, k=t_n + 1, p=float('inf'), workers=-1)[0][:, t_n]
    
    # Initialize arrays to store results for all points
    kp_arr = np.full(q, t_n, dtype=int)
    m_arr = np.zeros(q, dtype=int)
    n_arr = np.zeros(q, dtype=int)
    
    # Handle zero distances
    zero_distances = (dist == 0)
    if np.any(zero_distances):
        zero_indices = np.where(zero_distances)[0]
        zero_points = Tr_[zero_indices]
        kp_arr[zero_distances] = tree1.query_ball_point(zero_points, 1e-15, p=float('inf'), workers=-1, return_length=True)
        m_arr[zero_distances] = tree2.query_ball_point(zero_points[:, 0].reshape(-1, 1), 1e-15, p=float('inf'), workers=-1, return_length=True)
        n_arr[zero_distances] = tree3.query_ball_point(zero_points[:, 1].reshape(-1, 1), 1e-15, p=float('inf'), workers=-1, return_length=True)
    
    # Handle non-zero distances
    non_zero_indices = np.where(~zero_distances)[0]
    if np.any(non_zero_indices):
        non_zero_points = Tr_[non_zero_indices]
        m_arr[non_zero_indices] = tree2.query_ball_point(non_zero_points[:, 0].reshape(-1, 1), dist[non_zero_indices], p=float('inf'), workers=-1, return_length=True)
        n_arr[non_zero_indices] = tree3.query_ball_point(non_zero_points[:, 1].reshape(-1, 1), dist[non_zero_indices], p=float('inf'), workers=-1, return_length=True)
    
    # Calculate the mutual information estimate
    digamma_kp = scipy.special.digamma(kp_arr)
    log_q = np.log(q)
    log_m = np.log(m_arr + 1)
    log_n = np.log(n_arr + 1)
    
    ans = np.sum(digamma_kp + log_q - log_m - log_n) / q
    
    # Adjust result for logarithm base conversion
    result = ans * np.log2(math.e)
    
    return result




def MI_mixt_KSG(q, t_n, pred_leakage, Tr):
    '''
    KSG MI estimator for discrete, continuous data (univariate Tr)
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
    
    # Combine Tr and pred_leakage into one array
    Tr_ = np.column_stack((Tr, pred_leakage))
    
    # Build KD-trees
    Tr = np.array(Tr).reshape((q, 1))
    pred_leakage = np.array(pred_leakage).reshape((q, 1))
    
    tree1 = cKDTree(Tr_)
    tree2 = cKDTree(Tr)
    tree3 = cKDTree(pred_leakage)

    
    # Find distances to the t_n-th nearest neighbor
    dist = tree1.query(Tr_, k=t_n + 1, p=float('inf'), workers=-1)[0][:, t_n]
    

    
    m_arr = tree2.query_ball_point(Tr, dist, p=float('inf'), workers=-1, return_length=True)
    n_arr = tree3.query_ball_point(pred_leakage, dist, p=float('inf'), workers=-1, return_length=True)
    
    # Calculate the mutual information estimate
    digamma_kp = scipy.special.digamma(t_n)
    log_q = np.log(q)
    log_m = np.log(m_arr + 1)
    log_n = np.log(n_arr + 1)
    
    ans = np.sum(digamma_kp + log_q - log_m - log_n) / q
    
    # Adjust result for logarithm base conversion
    result = ans * np.log2(math.e)
    
    return result





def MI_gao_multi_(q, t_n, pred_leakage, Tr):
    '''
    GKOV MI estimator for discrete, discrete-continuous mixture or continuous data (multivariate leakage traces)
    Parameters
    ----------
    q : Number of Samples (or Traces)
    t_n : Nearest neighbor value (e.g. t_n = int(np.log(n)))
    pred_leakage : A list of Univariate Predicted Leakage variable (e.g. intermediate outcome)
    Tr : Multidimensional Trace data (dimension >=2)

    Returns
    -------
    MI estimate (float value)
    
    url = https://papers.nips.cc/paper_files/paper/2017/hash/ef72d53990bc4805684c9b61fa64a102-Abstract.html
    '''
    
    # Combine Tr and pred_leakage into one array
    Tr_ = np.column_stack((Tr, pred_leakage))
    
    # Build KD-trees
    tree1 = cKDTree(Tr_)
    Tr = np.array(Tr)
    pred_leakage = np.array(pred_leakage).reshape((q, 1))
    tree2 = cKDTree(Tr)
    tree3 = cKDTree(pred_leakage)
    
    # Find distances to the t_n-th nearest neighbor
    dist = tree1.query(Tr_, k=t_n + 1, p=float('inf'), workers=-1)[0][:, t_n]
    # Find distances to the t_n-th nearest neighbor (with better memory complexity)
    # dist = ends_gap_chunk1(tree1, Tr_, t_n, chunk_size)
    
    
    # Initialize arrays to store results for all points
    kp_arr = np.full(q, t_n)
    m_arr = np.zeros(q, dtype=int)
    n_arr = np.zeros(q, dtype=int)
    
    # Handle zero distances
    zero_distances = (dist == 0)
    non_zero_distances = ~zero_distances
    
    # Query ball points for zero distances
    if np.any(zero_distances):
        zero_indices = np.where(zero_distances)[0]
        zero_tr_points = Tr_[zero_indices]
        zero_tr_reshaped = Tr[zero_indices]
        zero_pred_leakage_reshaped = pred_leakage[zero_indices]
        
        kp_arr[zero_distances] = tree1.query_ball_point(zero_tr_points, 1e-15, p=float('inf'), workers=-1, return_length=True)
        m_arr[zero_distances] = tree2.query_ball_point(zero_tr_reshaped, 1e-15, p=float('inf'), workers=-1, return_length=True)
        n_arr[zero_distances] = tree3.query_ball_point(zero_pred_leakage_reshaped, 1e-15, p=float('inf'), workers=-1, return_length=True)
    
    # Query ball points for non-zero distances
    if np.any(non_zero_distances):
        non_zero_indices = np.where(non_zero_distances)[0]
        non_zero_tr_points = Tr[non_zero_indices]
        non_zero_pred_leakage_reshaped = pred_leakage[non_zero_indices]
        
        kp_arr[non_zero_distances] = t_n
        m_arr[non_zero_distances] = tree2.query_ball_point(non_zero_tr_points, dist[non_zero_distances], p=float('inf'), workers=-1, return_length=True)
        n_arr[non_zero_distances] = tree3.query_ball_point(non_zero_pred_leakage_reshaped, dist[non_zero_distances], p=float('inf'), workers=-1, return_length=True)
    
    # Calculate the mutual information estimate
    digamma_kp = scipy.special.digamma(kp_arr)
    log_q = np.log(q)
    log_m = np.log(m_arr + 1)
    log_n = np.log(n_arr + 1)
    
    ans = np.sum(digamma_kp + log_q - log_m - log_n) / q
    
    # Adjust result for logarithm base conversion
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
    if len(np.shape(Tr)) == 1:
        Tr = np.array(Tr).reshape((q, 1))
    
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


