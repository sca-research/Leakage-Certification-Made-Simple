#!/usr/bin/python3
import pandas as pd
import numpy as np
import math
import os
import sys
import random

import scipy.special  # Digamma function
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors, BallTree
# from numba_kdtree import KDTree


# for optimizing my nearest neighbor MI code
from heapq import nsmallest
from bisect import bisect, bisect_right

# For fast conditional count computation
import numexpr as ne
from numba import njit, prange

from scipy.stats import dlaplace  # simulation from discrete lapalce


# Sbox look-up tables
AES_Sbox = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]

DES_Sbox = [
    14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
    0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
    4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
    15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]

# Different Leakage Models on intermediate-------------------------------------


def lkg_7LSB(a):
    return a & 0x7f


def lkg_3LSB(a):
    return a & 0x7


def lkg_4LSB(a):
    return a & 0xf


def lkg_6LSB(a):
    return a & 0x3f


def lkg_5LSB(a):
    return a & 0x1f


def lkg_4MSB(a):
    return (a & 0xf0) >> 4


def ham_wt(a):
    return bin(a).count("1")


def Non_lin(a):
    return DES_Sbox[lkg_6LSB(a)]


def ham_dist(x, y):
    return bin(x ^ y).count("1")


def quad(x):
    return x*(x+1)


def extractBits(num, k):
    binary = "{:08b}".format(num)
    if k == 3:
        return [int(binary[:2], 2),  int(binary[2:4], 2), int(binary[4:], 2)]
    if k == 4:
        return [int(binary[:2], 2), int(binary[2:4], 2), int(binary[4:6], 2), int(binary[6:], 2)]


# Univariate Leakage (Unmasked implementation)-------------------------------------
def trace(sbox, key, pt, sigma):
    '''
    Parameters
    ----------
    sbox : Subbyte operation based on lookup table
    key : Secret Key
    pt : a list of Plain texts (Usually generate from Unif(0,256)
    sigma : Specify the standard deviation of Noise variable

    Returns
    -------
    trace : a list of univariate Traces

    '''
    trace = []
    q = len(pt)
    for text in np.arange(0, q):
        ''' Linear Model--------------------------------------------------------'''
        inter_med = sbox[pt[text] ^ key]
        wt = 0x71     # Define linear weights
        x = float(ham_wt(inter_med & wt))  # Weighted Hamming Weight
        # x = float(ham_dist(inter_med, pt[text] ^ key)) # hamming distance
        # x = float(ham_wt(inter_med))   # Hamming weight
        # x= float(lkg_5LSB(inter_med))    # 5-Least  significant bits of Intermediate

        ''' Nonlinear Model---------------------------------------------------'''
        # x = float(Non_lin(inter_med)) # Double cryptographic permutation

        ''' Adding Continuous noise with leakage model ouptput-----------------'''
        # trace.append(x + float(np.random.normal(0, sigma, 1)))  # Gaussian noise
        # trace.append(x + float(np.random.laplace(0,sigma,1)))  ## laplace noise
        # trace.append(x + float(np.random.uniform(-sigma*math.sqrt(8), sigma*math.sqrt(8), 1))) ## Uniform Noise
        trace.append(x + float(np.random.lognormal(0, sigma, 1)))

        ''' Discrete Leakage Simulation---------------------------------------'''
        # x = int(ham_wt(inter_med))
        # x = int(ham_dist(inter_med, pt[text] ^ key))
        # x = int(Non_lin(inter_med))
        # x = int(ham_wt(inter_med & wt))
        ''' Adding Continuous noise with leakage model ouptput----------------'''
        # trace.append(x + int(dlaplace.rvs(sigma, 1)))
        # trace.append( x + int(np.random.poisson(sigma)))

    return trace


# ================================================================================
# NearestNeighbors based MI estimators--------------------------------------------

# Optimitzed Code for BC ross MI estimator----------------------------------------
# K-nearest points----------------------------------------------------------------
def k_nearest(k, center, sorted_data):
    'Return *k* members of *sorted_data* nearest to *center*'
    i = bisect(sorted_data, center)
    segment = sorted_data[max(i-k, 0): i+k]
    return nsmallest(k, segment, key=lambda x: abs(x - center))

# Find numbers within a range by bisect


def NumbersWithinRange(items, lower, upper):
    start = bisect(items, lower - 0.000005)
    end = bisect_right(items, upper + 0.000005)
    return items[start:end]


def NumbersWithinRange2(trace_data, i, delta):
    count = 0
    j = i-1
    centre = trace_data[i]

    while not j < 0 and trace_data[j] >= centre - delta:
        count += 1
        j -= 1

    j = i + 1
    while not j > len(trace_data) - 1 and trace_data[j] <= centre + delta:
        count += 1
        j += 1
    return count


def Muti_MI_opt(q, t_n, pred, trace_):
    '''
    BC ross MI estimate of a set of continuous-discrete paired data
    Parameters
    ----------
    q : Number of Samples (or Traces )
    t_n : Nearest neighbor intiger value (e.g. 2,3,4)
    pred_leakage : an array of Univariate 'discrete' Predicted Leakage variable (e.g. intermediate outcome)
    Tr : Unidimensional Trace data 

    Returns
    -------
    BC ross MI estimate

    '''
    pred_leakage = [pred[x] for x in np.argsort(trace_)]
    trace_data = np.sort(trace_)
    m = 0  # Calculating M_i and the vector Nx_i
    n = 0
    for i in range(q):
        cond_tr = [trace_data[j]
                   for j in range(q) if pred_leakage[j] == pred_leakage[i]]
        if(len(cond_tr) >= t_n):
            final = k_nearest(t_n, trace_data[i], cond_tr)
            dist = [abs(final[0]-f) for f in final]
            delta = max(dist)
            filtered_data = NumbersWithinRange(
                trace_data, trace_data[i] - delta, trace_data[i] + delta)
            m += scipy.special.digamma(len(filtered_data) - 1)
            n += scipy.special.digamma(len(cond_tr))  # making the count
            # m += scipy.special.digamma(NumbersWithinRange2(trace_data, i, delta))
            # n += scipy.special.digamma(len(cond_tr))  # making the count

    return (scipy.special.digamma(t_n) + scipy.special.digamma(q)) - ((m+n)/q)


# ================================================================================
# GAO MI estimator for mixture (discrt, cont) pairs, with digamma-----------------
@njit(parallel=True, fastmath=True)
def method1(X, cx, r):
    acc = 0
    lb = cx - r
    ub = cx + r
    for i in prange(X.shape[0]):
        if (X[i] <= ub) and (X[i] >= lb):
            acc += 1
    return acc


# q=Sample size  # t_n= t-th neighbourhood-------------------------------------
def MI_mixt_gao_naive(q, t_n, pred_leakage, Tr):
    '''
    GKOV MI estimator for discrete, discrete-continuous mixture or continuous data(univariate leakage traces)
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

    z = np.column_stack((Tr, pred_leakage))

    tree = cKDTree(z)
    dist = [tree.query(z[i], k=t_n + 1, p=float('inf'),
                       workers=-1)[0][t_n] for i in np.arange(0, q)]

    ans = 0
    Tr = np.array(Tr)
    pred_leakage = np.array(pred_leakage)
    for i in np.arange(0, q):
        kp, m, n = t_n, t_n, t_n
        if dist[i] == 0:
            kp = len(tree.query_ball_point(
                z[i], 1e-15, p=float('inf'), workers=-1))
            m = method1(Tr, Tr[i], dist[i])
            n = method1(pred_leakage, pred_leakage[i], dist[i])
        else:
            m = method1(Tr, Tr[i], dist[i])
            n = method1(pred_leakage, pred_leakage[i], dist[i])

        ans += (scipy.special.digamma(kp) +
                np.log(q) - np.log(m) - np.log(n))/q
    result = ans*np.log2(math.e)
    return result
# ------------------------------------------------------------------------------------
# Vectorized fast algorithm with using cKDTree----------------------------------------
def MI_mixt_gao(q, t_n, pred_leakage, Tr):
    '''
    GKOV MI estimator for discrete, discrete-continuous mixture or continuous data(univariate leakage traces)
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
    Tr_ = np.column_stack((Tr, pred_leakage))
    tree1 = cKDTree(Tr_)
    dist_ =  tree1.query(Tr_, k=t_n + 1, p=float('inf'), workers=-1)[0]
    dist = dist_[:, t_n]
    del dist_
    cond_dist = np.where(dist == 0, dist, dist-2e-15)
    Tr = np.array(Tr).reshape((q,1))
    tree2 = cKDTree(Tr)
    result_m2 = tree2.query_ball_point(Tr, cond_dist+1e-15,
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
            kp = tree1.query_ball_point(Tr_[i], 1e-15, p =float('inf'),
                                        workers = -1, return_length = True)             
        ans += (scipy.special.digamma(kp) - np.log(m+1) - np.log(n+1))/q
    ans = ans + np.log(q)
    del dist, tree1, result_m2, result_n2
    result = ans * np.log2(math.e)                                                                                                                                                                                                                            
    return result

# ------------------------------------------------------------------------------------
def MI_gao_multi_naive(q, t_n, pred_leakage, Tr):
    '''
    GKOV MI estimator(Multivariate version)

    Parameters
    ----------
    q : Number of Samples (or Traces )
    t_n : Nearest neighbor value (e.g. t_n = int(np.log(n)))
    pred_leakage : A list of uni(/multi)variate predicted leakage (e.g. intermediate outcome)
    Tr : Multidimensional Trace data (dimension >=2)

    Returns
    -------
    MI estimate (float value)
    '''

    Tr_ = np.column_stack((Tr, pred_leakage))
    tree = cKDTree(Tr_)
    dist = [tree.query( Tr_[i], k = t_n + 1, p = float('inf'),
                          workers = -1 )[0][t_n]   for i in np.arange(0,q)]
    
    tree2 = cKDTree(Tr)
    pred_leakage = np.array(pred_leakage).reshape((q,1)) #comment this when p is multivariate
    tree3 = cKDTree(pred_leakage)


    ans = 0
    # pred_leakage = np.array(pred_leakage)
    for i in np.arange(0, q):
        kp, m, n = t_n, t_n, t_n
        if dist[i] == 0:
            kp =   len(tree.query_ball_point( Tr_[i], 1e-15,
                                                    p =float('inf'), workers = -1))
            m = len(tree2.query_ball_point(Tr[i], 1e-15,
                                        p = float('inf'), workers = -1))
            # n = method1(pred_leakage, pred_leakage[i],  1e-15)
            n = len(tree3.query_ball_point(pred_leakage[i], 1e-15,
                                              p = float('inf'), workers = -1))
        else:
            m = len(tree2.query_ball_point(Tr[i], dist[i]-1e-15,
                                          p = float('inf'), workers = -1))
            # n = method1(pred_leakage, pred_leakage[i], dist[i])
            n = len(tree3.query_ball_point(pred_leakage[i], dist[i]-1e-15,
                                              p = float('inf'), workers = -1))

        ans +=  (scipy.special.digamma(kp) + np.log(q)- np.log(m+1) - np.log(n+1))/q
    del Tr, Tr_ , dist, tree, tree2
    del tree3
    result =  ans*np.log2(math.e)
    return result

# Vectorized fast algorithm with solving the memory overflow:  https://github.com/scipy/scipy/issues/15621--
# We avoid cKDtree to solve the memory overflow issue when data size is more than equal to 10^7



def ends_gap_chunk1(poss, t_n,  chunk_size):
    kdtree = cKDTree(poss)  # ckdtree function
    # kdtree = KDTree(poss, leafsize= 16) #using numba
    chunk_count = poss.shape[0] // chunk_size
    out = []
    for chunk in np.array_split(poss, chunk_count):
        out.append(np.array(kdtree.query(chunk, k=t_n + 1,
                        p=float('inf'), workers=-1)[0], dtype=np.float32))

    return np.concatenate(out, dtype=np.float32)



def ends_gap_chunk3(poss, dia_max,  chunk_size):
    kdtree = cKDTree(poss)  # cKdtree version
    # We couldnot use numba_kdtree for this case-----------------------
    chunk_count = poss.shape[0] // chunk_size
    out = []
    for (poss_chunk, dia_max_chunk) in zip(np.array_split(
            poss, chunk_count), np.array_split(dia_max, chunk_count)):
        
        out.append(kdtree.query_ball_point(poss_chunk, dia_max_chunk,
                        p=float('inf'), workers=-1, return_length=True))
    
    return np.concatenate(out, dtype= np.int32)


def MI_gao_multi(q, t_n, pred_leakage, Tr, chunk_size = 50000):
    '''
    GKOV MI estimator(Multivariate version)

    Parameters
    ----------
    q : Number of Samples (or Traces )
    t_n : Nearest neighbor value (e.g. t_n = int(np.log(n)))  
    pred_leakage : A list of uni(/multi)variate predicted leakage (e.g. intermediate outcome)
    Tr : Multidimensional Trace data (dimension >=2)

    Returns                                                                                                                                                                                                                                                                               
    -------
    MI estimate (float value)
    '''
    
    
    pred_leakage = np.array(pred_leakage).reshape((q, 1))       # comment this when pred_leakage is multivariate
    
    
    # Tr_ = np.column_stack((Tr, pred_leakage))
    Tr_ = np.concatenate(( Tr, pred_leakage),axis=1)
    
    dist_ = ends_gap_chunk1(Tr_, t_n, chunk_size)
    dist = dist_.T[t_n]
    del dist_
    cond_dist = np.where(dist == 0, dist, dist-2e-15)
    
    # ---------------------------------------------------------------------------------
    result_m2 = ends_gap_chunk3(Tr, cond_dist+1e-15, chunk_size)
    del Tr
    # ---------------------------------------------------------------------------------
    
    result_n2 = ends_gap_chunk3(pred_leakage, cond_dist+1e-15, chunk_size)
    del pred_leakage
    
    tree_xy = cKDTree(Tr_)
    ans = 0
    for i in np.arange(0, q):
        kp, m, n = t_n, result_m2[i], result_n2[i]
        if dist[i] == 0:
            kp = len(tree_xy.query_ball_point(Tr_[i], 1e-15, workers = -1, p=float('inf')))
            
        # ans +=  (scipy.special.digamma(kp) - np.log(m+1) - np.log(n+1))/q
        ans += (scipy.special.digamma(kp) - (scipy.special.digamma(m) + scipy.special.digamma(n))) / q
        
        
    ans = ans + np.log(q)
    del dist , result_m2, result_n2
 

    # Some short-cut for only continuous Tr(KSG MI estimator)---------------------------
    # m = np.sum(np.log(result_m2 + 1))
    # n = np.sum(np.log(result_n2 + 1))
    # del dist, result_m2, result_n2
    # ans = scipy.special.digamma(t_n) + np.log(q) - ((m + n)/q)
    result = ans * np.log2(math.e)
    return result


# ===============================================================================
# Histogram estimators of MI(L ; Y)----------------------------------------------
# Y: pred_leakage(discrete) and L = observed traces(continuous)------------------
def mi_entrpy_hist(pred_leakage, ot, nbins1, nbins2):
    '''
    Parameters
    ----------
    pred_leakage : A list of univariate predicted leakage (e.g. intermediate outcome)
    ot : A list of univariate continuous Observable Traces
    nbins1 : number of bins for the univariate continuous observable Trace data
    nbins2 : number of bins for the conditional contniuous observable trace (condition has been taken on predicted leakage)

    Returns
    -------
    Histogram MI estimator for the (continous,discrete) paired data

    '''

    h1, bin_edges = np.histogram(ot, nbins1)

    prob = h1/np.sum(h1)

    s = bin_edges[1]-bin_edges[0]
    entrpy_trc = 0
    for i in range(len(prob)):
        if(prob[i] != 0):
            entrpy_trc -= np.log2(prob[i]/s)*prob[i]

    all_cond_probs = []
    steps = []

    uniq_lkg = list(set(pred_leakage))

    for i in uniq_lkg:

        cond_tr = [ot[j] for j, l in enumerate(pred_leakage) if l == i]
        cond_prob, bin_edges2 = np.histogram(cond_tr, nbins2)
        all_cond_probs.append(cond_prob/np.sum(cond_prob))
        steps.append(bin_edges2[1] - bin_edges2[0])

    N = len(all_cond_probs)
    cond_entrps = [0]*N

    for i in range(N):
        for j in range(len(all_cond_probs[i])):
            if(all_cond_probs[i][j] != 0):
                cond_entrps[i] -= np.log2(all_cond_probs[i]
                                          [j]/steps[i])*all_cond_probs[i][j]

    pred_lkg_hist = [pred_leakage.count(i) for i in uniq_lkg]
    temp_counts = np.array(pred_lkg_hist)
    pred_lkg_prob = temp_counts/np.sum(pred_lkg_hist)

    cond_entrp = 0

    for i in range(N):
        cond_entrp += cond_entrps[i]*pred_lkg_prob[i]

    return entrpy_trc-cond_entrp


# histogram estimator of MI({L1,L2,L3} ; Y)----------------------------------------
# Y: pred_leakage(discrete) and L_i = observed traces(continuous)------------------

def mi_entrpy_histdd(pred_leakage, ot, nbins):
    '''
    Parameters
    ----------
    pred_leakage : pred_leakage : A list of univariate predicted leakage (e.g. intermediate outcome)
    ot : Multidimensional Trace data (dimension >=2)
    nbins : a set of Number of bins for d-dimensional data 

    Returns
    -------
    d-dimensional Histogram MI estimator for the (continous,discrete) paired data.
    (Here Continuous data(traces) is of d-dimensional)
    '''

    n_shr = len(nbins)  # for bivariate = 2 for tri-variate = 3
    h1, edges = np.histogramdd(np.array(ot),
                               bins=(nbins[0], nbins[1], nbins[2]))  # 'bins' need to be changed for different dimension

    h_ = np.concatenate(np.concatenate(h1))
    prob = h_/np.sum(h_)

    # s = (edges[0][1]-edges[0][0])*(edges[1][1]-edges[1][0]) # for 2D data
    s = (edges[0][1]-edges[0][0])*(edges[1][1]
                                   - edges[1][0])*(edges[2][1]-edges[2][0])
    del h1
    del h_
    del edges

    entrpy_trc = 0
    for i in range(len(prob)):
        if(prob[i] != 0):
            entrpy_trc -= np.log2(prob[i]/s)*prob[i]
    del prob
    del s

    all_cond_probs = []
    steps = []

    uniq_lkg = list(set(pred_leakage))
    ot = np.array(ot).T.tolist()
    for i in uniq_lkg:
        cond_tr = [[ot[x][j] for j, l in
                    enumerate(pred_leakage) if l == i] for x in range(n_shr)]

        cond_prob, edges2 = np.histogramdd((cond_tr[0], cond_tr[1], cond_tr[2]),
                                           bins=(nbins[0], nbins[1], nbins[2]))
        del cond_tr
        cond_pro = np.concatenate(np.concatenate(cond_prob))
        all_cond_probs.append(cond_pro/np.sum(cond_pro))
        # steps.append((edges2[0][1]-edges2[0][0])*(edges2[1][1]-edges2[1][0])) # for 2D data
        steps.append((edges2[0][1]-edges2[0][0])
                     * (edges2[1][1]-edges2[1][0])*(edges2[2][1]-edges2[2][0]))
        del cond_prob
        del cond_pro
        del edges2
    del ot

    N = len(all_cond_probs)
    cond_entrps = [0]*N

    for i in range(N):
        for j in range(len(all_cond_probs[i])):
            if(all_cond_probs[i][j] != 0):
                cond_entrps[i] -= np.log2(all_cond_probs[i]
                                          [j]/steps[i])*all_cond_probs[i][j]
    del all_cond_probs
    del steps

    pred_lkg_hist = [pred_leakage.count(i) for i in uniq_lkg]
    temp_counts = np.array(pred_lkg_hist)
    pred_lkg_prob = temp_counts/np.sum(pred_lkg_hist)
    del pred_lkg_hist, temp_counts, pred_leakage

    cond_entrp = 0

    for i in range(N):
        cond_entrp += cond_entrps[i]*pred_lkg_prob[i]

    return entrpy_trc-cond_entrp

# ==============================================================================
# -----------------------------------------------------------------------------
# plug-in estimator of MI(L ; Y), when observed trace(ot) is discrete ---------


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

    unique_ot, count_uni = np.unique(ot, return_counts=True)
    ot_prob = count_uni/np.sum(count_uni)
    del unique_ot, count_uni
    entrop_trc = 0
    for i in range(len(ot_prob)):
        entrop_trc -= np.log2(ot_prob[i])*ot_prob[i]

    all_cond_probs = []
    unique_Y, count_Y = np.unique(pred_leakage, return_counts=True)
    Y_prob = count_Y/np.sum(count_Y)

    for i in unique_Y:
        cond_tr = [ot[j] for j, l in enumerate(pred_leakage) if l == i]
        uniq2 = set(cond_tr)
        cond_tr_count = np.array([cond_tr.count(j) for j in uniq2])
        all_cond_probs.append(cond_tr_count / np.sum(cond_tr_count))

    N = len(all_cond_probs)
    cond_entrps = [0]*N
    for i in range(N):
        for j in range(len(all_cond_probs[i])):
            if(all_cond_probs[i][j] != 0):
                cond_entrps[i] -= np.log2(all_cond_probs[i]
                                          [j])*all_cond_probs[i][j]

    cond_entrp = 0

    for i in range(N):
        cond_entrp += cond_entrps[i] * Y_prob[i]

    return entrop_trc - cond_entrp

# plug-in estimator of MI( {L_1,L_2} ; Y)---------------------------------------------
# Multidimensional plug-in estimator-----------------------------------------------------


def mi_plug_indd(pred_lkg, ot):
    '''
    Parameters
    ----------
    pred_lkg : An array(1 x n) of univariate Pred_leakage (an arbitrary functional output of the intermediate value)
    ot : A D-dimensional array of multivariate Trace data (D x n discrete arrays) 

    Returns: MI plugin estimator of the multivariate discrete leakage and a univariate function of intermediate (discrete in nature)
    -------
    Note: This function is not applicable for non-discrete data

    '''
    unique_ot, count_multi = np.unique(ot, axis=0, return_counts=True)
    ot_prob = count_multi/np.sum(count_multi)
    del count_multi, unique_ot
    entrop_trc = 0
    for i in range(len(ot_prob)):
        entrop_trc -= np.log2(ot_prob[i])*ot_prob[i]
    del ot_prob
    
    
    
    
    all_cond_probs = []
    unique_Y, count_Y = np.unique(pred_lkg, return_counts=True)
    Y_prob = count_Y/np.sum(count_Y)
    
    
    all_cond_probs = []
    uniq_lkg = list(set(pred_lkg))
    for i in uniq_lkg:
        cond_ot = [ot[j] for j, l in enumerate(pred_lkg) if l == i]
        unique_cond_ot, cond_ot_count = np.unique(
            cond_ot, axis=0,  return_counts=True)
        del cond_ot
        all_cond_probs.append(cond_ot_count / np.sum(cond_ot_count))
        del cond_ot_count, unique_cond_ot

    N = len(all_cond_probs)

    cond_entrps = [0]*N
    for i in range(N):
        for j in range(len(all_cond_probs[i])):
            if(all_cond_probs[i][j] != 0):
                cond_entrps[i] -= np.log2(all_cond_probs[i]
                                          [j]) * all_cond_probs[i][j]
    del all_cond_probs


    cond_entrp = 0
    for i in range(N):
        cond_entrp += cond_entrps[i] * Y_prob[i]
    del cond_entrps

    return entrop_trc - cond_entrp
# ====================================================================================

# -----------------------------------------------------------------------------------
# True MI value calculation by Numerical Analaysis method----------------------------


def main():

    f = math.factorial

# Distribution (pdf) of Noise--------------------------------------------------------------
    def Normal_(x, mu, sigma):
        return math.exp(- pow(x-mu, 2) / (2 * pow(sigma, 2))) / math.sqrt(pow(sigma, 2) * 2 * math.pi)

    def log_normal_(x, mu, sigma):
        return math.exp( - pow(math.log(x) - mu, 2) / (2 * pow(sigma, 2))) / (math.sqrt(2 * math.pi) * sigma * x)

    def Unif_(a, b):
        return 1 / (b-a)

    def Laplace_(x, mu, b):
        return (1 / (2 * b)) * math.exp(- (abs(x - mu) / b))

# Distribution (pdf) of leakage model output--------------------------------------------
    def Bin_(x, n):                                    # HW, WHW and HD
        return f(n)/(f(x)*f(n-x)*pow(2, n))


# Distribution (pdf) of Observed Traces--------------------------------------------------


    def GM_bin(z, sig, n):  # Gaussian Mixture with Binomial coeffcients as weights
        return sum(Bin_(x, n)*Normal_(z, x, sig) for x in range(n+1))

    def lNM_bin(z, sig, n):  # Lognormal Mixture with  Binomial coeffcients as weights
        return sum(Bin_(x, n)*log_normal_(z, x, sig) for x in range(n+1))

    def UM_bin(z, sig, n):  # Uniform Mixture with Binomial coeffcients as weights
        return sum(Bin_(x, n)/(2*sig*math.sqrt(3)) for x in range(n+1)
                   if z > x - sig*math.sqrt(3) and z < x + sig*math.sqrt(3))

    def LM_bin(z, sig, n):  # Laplace Mixture with Binomial coeffcients as weights
        return sum(Bin_(x, n) * Laplace_(z, x, sig) for x in range(n+1))

    def LM_unif(z, sig, n):  # Laplace Mixture with identity device leakage model
        return sum(Laplace_(z, x, sig)/pow(2, n) for x in range(pow(2, n)))

    def GM_unif(z, sig, n):  # Gaussian Mixture with identity device leakage model
        return sum((Normal_(z, x, sig)/pow(2, n)) for x in range(pow(2, n)))

    def GM_quad(z, sig, n):  # Gaussian mixture with Quadratic device leakage model
        unif_s = list(range(pow(2, n)))
        u_q_s = [quad(x) for x in unif_s]
        return sum([(Normal_(z, x, sig)/pow(2, n)) for x in u_q_s])


# Process of Numerical integration by trapezoidal rule-------------------------


    def trapezoidal(f, a, b, n):
        h = float(b - a) / n
        s = 0.0
        s += f(a)/2.0
        for i in range(1, n-1):
            s += f(a + i*h)
        s += f(b)/2.0
        return s * h

    sig = 8
    n = 8  # no. of bits, here we consider 1-byte AES permutation

    Min_value = 0 - 6*sig                 # For Gaussian Noise and HW device_lkg
    Max_value = 8 + 6*sig

    # Min_value= 0 - 6*sig               # For Gaussian Noise and n-LSB device_lkg
    # Max_value= pow(2,n)-1 + 6*sig

    # Min_value = 0 - 6*sig             # For Gaussian Noise and Quadratic device leakage model
    # Max_value = quad(pow(2,n)-1) + 6*sig

    # Min_value = 0 - sig*math.sqrt(3)  # For Uniform device Noise and HW device_lkg
    # Max_value = 8 + sig*math.sqrt(3)

    # Min_value = 0 - 11*sig           # For Laplace Noise
    # Max_value = 8 + 11*sig

    def v(z): return GM_bin(z, sig, n)*np.log2(GM_bin(z, sig, n))
    #def v_(z): return GM_unif(z, sig, n)* np.log2(GM_unif(z, sig, n))
    #def v2(z): return LM_bin(z , sig , n)*np.log2 (LM_bin(z , sig , n))
    # def v3(z):
    #    if  UM_bin(z , sig , n) != 0:
    #        return UM_bin(z , sig , n)*np.log2 (UM_bin(z , sig , n))
    #    else:
    #        return 0
    # def v4(z):
    #    if  GM_quad(z , sig , n) != 0:
    #        return GM_quad(z, sig, n)*np.log2(GM_quad(z, sig, n))
    #    else:
    #        return 0
    """ True MI calucated values """
    True_MI1 = -trapezoidal(v, Min_value, Max_value, 4000) - \
        0.5*np.log2(2*math.pi*math.e*pow(sig, 2))

    # True_MI4 = -trapezoidal(v4, Min_value, Max_value, 50) - \
    #    0.5*np.log2(2*math.pi*math.e*pow(sig, 2))

    # True_MI2 = -trapezoidal(v2, Min_value, Max_value, 90) - \
    #    np.log2( 2 * sig * math.e)

    # True_MI3 = -trapezoidal(v3, Min_value, Max_value, 50) - \
    #    np.log2( 2*sig*math.sqrt(3))

    print(True_MI1)


# if __name__== '__main__':
#   main()
