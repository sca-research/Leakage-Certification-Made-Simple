#!/usr/bin/python3

import numpy as np

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
    '''

    Parameters
    ----------
    num : An 8-bit intiger (Can be generalized for an arbitrary n-bit)
    k : extract bits in 'k' components

    Returns
    -------
    list
    Produce 'k' independent 2-bit values required for simulating multivariate traces

    '''

    binary = "{:08b}".format(num)  # for n-bit, 8 should be replaced by n

    if k == 3:
        return [int(binary[:2], 2),  int(binary[2:4], 2), int(binary[4:], 2)]
    if k == 4:
        return [int(binary[:2], 2), int(binary[2:4], 2), int(binary[4:6], 2), int(binary[6:], 2)]


# Univariate Leakage (Unmasked implementation)-------------------------------------
def univariate_trace(Encrypt_key, n_traces, sigma):
    '''
    Parameters
    ----------
    n_traces : Number of traces
    Encrypt_key : Secret Key

    sigma : Specify the standard deviation of Noise variable

    Returns
    -------
    trace : a list of univariate Traces
    pt_ : a list of univariate input plain_text
    
    '''
    trace = []
    pt_ = []
    for text in np.arange(0, n_traces):
        ''' Linear Model--------------------------------------------------------'''
        pt = np.random.randint(0, 256, dtype=np.uint8)
        inter_med = AES_Sbox[pt ^ Encrypt_key]
        # wt = 0x71     # Define linear weights
        # x = float(ham_wt(inter_med & wt))  # Weighted Hamming Weight
        # x = float(ham_dist(inter_med, pt[text] ^ key)) # hamming distance
        x = float(ham_wt(inter_med))   # Hamming weight
        # x= float(lkg_5LSB(inter_med))    # 5-Least  significant bits of Intermediate

        ''' Nonlinear Model----------------------------------------------------'''
        # x = float(Non_lin(inter_med)) # Double cryptographic permutation

        ''' Adding Continuous noise with leakage model ouptput-----------------'''
        trace.append(x + float(np.random.normal(0, sigma, 1)))  # Gaussian noise
        # trace.append(x + float(np.random.laplace(0,sigma,1)))  ## laplace noise
        # trace.append(x + float(np.random.uniform(-sigma*math.sqrt(8), sigma*math.sqrt(8), 1))) ## Uniform Noise
        # trace.append(x + float(np.random.lognormal(0, sigma, 1)))

        ''' Discrete Leakage Simulation---------------------------------------'''
        # x = int(ham_wt(inter_med))
        # x = int(ham_dist(inter_med, pt[text] ^ key))
        # x = int(Non_lin(inter_med))
        # x = int(ham_wt(inter_med & wt))
        ''' Adding Continuous noise with leakage model ouptput----------------'''
        # trace.append(x + int(dlaplace.rvs(sigma, 1)))
        # trace.append( x + int(np.random.poisson(sigma)))
        pt_.append(pt)
        
    return trace, pt_


def Bivariate_trace(Encrypt_key, n_traces, sigma_val):
    ''' Bivariate Trace Generation'''

    Tr = []
    # Tr_1 = []
    # Tr_2 = []
    plain_text = []
    for i in np.arange(n_traces):
        pt = np.random.randint(0, 256, dtype=np.uint8)
        # m = np.random.randint(0, 256, dtype = np.uint8) # msk implimentation

        inter_md = AES_Sbox[pt ^ Encrypt_key]
        # intr_md_msk = inter_md ^ m
        intr_md_1 = lkg_4LSB(inter_md)
        intr_md_2 = lkg_4MSB(inter_md)

        # Tr1 = float(ham_wt(intr_md_1)) + float(np.random.laplace(0,sig,1))
        Tr1 = float(ham_wt(intr_md_1)) + \
            float(np.random.normal(0, sigma_val, 1))

        # Tr2 = float(ham_wt(intr_md_2)) + float(np.random.laplace(0,sig,1))
        Tr2 = float(ham_wt(intr_md_2)) + \
            float(np.random.normal(0, sigma_val, 1))

        Trc = [Tr1, Tr2]

        plain_text.append(pt)
        Tr.append(Trc)

    return Tr, plain_text




def Multivariate_trace(Encrypt_key, n_traces, sigma_val, k = 3):

 #  For multivariate leakage simulation---------------------------------------------------

    inter_mediate = []
    # plain_text = []
    Tr = []
    for txt in np.arange(0, n_traces):
        pt = np.random.randint(0, 256, dtype=np.uint8)

        intr_md = AES_Sbox[pt ^ Encrypt_key]
        
        Intr_md_extr = extractBits( intr_md, k)   # k=4 for 4-variate convergence

        Trc =  [ float(ham_wt(x)) + float(np.random.normal(0,sigma_val,1))  for x in Intr_md_extr]



        Tr.append(Trc)
        inter_mediate.append(intr_md)
        # plain_text.append(pt)
        del Trc
        del intr_md
        # del pt
    return Tr, inter_mediate
