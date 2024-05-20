#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:41:36 2023

@author: Aakash Chowdhury
"""

import math
import numpy as np
from scipy.stats import dlaplace

f_ = math.factorial

def Normal_(x, mu, sigma):                         #pdf of Normal(Gaussian Noise)
    return math.exp(-pow(x-mu, 2)/(2*pow(sigma, 2)))/math.sqrt(pow(sigma, 2)*2*math.pi)

def Laplace_(x, mu, b):             #pdf of Laplacian distribution
    return  math.exp(-abs(x-mu)/b)/(2*b)

def Bin_(x, n):                                 # Distribution of Binomial (Hammimg Weight/ Hamming distance)
    return f_(n)/(f_(x)*f_(n-x)*pow(2, n))


def GM_bin(z, sig, n):                             # Gaussian Mixture with Binomial coeffcients as weights
    return sum(Bin_(x, n)*Normal_(z, x, sig) for x in range(n+1))   

def GM_unif(z, sig, n):                      # Gaussian Mixture with n-bit uniform leakage model
    return sum((Normal_(z, x, sig)/pow(2, n)) for x in range(pow(2, n)))


def LM_bin(z , sig , n):                          #Laplace Mixture with Binomial coeffcients as weights
    return sum( Bin_(x, n) * Laplace_(z, x, sig)  for x in range(n+1))

def LM_unif(z, sig, n):                         # Laplace Mixture with identity(for nonlinear) device leakage model
    return sum(Laplace_(z, x, sig)/pow(2, n) for x in range(pow(2, n)))

def DLM_bin(z , sig , n):                # Discrete Laplace Mixture with Binomial coeffcients as weights
    return sum( Bin_(x, n) * dlaplace.pmf(z , sig, loc = x)  for x in range(n+1))

def DLM_unif(z, sig, n):  # Discrete Laplace Mixture with identity device leakage model
    return sum(dlaplace.pmf(z , sig, loc = x)/pow(2, n) for x in range(pow(2, n)))

# Process of Numerical integration by trapezoidal rule-------------------------------------
def trapezoidal(f, a, b, n):
    h = float(b - a) / n
    s = 0.0
    s += f(a)/2.0
    for i in range(1, n-1):
        s += f(a + i*h)
    s += f(b)/2.0
    return s * h

def main1():
    # True MI computation for paired random variables (X,Y)---------------------------------
    '''
    True MI calculation for (Univariate Leakage + Gaussian noise)---------------------------------------
    '''
    
    sig = 10  # Change the sigma value
    n = 8
    # Declration of Maximum and Minimum Value-----------------------------------------------
    Min_value = 0 - 6*sig             # For Gaussian Noise and HW device_lkg
    Max_value = n + 6*sig
    
    # Min_value= 0 - 6*sig               # For Gaussian Noise and n-bit uniform model                                   
    # Max_value= pow(2,n)-1 + 6*sig      # Used for non-linear leakage model
    
    
    def v(z): return GM_bin(z, sig, n)*np.log(GM_bin(z, sig, n)) #part inside the integration
    def v_(z): return GM_bin(z, sig, n)*np.log2(GM_bin(z, sig, n)) 
    def u_(z): return GM_unif(z, sig, n)* np.log2(GM_unif(z, sig, n))
    
    #Finally, True MI calculation------------------------------------------------------------
    #True_MI_base_e = -trapezoidal(v, Min_value, Max_value, 4000) - 0.5*np.log(2*math.pi*math.e*pow(sig, 2))
    #True_MI_ = -trapezoidal(u_, Min_value, Max_value, 4000) - 0.5*np.log2(2*math.pi*math.e*pow(sig, 2))
    
    True_MI_ = -trapezoidal(v_, Min_value, Max_value, 4000) - 0.5*np.log2(2*math.pi*math.e*pow(sig, 2))
    
    #print(True_MI_base_e)
    print(True_MI_)

def main2():    
    '''
    True MI calculation for (Univariate Leakage + Laplacian noise)---------------------------------------
    '''
    
    sig = 10    #Change the sigma value
    n = 8
    Min_value_ = 0 - 10.819778284410283*sig             #For Laplace Noise         
    Max_value_ = n + 10.819778284410283*sig
    
    # Min_value_ = 0 - 10.819778284410283*sig               # For Laplace Noise and n-bit uniform model
    # Max_value_ = pow(2,n)-1 + 10.819778284410283*sig
        
    def v2(z): return LM_bin(z , sig , n)*np.log2(LM_bin(z , sig , n))
    def v2_(z): return LM_bin(z , sig , n)*np.log(LM_bin(z , sig , n))
    def u2_(z): return LM_unif(z, sig, n)* np.log2(LM_unif(z, sig, n))
    
    #True_MI2_base_e = -trapezoidal(v2_, Min_value_, Max_value_, 4000) - np.log( 2 * sig * math.e)
    True_MI2_ = -trapezoidal(v2, Min_value_, Max_value_, 4000) - np.log2( 2 * sig * math.e)
    # True_MI2_ = -trapezoidal(u2_, Min_value_, Max_value_, 4000) - np.log2( 2 * sig * math.e)
    print(True_MI2_)
    
def main3():    
    '''
    True MI calculation for (Univariate Leakage + discrete Laplacian noise)---------------------------------
    '''
    # Computing the minimum and maximum value of the summation
    n = 4
    sig = 0.25
    interv = dlaplace.interval( 0.999, a = sig, loc=0)
    max_value = 0 + interv[0] 
    min_value = n + interv[1]
    #z_entrop1 = -sum( [  DLM_bin(z , sig , n)*np.log2(DLM_bin(z , sig , n))   for z in range( 0 - 64, 8 + 64 + 1, 1)] )
    z_entrop2 = -sum( [  DLM_unif(z , sig , n)*np.log2(DLM_unif(z , sig , n))   for z in range( 0 - 64, 4 + 64 + 1, 1)] )
    print(z_entrop2- dlaplace.entropy(a = sig, loc=0) * np.log2(math.e))
    
def main4():
    '''
    True MI calculation for Multivariate Leakage + Continuous noise----------------------------------------
    '''
    # For bivariate data HW-HW or HW-HD ---------------------------------------------------------------------------
    sig = 4
    n = 4
    
    Min_value = 0 - 6*sig             # For Gaussian Noise 
    Max_value = n + 6*sig
    
    # Min_value = 0 - 10.819778284410283*sig             #For Laplace Noise          
    # Max_value = n + 10.819778284410283*sig
    
    def v_(z): return GM_bin(z, sig, n)*np.log2(GM_bin(z, sig, n)) 
    # def u_(z): return LM_bin(z, sig, n)*np.log2(LM_bin(z, sig, n)) 
    
    True_MI_base_2 = -2*trapezoidal(v_, Min_value, Max_value, 4000) - np.log2(2*math.pi*math.e*pow(sig, 2))
    # True_MI_base_2 = -2*trapezoidal(u_, Min_value, Max_value, 4000) - 2* np.log2( 2 * sig * math.e) 
    
    print(True_MI_base_2)                                                                         
    # 0.08746264330880749  #Gaussian noise 
    # 0.07498632668639615  #Laplacian noise
    # For Tri-variate (HW-HW-HW or HW-HW-HD) ----------------------------------------------------------------------
    # sig = 2
    # n_1 = 2
    # n_2 = 4
    # Min_value = 0 - 6*sig             
    # Max_value_1 = n_1 + 6*sig
    # Max_value_2 = n_2 + 6*sig
    
    # def v_1(z): return GM_bin(z, sig, n_1)*np.log2(GM_bin(z, sig, n_1))
    # def v_2(z): return GM_bin(z, sig, n_2)*np.log2(GM_bin(z, sig, n_2))
    
    # True_MI_3 = -2*trapezoidal(v_1, Min_value, Max_value_1, 4000) - trapezoidal(v_2, Min_value, Max_value_2, 4000) - (3*0.5*np.log2(2*math.pi*math.e*pow(sig, 2)))
    # print(True_MI_3) #0.33086741494644123
    
    # For 4-variate (HW-HW-HD-HD/ HW-HW-HW-HW) ----------------------------------------------------------------------------------
    # sig = 2
    # n = 2
    # Min_value = 0 - 6*sig   #For Gaussian Noise-------------------------------
    # Max_value = n + 6*sig
    
    # def v_(z): return GM_bin(z, sig, n)*np.log2(GM_bin(z, sig, n))
    
    # True_MI_4 = -4*trapezoidal(v_, Min_value, Max_value, 4000) - (4*0.5*np.log2(2*math.pi*math.e*pow(sig, 2)))
    # print(True_MI_4) # 0.339831313347835
if __name__== '__main__':
    main4()
#    main1()
#    main2()
#    main3()
        