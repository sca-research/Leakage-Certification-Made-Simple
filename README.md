## Leakage Certification Made Simple
The project is focused on certifying side-channel leakage using mutual information in simulated and public datasets. This research was conducted by the Cybersecurity research group at the University of Klagenfurt, Austria. If you want more insight into this pioneering work, we invite you to explore our archived version [eprint](https://eprint.iacr.org/archive/2022/1201).

## General Introduction
The project contains the `Python` and `C++` code for fast evaluation of different MI estimators, considered in side-channel leakage certification. User guide and detailed instructions are provided in two separate folders in [Python](https://github.com/sca-research/Leakage-Certification-Made-Simple/tree/main/Python_implementation) and [CPP](https://github.com/sca-research/Leakage-Certification-Made-Simple/tree/main/CPP_implementation)

Based on our running time, we recommend using CPP code for univariate and bivariate experiments and Python code for higher dimensional (> 2) experiments as it utilizes the `cKDTree` module from `scipy` for vectorized parallel tree search algorithm.    

## Datasets
We have considered both simulation and practical datasets.
-  In simulation experiments, we have considered different linear leakage models, like, hamming weight, hamming distance, weighted hamming weight, and one non-linear model (by considering the double permutation). Along with leakage models we also consider the Gaussian and non-gaussian additive noises.
- Two practical datasets are also used:
[LPC1313](https://zenodo.org/records/11396347), [AES_HD_ext](https://github.com/AISyLab/AES_HD_Ext)

## Acknowldgement
This project is supported in part by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (grant agreement No 725042), and by the Austrian Science Fund (FWF) 10.55776/F85 (SFB SpyCode).

 ![EU Logo](https://github.com/sca-research/Leakage-Certification-Made-Simple/blob/main/LOGO_ERC-FLAG_EU.jpg)
 
 ![FWF Logo](https://github.com/sca-research/Leakage-Certification-Made-Simple/blob/main/FWF_Logo_Zusatz_Dunkelblau_RGB_EN.png)
 
 
