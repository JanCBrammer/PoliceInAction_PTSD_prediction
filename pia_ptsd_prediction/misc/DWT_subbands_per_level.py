# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:03:40 2019

@author: John Doe
"""

import numpy as np


def subband(n, sfreq, levels):
    
#    subbands = []

    for level in np.arange(1, levels + 1, dtype=float):
        
        lower, upper = 2**-(level + 1) * sfreq, (2**-level) * sfreq
        
        print('in the band between {} and {}, DWT returns {} coefficients'.format(lower, upper, n / (level**2)))
#        subbands.append((lower, upper))
        
#    return subbands


subband(1000, 4, 4)