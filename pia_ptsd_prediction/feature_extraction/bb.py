# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:51:14 2020

@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from config import bb_channels, bb_sfreq_decimated
from utils.analysis_utils import decimate_signal


def preprocessing(readpath, writepath, show=False):

    raw = mne.io.read_raw_brainvision(readpath, preload=False, verbose="error")
    bb = raw.get_data(picks=bb_channels)
    sfreq = raw.info["sfreq"]

    # Decimate the four balance board channels from original sampling rate to
    # 20 HZ. Note that MNE's raw.apply_function() cannot be used since it
    # requires the preservation of the original sampling frequency.
    decimation_factor = int(np.ceil(sfreq / bb_sfreq_decimated))
    bb_decimated = decimate_signal(bb, decimation_factor)

    print(bb_decimated.shape)

