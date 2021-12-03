# -*- coding: utf-8 -*-
"""
@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

# General ######################################################################
ROOTDIR = "./data"
DATADIR_RAW = "./data/raw"
DATADIR_PROCESSED = "./data/processed"
SUBJECTS = [f"subj{str(i).zfill(3)}" for i in range(1, 428)]    # subjects 1 trough 427

# ECG ##########################################################################
ECG_CHANNELS = ["ECG"]
ECG_SFREQ_ORIGINAL = 2500
ECG_SFREQ_DECIMATED = 500    # Hz
# The sampling frequency of the heart period must not be too low in order to
# not loose too much temporal precision during event-related analyses. See
# `demo_influcence_sfreq_event_timing`.
ECG_PERIOD_SFREQ = 16    # Hz

# Balance-board  ###############################################################
BB_CHANNELS = ["BB1", "BB2", "BB3", "BB4"]
BB_SFREQ_ORIGINAL = 2500
BB_SFREQ_DECIMATED = 32    # Hz
BB_FILTER_CUTOFFS = [.01, 10]    # lowcut (highpass) and highcut (lowpass) in Hz
BB_MIN_EMPTY = 10    # seconds
# TODO: verify board length!
BB_BOARDLENGTH = 425
BB_MOVING_WINDOW = 1    # seconds
