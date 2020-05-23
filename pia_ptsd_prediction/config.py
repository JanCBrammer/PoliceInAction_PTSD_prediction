# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:51:14 2020

@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

# ECG configuration ###########################################################
ecg_channels = ["ECG"]
ecg_sfreq_original = 2500
ecg_sfreq_decimated = 500    # Hz
# The sampling frequency of the heart period must not be too low in order to
# not loose too much temporal precision during event-related analyses. See
# `demo_influcence_sfreq_event_timing`.
ecg_period_sfreq = 16    # Hz

# Balance-board configuration #################################################
bb_channels = ["BB1", "BB2", "BB3", "BB4"]
bb_sfreq_original = 2500
bb_sfreq_decimated = 32    # Hz
bb_filter_cutoffs = [.01, 10]    # lowcut (highpass) and highcut (lowpass) in Hz
bb_min_empty = 10    # seconds
# TODO: verify board length!
bb_boardlength = 425



