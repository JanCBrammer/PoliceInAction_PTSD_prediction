# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:51:14 2020

@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

# ECG configuration ###########################################################
ecg_channels = [0]
ecg_sfreq_decimated = 500    # Hz
# The sampling frequency of the heart period must not be too low in order to
# not loose too much temporal precision during event-related analyses. See
# `demo_influcence_sfreq_event_timing`.
period_sfreq = 10    # Hz

# Balance-board configuration #################################################
bb_channels = [1, 2, 3, 4]
bb_sfreq_decimated = 20    # Hz


