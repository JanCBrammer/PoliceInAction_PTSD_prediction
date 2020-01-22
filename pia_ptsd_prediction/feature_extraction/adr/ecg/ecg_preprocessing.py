# -*- coding: utf-8 -*-

import glob
import mne
import matplotlib.pyplot as plt
import numpy as np
from biopeaks.ecg import ecg_peaks, ecg_cleanperiods


def invert_signal(signal):

    signal_mean = np.mean(signal)
    signal_inverted = (signal_mean - signal) + signal_mean

    return signal_inverted

# Get all
physiopaths = glob.glob(r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data\wave2_raw\*.vhdr")


for physiopath in physiopaths[:2]:

    data = mne.io.read_raw_brainvision(physiopath,
                                       preload=True)
    sfreq = data.info['sfreq']
    ecg = data._data[0, :]
    ecg = invert_signal(ecg)    # the ECG is inverted

    peaks = ecg_peaks(ecg, sfreq)

    # plt.plot(np.ediff1d(peaks, to_begin=0) / 2500)

    peaks_corrected, nn, artifacts = ecg_cleanperiods(peaks, sfreq,
                                                      return_artifacts=True)

    if len(artifacts["missed"]) > 0:
        break

# plt.plot(nn)


