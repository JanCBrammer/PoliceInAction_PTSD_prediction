# -*- coding: utf-8 -*-

from io_utils import get_subjectpath, make_subjectpath
import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biopeaks.ecg import ecg_peaks, ecg_period


def invert_signal(signal):

    signal_mean = np.mean(signal)
    signal_inverted = (signal_mean - signal) + signal_mean

    return signal_inverted


rootdir_read = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data\raw"
subdir_read = r"shootingtask\physiology"
rootdir_write = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data\processed"
subdir_write = r"adr\ecg"
regex = "*.vhdr"

subjdirs = os.listdir(rootdir_read)

fig, ax = plt.subplots(nrows=1, ncols=1)

for subject in subjdirs:

    # Skip everything other than subject directories.
    if subject[:4] != "subj":
        continue

    subjpath_read = get_subjectpath(rootdir_read, subject, subdir_read, regex)
    data = mne.io.read_raw_brainvision(subjpath_read, preload=True)
    sfreq = data.info["sfreq"]
    ecg = data._data[0, :]
    ecg = invert_signal(ecg)    # flip back the inverted ECG signal

    peaks = ecg_peaks(ecg, sfreq)
    # Correct artifacts in peak detection and compute instantaneous heart
    # period.
    peaks_corrected, period, _ = ecg_period(peaks, sfreq, nsamp=ecg.size)

    subjpath_write = make_subjectpath(rootdir_write, subject, subdir_write,
                                      f"{subject}_peaks.tsv")

    pd.Series(peaks_corrected).to_csv(subjpath_write, sep="\t", header=False,
                                      index=False)

    subjpath_write = make_subjectpath(rootdir_write, subject, subdir_write,
                                      f"{subject}_period.tsv")
    pd.Series(period).to_csv(subjpath_write, sep="\t", header=False,
                             index=False, float_format="%.4f")

    ax.scatter(np.mean(period), np.std(period))
