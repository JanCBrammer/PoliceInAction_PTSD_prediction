# -*- coding: utf-8 -*-

import mne
import matplotlib.pyplot as plt
import numpy as np
from biopeaks.ecg import ecg_peaks


def invert_signal(signal):

    signal_mean = np.mean(signal)

    # Invert signal.
    signal_inverted = (signal_mean - signal) + signal_mean

    return signal_inverted

physiopath = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\wave2_raw\PIA_w2_SH_ses1_subj046_280916.vhdr"
# load data per subject
data = mne.io.read_raw_brainvision(physiopath,
                                   preload=True)
ecg = data._data[0, :]

plt.plot(ecg)

ecg = invert_signal(ecg)

plt.plot(ecg)

peaks = ecg_peaks(ecg, 2500)

plt.scatter(peaks, ecg[peaks])


