# -*- coding: utf-8 -*-

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import decimate
from biopeaks.heart import ecg_peaks, correct_peaks, heart_period
from config import ecg_sfreq_downsampled


def _invert_signal(signal):

    signal_mean = np.mean(signal)
    signal_inverted = (signal_mean - signal) + signal_mean

    return signal_inverted


def preprocessing(readpath, writepath, show=False):

    data = mne.io.read_raw_brainvision(readpath, preload=True, verbose="error")
    ecg = data._data[0, :]
    sfreq = data.info["sfreq"]

    # Downsample the ECG from original sampling rate to 500 HZ.
    downsampling_factor = int(np.ceil(sfreq / ecg_sfreq_downsampled))
    ecg_downsampled = decimate(ecg, q=downsampling_factor, zero_phase=True)
    # Flip the inverted ECG signal (around time axis).
    ecg_inverted = _invert_signal(ecg_downsampled)

    pd.Series(ecg_inverted).to_csv(writepath, sep="\t", header=False,
                                   index=False, float_format="%.4f")

    if show:
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        sec = np.linspace(0, len(ecg) / sfreq, len(ecg))
        ax0.plot(sec, ecg, label=f"original ({sfreq}Hz)")
        ax0.legend(loc="upper right")
        sec = np.linspace(0, len(ecg_downsampled) / ecg_sfreq_downsampled,
                          len(ecg_downsampled))
        ax1.plot(sec, ecg_downsampled,
                 label="downsampled ({ecg_sfreq_downsampled}Hz)")
        ax1.plot(sec, ecg_inverted, label="flipped ({ecg_sfreq_downsampled}Hz)")
        ax1.legend(loc="upper right")


def peaks(readpath, writepath, show=False):

    ecg = np.ravel(pd.read_csv(readpath, sep='\t'))
    # Detect R-peaks
    peaks = ecg_peaks(ecg, ecg_sfreq_downsampled)
    # Correct artifacts in peak detection.
    peaks_corrected = correct_peaks(peaks, ecg_sfreq_downsampled,
                                    iterative=True)

    pd.Series(peaks_corrected).to_csv(writepath, sep="\t", header=False,
                                      index=False, float_format="%.4f")

    if show:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        sec = np.linspace(0, len(ecg) / ecg_sfreq_downsampled, len(ecg))
        ax.plot(sec, ecg)
        ax.scatter(sec[peaks], ecg[peaks], zorder=3, c="r", marker="+", s=300,
                   label="uncorrected R-peaks")
        ax.scatter(sec[peaks_corrected], ecg[peaks_corrected], zorder=4,
                   c="g", marker="x", s=300, label="corrected R-peaks")
        ax.legend(loc="upper right")


def period(readpath, writepath, show=False):
    pass
