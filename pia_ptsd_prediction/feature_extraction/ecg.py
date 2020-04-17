# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:51:14 2020

@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.analysis_utils import (invert_signal, decimate_signal,
                                  interpolate_signal)
from biopeaks.heart import ecg_peaks, correct_peaks
from config import (ecg_channels, ecg_sfreq_original, ecg_sfreq_decimated,
                    ecg_period_sfreq)


def preprocessing(readpath, writepath, show=False):

    raw = mne.io.read_raw_brainvision(readpath, preload=False, verbose="error")
    ecg = raw.get_data(picks=ecg_channels).ravel()
    sfreq = raw.info["sfreq"]

    # Raise AssertionError for unexpected sampling frequencies.
    assert sfreq == ecg_sfreq_original, (f"Sampling frequency {sfreq} doesn't"
                                         " match expected sampling frequency"
                                         f" {ecg_sfreq_original}.")

    # Decimate the ECG from original sampling rate to 500 HZ.
    decimation_factor = int(np.floor(sfreq / ecg_sfreq_decimated))
    ecg_decimated = decimate_signal(ecg, decimation_factor)
    # Flip the inverted ECG signal (around time axis).
    ecg_inverted = invert_signal(ecg_decimated)

    pd.Series(ecg_inverted).to_csv(writepath, sep="\t", header=False,
                                   index=False, float_format="%.4f")

    if show:
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        sec = np.linspace(0, len(ecg) / sfreq, len(ecg))
        ax0.plot(sec, ecg, label=f"original ({sfreq}Hz)")
        ax0.set_xlabel("seconds")
        ax0.legend(loc="upper right")
        sec = np.linspace(0, len(ecg_decimated) / ecg_sfreq_decimated,
                          len(ecg_decimated))
        ax1.plot(sec, ecg_decimated,
                 label=f"downsampled ({ecg_sfreq_decimated}Hz)")
        ax1.plot(sec, ecg_inverted,
                 label=f"flipped ({ecg_sfreq_decimated}Hz)")
        ax1.set_xlabel("seconds")
        ax1.legend(loc="upper right")


def peaks(readpath, writepath, show=False):

    ecg = np.ravel(pd.read_csv(readpath, sep="\t"))
    # Detect R-peaks.
    peaks = ecg_peaks(ecg, ecg_sfreq_decimated)
    # Correct artifacts in peak detection.
    peaks_corrected = correct_peaks(peaks, ecg_sfreq_decimated,
                                    iterative=True)
    # Save peaks as samples.
    pd.Series(peaks_corrected).to_csv(writepath, sep="\t", header=False,
                                      index=False, float_format="%.4f")

    if show:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        sec = np.linspace(0, len(ecg) / ecg_sfreq_decimated, len(ecg))
        ax.plot(sec, ecg)
        ax.scatter(sec[peaks], ecg[peaks], zorder=3, c="r", marker="+", s=300,
                   label="uncorrected R-peaks")
        ax.scatter(sec[peaks_corrected], ecg[peaks_corrected], zorder=4,
                   c="g", marker="x", s=300, label="corrected R-peaks")
        ax.set_xlabel("seconds")
        ax.legend(loc="upper right")


def period(readpath, writepath, show=False):

    peaks = np.ravel(pd.read_csv(readpath, sep="\t"))

    # Compute period in milliseconds.
    period = np.ediff1d(peaks, to_begin=0) / ecg_sfreq_decimated * 1000    # make sure period has same number of elements as peaks
    period[0] = period[1]    # make sure that the first element has a realistic value

    # Interpolate instantaneous heart period at 4 Hz. Interpolate up until the
    # last R-peak.
    duration = peaks[-1] / ecg_sfreq_decimated    # in seconds
    nsamples = int(np.rint(duration * ecg_period_sfreq))
    period_interpolated = interpolate_signal(peaks, period, nsamples)

    pd.Series(period_interpolated).to_csv(writepath, sep="\t", header=False,
                                          index=False, float_format="%.6f")

    if show:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        sec = np.linspace(0, duration, peaks[-1])
        ax.vlines(sec[peaks[:-1]], ymin=min(period), ymax=max(period),
                  label="R-peaks", alpha=.3, colors="r")
        sec = np.linspace(0, duration, nsamples)
        ax.plot(sec, period_interpolated,
                label=("period interpolated between R-peaks at"
                       f" {ecg_period_sfreq}Hz"))
        ax.set_xlabel("seconds")
        ax.legend(loc="upper right")
