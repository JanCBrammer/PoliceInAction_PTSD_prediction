# -*- coding: utf-8 -*-

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import decimate
from scipy.interpolate import interp1d
from biopeaks.heart import ecg_peaks, correct_peaks
from config import ecg_sfreq_downsampled, period_sfreq


def _invert_signal(signal):

    signal_mean = np.mean(signal)
    signal_inverted = (signal_mean - signal) + signal_mean

    return signal_inverted


def _decimate_signal(signal, decimation_factor):

    if decimation_factor > 12:
        print("Decimation-factors larger than 12 leads to signal distortions")

    decimated = decimate(signal, decimation_factor, ftype="iir",
                         zero_phase=True)

    return decimated


def _interpolate_period(peaks, period, nsamples):

    samples = np.linspace(0, peaks[-1], nsamples)

    f = interp1d(peaks, period, kind='slinear', bounds_error=False,
                 fill_value=([period[0]], [period[-1]]))

    period_interpolated = f(samples)

    return period_interpolated


def preprocessing(readpath, writepath, show=False):

    data = mne.io.read_raw_brainvision(readpath, preload=True, verbose="error")
    ecg = data._data[0, :]
    sfreq = data.info["sfreq"]

    # Decimate the ECG from original sampling rate to 500 HZ.
    decimation_factor = int(np.ceil(sfreq / ecg_sfreq_downsampled))
    ecg_downsampled = _decimate_signal(ecg, decimation_factor)
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
                 label=f"downsampled ({ecg_sfreq_downsampled}Hz)")
        ax1.plot(sec, ecg_inverted,
                 label=f"flipped ({ecg_sfreq_downsampled}Hz)")
        ax1.legend(loc="upper right")


def peaks(readpath, writepath, show=False):

    ecg = np.ravel(pd.read_csv(readpath, sep="\t"))
    # Detect R-peaks.
    peaks = ecg_peaks(ecg, ecg_sfreq_downsampled)
    # Correct artifacts in peak detection.
    peaks_corrected = correct_peaks(peaks, ecg_sfreq_downsampled,
                                    iterative=True)
    # Save peaks as samples.
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

    peaks = np.ravel(pd.read_csv(readpath, sep="\t"))

    # Compute period in milliseconds.
    period = np.ediff1d(peaks, to_begin=0) / ecg_sfreq_downsampled * 1000    # make sure period has same number of elements as peaks
    period[0] = period[1]    # make sure that the first element has a realistic value

    # Interpolate instantaneous heart period at 4 Hz. Interpolate up until the
    # last R-peak.
    duration = peaks[-1] / ecg_sfreq_downsampled    # in seconds
    nsamples = int(np.rint(duration * period_sfreq))
    period_interpolated = _interpolate_period(peaks, period, nsamples)

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
                        f" {period_sfreq}Hz"))
        ax.legend(loc="upper right")
