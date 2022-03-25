"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import mne
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from numpy.ma.extras import count_masked
import pandas as pd
from pathlib import Path
from biopeaks.heart import ecg_peaks, correct_peaks
from scipy.stats import median_absolute_deviation
from scipy.ndimage import median_filter
from pia_ptsd_prediction.utils.analysis_utils import (invert_signal,
                                                      decimate_signal,
                                                      interpolate_signal)
from pia_ptsd_prediction.utils.io_utils import individualize_path
from pia_ptsd_prediction.config import (ECG_CHANNELS, ECG_SFREQ_ORIGINAL,
                                        ECG_SFREQ_DECIMATED, ECG_PERIOD_SFREQ,
                                        HR_MIN, HR_MAX, RUNNING_MEDIAN_KERNEL_SIZE,
                                        MAD_THRESHOLD_MULTIPLIER)


def preprocess_ecg(subject, inputs, outputs, recompute, logpath):
    """Preprocessing of raw ECG from BrainVision files.

    1. downsample from 2500Hz to 500Hz
    2. flip inverted signal
    """
    save_path = Path(individualize_path(outputs["save_path"], subject, expand_name=True))
    if save_path.exists() and not recompute:    # only recompute if requested
        print(f"Not re-computing {save_path}")
        return
    physio_path = next(Path(".").glob(individualize_path(inputs["physio_path"], subject)))

    raw = mne.io.read_raw_brainvision(physio_path, preload=False, verbose="error")
    ecg = raw.get_data(picks=ECG_CHANNELS).ravel()
    sfreq = raw.info["sfreq"]
    assert sfreq == ECG_SFREQ_ORIGINAL, (f"Sampling frequency {sfreq} doesn't"
                                         " match expected sampling frequency"
                                         f" {ECG_SFREQ_ORIGINAL}.")

    # Decimate the ECG from original sampling rate to 500 HZ.
    decimation_factor = int(np.floor(sfreq / ECG_SFREQ_DECIMATED))
    ecg_decimated = decimate_signal(ecg, decimation_factor)
    # Flip the inverted ECG signal.
    ecg_inverted = invert_signal(ecg_decimated)

    pd.Series(ecg_inverted).to_csv(save_path, sep="\t", header=False,
                                   index=False, float_format="%.4f")

    if not logpath:
        return

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    sec = np.linspace(0, len(ecg) / sfreq, len(ecg))
    ax0.plot(sec, ecg, label=f"original ({sfreq}Hz)")
    ax0.set_xlabel("seconds")
    ax0.legend(loc="upper right")
    sec = np.linspace(0, len(ecg_decimated) / ECG_SFREQ_DECIMATED,
                      len(ecg_decimated))
    ax1.plot(sec, ecg_decimated, label=f"downsampled ({ECG_SFREQ_DECIMATED}Hz)")
    ax1.plot(sec, ecg_inverted, label=f"flipped ({ECG_SFREQ_DECIMATED}Hz)")
    ax1.set_xlabel("seconds")
    ax1.legend(loc="upper right")

    fig.savefig(logpath, dpi=200)
    plt.close(fig)



def get_peaks_ecg(subject, inputs, outputs, recompute, logpath):
    """Detect R-peaks in ECG.

    1. Detect R-peaks
    2. autocorrect artifacts in R-peaks detection.
    """
    save_path = Path(individualize_path(outputs["save_path"], subject, expand_name=True))
    if save_path.exists() and not recompute:    # only recompute if requested
        print(f"Not re-computing {save_path}")
        return
    physio_path = next(Path(".").glob(individualize_path(inputs["physio_path"], subject)))

    ecg = np.ravel(pd.read_csv(physio_path, sep="\t", header=None))
    # Detect R-peaks.
    peaks = ecg_peaks(ecg, ECG_SFREQ_DECIMATED)
    # Correct artifacts in peak detection.
    peaks_corrected = correct_peaks(peaks, ECG_SFREQ_DECIMATED,
                                    iterative=True)
    # Save peaks as samples.
    pd.Series(peaks_corrected).to_csv(save_path, sep="\t", header=False,
                                      index=False, float_format="%.4f")

    if not logpath:
        return

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sec = np.linspace(0, len(ecg) / ECG_SFREQ_DECIMATED, len(ecg))
    ax.plot(sec, ecg)
    ax.scatter(sec[peaks], ecg[peaks], zorder=3, c="r", marker="+", s=300,
               label="uncorrected R-peaks")
    ax.scatter(sec[peaks_corrected], ecg[peaks_corrected], zorder=4,
               c="g", marker="x", s=300, label="corrected R-peaks")
    ax.set_xlabel("seconds")
    ax.legend(loc="upper right")

    fig.savefig(logpath, dpi=200)
    plt.close(fig)


def get_period_ecg(subject, inputs, outputs, recompute, logpath):
    """Compute continuous heart period.

    1. Compute inter-beat-intervals
    2. Interpolate inter-beat-intervals to time series sampled at ECG_PERIOD_SFREQ Hz.
    """
    save_path = Path(individualize_path(outputs["save_path"], subject, expand_name=True))
    if save_path.exists() and not recompute:    # only recompute if requested
        print(f"Not re-computing {save_path}")
        return
    physio_path = next(Path(".").glob(individualize_path(inputs["physio_path"], subject)))

    peaks = np.ravel(pd.read_csv(physio_path, sep="\t", header=None))

    # Compute period in milliseconds.
    period = np.ediff1d(peaks, to_begin=0) / ECG_SFREQ_DECIMATED * 1000    # make sure period has same number of elements as peaks
    period[0] = period[1]    # make sure that the first element has a realistic value

    # Interpolate instantaneous heart period at ECG_PERIOD_SFREQ Hz. Interpolate up until the
    # last R-peak.
    duration = peaks[-1] / ECG_SFREQ_DECIMATED    # in seconds
    nsamples = int(np.rint(duration * ECG_PERIOD_SFREQ))
    period_interpolated = interpolate_signal(peaks, period, nsamples)

    pd.Series(period_interpolated).to_csv(save_path, sep="\t", header=False,
                                          index=False, float_format="%.6f")

    if not logpath:
        return

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    sec = np.linspace(0, duration, peaks[-1])
    ax.vlines(sec[peaks[:-1]], ymin=min(period), ymax=max(period),
              label="R-peaks", alpha=.3, colors="r")
    sec = np.linspace(0, duration, nsamples)
    ax.plot(sec, period_interpolated,
            label=("period interpolated between R-peaks at"
                    f" {ECG_PERIOD_SFREQ}Hz"))
    ax.set_xlabel("seconds")
    ax.legend(loc="upper right")

    fig.savefig(logpath, dpi=200)
    plt.close(fig)


def remove_outliers_period_ecg(subject, inputs, outputs, recompute, logpath):
    """Remove outliers from heart period series."""
    save_path = Path(individualize_path(outputs["save_path"], subject, expand_name=True))
    if save_path.exists() and not recompute:    # only recompute if requested
        print(f"Not re-computing {save_path}")
        return
    physio_path = next(Path(".").glob(individualize_path(inputs["physio_path"], subject)))

    period = np.ravel(pd.read_csv(physio_path, sep="\t", header=None))

    # Remove outliers based on absolute cutoffs. Those cutoffs have been chosen
    # based on the visual inspection of all heart period time series data. The
    # cutoffs have been set such that they preserve the data as much as possible
    # (when in doubt don't flag a period as outlier).
    min_period = 60000 / HR_MAX
    max_period = 60000 / HR_MIN
    abs_outliers = np.where((period < min_period) | (period > max_period))

    # Median filter period with absolute outliers removed.
    period_without_abs_outliers = period.copy()
    period_without_abs_outliers[abs_outliers] = np.median(period)
    kernel = int(np.rint(ECG_PERIOD_SFREQ * RUNNING_MEDIAN_KERNEL_SIZE))
    if not kernel % 2: kernel += 1
    period_trend = median_filter(period_without_abs_outliers, size=kernel)

    # Remove outliers based on relative cutoffs.
    rel_threshold = MAD_THRESHOLD_MULTIPLIER * median_absolute_deviation(period_without_abs_outliers)
    upper_rel_threshold = period_trend + rel_threshold
    lower_rel_threshold = period_trend - rel_threshold
    period_masked_outliers = ma.masked_where((period < lower_rel_threshold) |
                                             (period > upper_rel_threshold),
                                             period)
    assert period_masked_outliers.size == period.size

    pd.Series(ma.filled(period_masked_outliers,
                        fill_value=np.nan)).to_csv(save_path, sep="\t", header=False,
                                                   index=False, float_format="%.6f",
                                                   na_rep="NaN")

    if not logpath:
        return

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    sec = np.linspace(0, period.size / ECG_PERIOD_SFREQ, period.size)
    ax0.plot(sec, period)
    ax0.fill_between(sec, upper_rel_threshold, lower_rel_threshold,
                     color="lime", alpha=.5)
    ax0.plot(sec, period_trend, c="lime")
    ax1.vlines(sec[period_masked_outliers.mask], period_masked_outliers.min(),
               period_masked_outliers.max(), colors="fuchsia")
    ax1.plot(sec, period_masked_outliers)
    ax1.set_xlabel("seconds")

    fig.savefig(logpath, dpi=200)
    plt.close(fig)
