"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from biopeaks.filters import butter_bandpass_filter
from pia_ptsd_prediction.utils.analysis_utils import (decimate_signal,
                                                      consecutive_samples,
                                                      cop_radius)
from pia_ptsd_prediction.utils.io_utils import individualize_filename
from pia_ptsd_prediction.config import (BB_CHANNELS, BB_SFREQ_ORIGINAL,
                                        BB_SFREQ_DECIMATED, BB_MIN_EMPTY,
                                        BB_BOARDLENGTH, BB_FILTER_CUTOFFS,
                                        BB_MOVING_WINDOW)


def preprocess_bb(subject, inputs, outputs, recompute, logpath):
    """Preprocessing of raw balance board channels from BrainVision files.

    1. downsample from 2500Hz to 32Hz
    2. transform to millimeter unit (board displacement)
    """
    root = outputs["save_path"][0]
    filename = individualize_filename(outputs["save_path"][1], subject)
    save_path = Path(root).joinpath(f"{subject}/{filename}")
    computed = save_path.exists()   # boolean indicating if file already exists
    if computed and not recompute:    # only recompute if requested
        print(f"Not re-computing {save_path}")
        return

    root = inputs["physio_path"][0]
    filename = inputs["physio_path"][1]
    physio_path = list(Path(root).joinpath(subject).glob(filename))

    raw = mne.io.read_raw_brainvision(*physio_path, preload=False, verbose="error")
    bb = raw.get_data(picks=BB_CHANNELS)
    sfreq = raw.info["sfreq"]

    assert sfreq == BB_SFREQ_ORIGINAL, (f"Sampling frequency {sfreq} doesn't"
                                        " match expected sampling frequency"
                                        f" {BB_SFREQ_ORIGINAL}.")

    # Decimate the four balance board channels from original sampling rate to
    # BB_SFREQ_DECIMATED HZ. Note that MNE's raw.apply_function() cannot be used since it
    # requires the preservation of the original sampling frequency.
    decimation_factor = int(np.floor(sfreq / BB_SFREQ_DECIMATED))
    bb_decimated = decimate_signal(bb, decimation_factor)

    # Assuming that the participant has been off the board at some time,
    # calculate the empty board value: For each channel, take the mean of a
    # consecutive chunk of data of at least 10 seconds duration that is below
    # the minimum value + std.
    bb_mins = bb_decimated.min(axis=1) + bb_decimated.std(axis=1)
    bb_minsconsecutive = np.zeros((bb_mins.size, 2)).astype(int)
    bb_empty = np.zeros(bb_mins.size)
    min_duration = int(np.ceil(BB_SFREQ_DECIMATED * BB_MIN_EMPTY))

    for i in range(bb_mins.size):

        begs, ends, n = consecutive_samples(bb_decimated[i, :],
                                            lambda x: x < bb_mins[i],
                                            min_duration)
        assert begs.size > 0, (f"Did not find {BB_MIN_EMPTY} consecutive"
                               " seconds of empty board values.")
        # Find longest chunk and save its beginning and end.
        longest = n.argmax()
        beg = begs[longest]
        end = ends[longest]
        bb_empty[i] = bb_decimated[i, beg:end].mean()
        bb_minsconsecutive[i, 0] = beg
        bb_minsconsecutive[i, 1] = end

    # Calculate weight of the participant.
    bb_chansum = np.sum(bb_decimated, axis=0)    # collapse sensors across time axis
    bb_chansum_empty = bb_empty.sum()
    bb_subjweight = np.median(bb_chansum) - bb_chansum_empty

    assert bb_subjweight > bb_chansum_empty, (f"Subject {bb_subjweight} is not"
                                              " heavier than empty board"
                                              f" ({bb_chansum_empty}).")

    # Transform sensor data to millimeter unit.
    bb_mm = np.subtract(bb_decimated, bb_empty.reshape(-1, 1))
    bb_mm = bb_mm / bb_subjweight    # scale by subject weight
    bb_mm = bb_mm * (BB_BOARDLENGTH / 2)    # express in mm

    pd.DataFrame(bb_mm).T.to_csv(save_path, sep="\t",
                                 header=["BB1", "BB2", "BB3", "BB4"],    # transpose to change from channels as rows to channels as columns (preserves ordering of channels)
                                 index=False, float_format="%.4f")

    if not logpath:
        return

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    sec = np.linspace(0, bb_decimated.shape[1] / BB_SFREQ_DECIMATED,
                      bb_decimated.shape[1])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    channames = ["PR", "PL", "AL", "AR"]

    for chan in range(bb_decimated.shape[0]):
        color = colors[chan]
        channame = channames[chan]
        ax0.plot(sec, bb_decimated[chan, :], c=color, label=f"{channame}")
        ax0.axvspan(xmin=sec[bb_minsconsecutive[chan, 0]],
                    xmax=sec[bb_minsconsecutive[chan, 1]],
                    ymin=bb_decimated[chan].min(),
                    ymax=bb_decimated[chan].max(),
                    color=color, alpha=.2, label="empty")

    ax0.hlines(y=bb_empty, xmin=0, xmax=sec[-1], colors=colors[:bb_mins.size],
               linestyles="dotted", label="empty")
    ax0.legend(loc="upper right")

    ax1.plot(sec, bb_chansum)
    ax1.axhline(y=bb_subjweight, c="r",
                label="subject weight minus board weight")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("seconds")

    for chan in range(bb_mm.shape[0]):
        color = colors[chan]
        channame = channames[chan]
        ax2.plot(sec, bb_mm[chan, :], c=color, label=f"{channame}")
    ax2.legend(loc="upper right")
    ax2.set_xlabel("seconds")
    ax2.set_ylabel("millimeters")

    fig.savefig(logpath, dpi=200)
    plt.close(fig)


def get_cop_bb(subject, inputs, outputs, recompute, logpath):
    """Compute center of pressure time series.

    1. combine preprocessed balance board channels to time series of anterior-
    posterior (forth-back) displacement and medio-lateral (left-right)
    discplacement
    2. Filter displacement time series
    """
    root = outputs["save_path"][0]
    filename = individualize_filename(outputs["save_path"][1], subject)
    save_path = Path(root).joinpath(f"{subject}/{filename}")
    computed = save_path.exists()   # Boolean indicating if file already exists.
    if computed and not recompute:    # only recompute if requested
        print(f"Not re-computing {save_path}")
        return

    root = inputs["physio_path"][0]
    filename = inputs["physio_path"][1]
    physio_path = list(Path(root).joinpath(subject).glob(filename))

    bb = pd.read_csv(*physio_path, sep="\t", header=0).to_numpy()

    ap = (bb[:, 2] + bb[:, 3]) - (bb[:, 0] + bb[:, 1])    # anterior-posterior displacement
    ml = (bb[:, 0] + bb[:, 3]) - (bb[:, 1] + bb[:, 2])    # medio-lateral displacement

    ap_filt = butter_bandpass_filter(ap, BB_FILTER_CUTOFFS[0],
                                     BB_FILTER_CUTOFFS[1], BB_SFREQ_DECIMATED)
    ml_filt = butter_bandpass_filter(ml, BB_FILTER_CUTOFFS[0],
                                     BB_FILTER_CUTOFFS[1], BB_SFREQ_DECIMATED)

    pd.DataFrame({"ap_filt": ap_filt,
                  "ml_filt": ml_filt}).to_csv(save_path, sep="\t", header=True,
                                              index=False, float_format="%.4f")

    if not logpath:
        return
        
    sec = np.linspace(0, bb.shape[0] / BB_SFREQ_DECIMATED, bb.shape[0])
    fig0, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax0.plot(sec, ap, label="anterior-posterior displacement")
    ax0.plot(sec, ap_filt, label="filtered anterior-posterior displacement")
    ax0.set_ylabel("millimeter")
    ax0.legend(loc="upper right")
    ax1.plot(sec, ml, label="medio-lateral displacement")
    ax1.plot(sec, ml_filt, label="filtered medio-lateral displacement")
    ax1.set_ylabel("millimeter")
    ax1.set_xlabel("seconds")
    ax1.legend(loc="upper right")

    fig1, ax = plt.subplots()
    ax.plot(ap_filt, ml_filt)
    ax.set_xlabel("anterior-posterior displacenment (mm)")
    ax.set_ylabel("medio-lateral displacenment (mm)")

    fig0.savefig(logpath.with_name(logpath.name + "_fig0"), dpi=200)
    plt.close(fig0)
    fig1.savefig(logpath.with_name(logpath.name + "_fig1"), dpi=200)
    plt.close(fig1)


def get_sway_bb(subject, inputs, outputs, recompute, logpath):
    """Compute body sway metrics.

    1. anterior-posterior (back-forth) sway
    2. medio-lateral (left-right) sway
    3. sway radius
    4. total sway path
    """
    root = outputs["save_path"][0]
    filename = individualize_filename(outputs["save_path"][1], subject)
    save_path = Path(root).joinpath(f"{subject}/{filename}")
    computed = save_path.exists()   # Boolean indicating if file already exists.
    if computed and not recompute:    # only recompute if requested
        print(f"Not re-computing {save_path}")
        return

    root = inputs["physio_path"][0]
    filename = inputs["physio_path"][1]
    physio_path = list(Path(root).joinpath(subject).glob(filename))

    cop = pd.read_csv(*physio_path, sep="\t", header=0)

    n_samples = int(np.rint(BB_MOVING_WINDOW * BB_SFREQ_DECIMATED))    # width of rolling window in samples

    # Compute body sway.
    ap_sway = cop.loc[:, "ap_filt"].rolling(window=n_samples,
                                            min_periods=n_samples,
                                            center=True).std()
    ml_sway = cop.loc[:, "ml_filt"].rolling(window=n_samples,
                                            min_periods=n_samples,
                                            center=True).std()

    # Compute moving average of the center-of-pressure's radial displacement.
    cop_avg = cop.rolling(window=n_samples, min_periods=n_samples,
                          center=True).mean()
    cop_demeaned = cop - cop_avg

    radius = cop_demeaned.loc[:, "ap_filt"].combine(cop_demeaned.loc[:, "ml_filt"], cop_radius)
    radius_avg = radius.rolling(window=n_samples, min_periods=n_samples,
                                center=True).mean()

    # Compute sway path.
    ap_path = np.ediff1d(cop.loc[:, "ap_filt"], to_begin=0)**2
    ml_path = np.ediff1d(cop.loc[:, "ml_filt"], to_begin=0)**2
    total_path = np.sqrt(ap_path + ml_path)

    pd.DataFrame({"ap_sway": ap_sway,
                  "ml_sway": ml_sway,
                  "radius": radius_avg,
                  "path": total_path}).to_csv(save_path, sep="\t", header=True,
                                              index=False, float_format="%.4f")    # NaNs are saved as empty strings

    if not logpath:
        return
        
    sec = cop.index / BB_SFREQ_DECIMATED
    fig0, ax = plt.subplots()
    ax.set_title(f"moving window of {BB_MOVING_WINDOW} seconds")
    ax.set_xlabel("seconds")
    ax.set_ylabel("sway (mm)")
    ax.plot(sec, ap_sway, label="anterior-posterior sway")
    ax.plot(sec, ml_sway, label="medio-lateral sway")
    ax.legend(loc="upper right")

    fig1, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax0.set_title("COP")
    ax1.set_title(f"COP demeaned with moving window of {BB_MOVING_WINDOW} seconds")
    ax0.plot(sec, cop)
    ax1.plot(sec, cop_demeaned)

    fig2, ax = plt.subplots()
    ax.plot(sec, radius, label="radial displacement of COP")
    ax.plot(sec, radius_avg, label="radial displacement of COP averaged over"
            f" moving window of {BB_MOVING_WINDOW} seconds")
    ax.legend(loc="upper right")

    fig3, ax = plt.subplots()
    ax.set_xlabel("seconds")
    ax.set_ylabel("mm")
    ax.set_title("Sway path length")
    ax.plot(sec, total_path)

    fig0.savefig(logpath.with_name(logpath.name + "_fig0"), dpi=200)
    plt.close(fig0)
    fig1.savefig(logpath.with_name(logpath.name + "_fig1"), dpi=200)
    plt.close(fig1)
    fig2.savefig(logpath.with_name(logpath.name + "_fig2"), dpi=200)
    plt.close(fig2)
    fig3.savefig(logpath.with_name(logpath.name + "_fig3"), dpi=200)
    plt.close(fig3)
