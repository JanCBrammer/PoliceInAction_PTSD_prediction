"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from biopeaks.filters import butter_bandpass_filter
from ..config import (bb_channels, bb_sfreq_original, bb_sfreq_decimated,
                      bb_min_empty, bb_boardlength, bb_filter_cutoffs)
from ..utils.analysis_utils import decimate_signal, consecutive_samples


def preprocess_bb(readpath, writepath, logfile=None):

    raw = mne.io.read_raw_brainvision(readpath, preload=False, verbose="error")
    bb = raw.get_data(picks=bb_channels)
    sfreq = raw.info["sfreq"]

    # Raise AssertionError for unexpected sampling frequencies.
    assert sfreq == bb_sfreq_original, (f"Sampling frequency {sfreq} doesn't"
                                        " match expected sampling frequency"
                                        f" {bb_sfreq_original}.")

    # Decimate the four balance board channels from original sampling rate to
    # 20 HZ. Note that MNE's raw.apply_function() cannot be used since it
    # requires the preservation of the original sampling frequency.
    decimation_factor = int(np.floor(sfreq / bb_sfreq_decimated))
    bb_decimated = decimate_signal(bb, decimation_factor)

    # Assuming that the participant has been off the board at some time,
    # calculate the empty board value: For each channel, take the mean of a
    # consecutive chunk of data of at least 10 seconds duration that is below
    # the minimum value + std.
    bb_mins = bb_decimated.min(axis=1) + bb_decimated.std(axis=1)
    bb_minsconsecutive = np.zeros((bb_mins.size, 2)).astype(int)
    bb_empty = np.zeros(bb_mins.size)
    min_duration = int(np.ceil(bb_sfreq_decimated * bb_min_empty))

    for i in range(bb_mins.size):

        begs, ends, n = consecutive_samples(bb_decimated[i, :],
                                            lambda x: x < bb_mins[i],
                                            min_duration)
        # Raise if no chunk of min_duration has been found.
        assert begs.size > 0, (f"Did not find {bb_min_empty} consecutive"
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
    bb_mm = bb_mm * (bb_boardlength / 2)    # express in mm
        
    pd.DataFrame(bb_mm).T.to_csv(writepath, sep="\t",
                                 header=["BB1", "BB2", "BB3", "BB4"],    # transpose to change from channels as rows to channels as columns (preserves ordering of channels)
                                 index=False, float_format="%.4f")

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    sec = np.linspace(0, bb_decimated.shape[1] / bb_sfreq_decimated,
                        bb_decimated.shape[1])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    channames = ["PR", "PL", "AL", "AR"]

    for chan in range(bb_decimated.shape[0]):
        color = colors[chan]
        channame = channames[chan]
        ax0.plot(sec, bb_decimated[chan, :], c=color,
                    label=f"{channame}")
        ax0.axvspan(xmin=sec[bb_minsconsecutive[chan, 0]],
                    xmax=sec[bb_minsconsecutive[chan, 1]],
                    ymin=bb_decimated[chan].min(),
                    ymax=bb_decimated[chan].max(),
                    color=color, alpha=.2, label="empty")

    ax0.hlines(y=bb_empty, xmin=0, xmax=sec[-1],
                colors=colors[:bb_mins.size], linestyles="dotted",
                label="empty")
    ax0.legend(loc="upper right")

    ax1.plot(sec, bb_chansum)
    ax1.axhline(y=bb_subjweight, c="r",
                label="subject weight minus board weight")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("seconds")

    for chan in range(bb_mm.shape[0]):
        color = colors[chan]
        channame = channames[chan]
        ax2.plot(sec, bb_mm[chan, :], c=color,
                    label=f"{channame}")
    ax2.legend(loc="upper right")
    ax2.set_xlabel("seconds")
    ax2.set_ylabel("millimeters")
    
    logfile.savefig(fig)
        

def get_cop_bb(readpath, writepath, logfile=None):
    
    bb = pd.read_csv(readpath, sep="\t", header=0).to_numpy()
    
    ap = (bb[:, 2] + bb[:, 3]) - (bb[:, 0] + bb[:, 1])    # anterior-posterior displacement
    ml = (bb[:, 0] + bb[:, 3]) - (bb[:, 1] + bb[:, 2])    # medio-lateral displacement
    
    ap_filt = butter_bandpass_filter(ap, bb_filter_cutoffs[0],
                                     bb_filter_cutoffs[1], bb_sfreq_decimated)
    ml_filt = butter_bandpass_filter(ml, bb_filter_cutoffs[0],
                                     bb_filter_cutoffs[1], bb_sfreq_decimated)

    pd.DataFrame({"ap_filt": ap_filt,
                  "ml_filt": ml_filt}).to_csv(writepath, sep="\t", header=True,
                                              index=False, float_format="%.4f")
    
    sec = np.linspace(0, bb.shape[0] / bb_sfreq_decimated, bb.shape[0])
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

    logfile.savefig(fig0)
    logfile.savefig(fig1)


def get_bodysway_bb(readpath, writepath, logfile=None):
    pass
    # sway_ml =
    # sway_ap =
    # sway_path =





