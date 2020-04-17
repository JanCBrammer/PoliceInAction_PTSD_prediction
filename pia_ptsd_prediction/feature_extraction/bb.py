# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:51:14 2020

@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from config import (bb_channels, bb_sfreq_original, bb_sfreq_decimated,
                    bb_min_empty)
from utils.analysis_utils import decimate_signal, consecutive_samples


def preprocessing(readpath, writepath, show=False):

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
        def empty(x): return x < bb_mins[i]
        begs, ends, n = consecutive_samples(bb_decimated[i, :],
                                            empty,
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

    # Sum sensors along time axis.
    bb_channelsum = np.sum(bb_decimated, axis=0)

    if show:
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        sec = np.linspace(0, bb_decimated.shape[1] / bb_sfreq_decimated,
                          bb_decimated.shape[1])
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for channel in range(bb_decimated.shape[0]):
            color = colors[channel]
            ax0.plot(sec, bb_decimated[channel, :], c=color,
                     label=f"channel {channel}")
            ax0.axvspan(xmin=sec[bb_minsconsecutive[channel, 0]],
                        xmax=sec[bb_minsconsecutive[channel, 1]],
                        ymin=bb_decimated[channel].min(),
                        ymax=bb_decimated[channel].max(),
                        color=color, alpha=.2, label="empty")

        ax0.hlines(y=bb_empty, xmin=0, xmax=sec[-1],
                   colors=colors[:bb_mins.size], linestyles="dotted",
                   label="empty")
        ax0.legend(loc="upper right")

        ax1.plot(sec, bb_channelsum)
        ax1.set_xlabel("seconds")
