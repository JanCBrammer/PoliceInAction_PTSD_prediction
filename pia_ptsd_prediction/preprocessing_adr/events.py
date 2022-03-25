"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import pandas as pd
from pathlib import Path
from mne.annotations import read_annotations
from pia_ptsd_prediction.utils.io_utils import individualize_path
from pia_ptsd_prediction.config import ECG_SFREQ_ORIGINAL


def get_trial_info(subject, inputs, outputs, recompute, logpath):
    """Extract trial information from Presentation log files and BrainVision
    marker files.

    1. trial type {high threat, low threat}
    2. start baseline (seconds)
    3. cue (seconds)
    4. stimulus (seconds)
    """
    save_path = Path(individualize_path(outputs["save_path"], subject, expand_name=True))
    if save_path.exists() and not recompute:    # only recompute if requested
        print(f"Not re-computing {save_path}")
        return
    log_path = next(Path(".").glob(individualize_path(inputs["log_path"], subject)))
    marker_path = next(Path(".").glob(individualize_path(inputs["marker_path"], subject)))

    df_log = pd.read_csv(log_path, sep="\t", usecols=["CSI", "shock"])
    markers = read_annotations(marker_path, ECG_SFREQ_ORIGINAL)
    cues = markers.onset[markers.description == "Stimulus/S  2"]
    assert len(cues) == df_log.shape[0], ("Unequal number of trials between"
                                          " BrainVision and Presentation files"
                                          f" for participant {subject}.")

    trial_types = df_log["shock"]
    stimuli = cues + df_log["CSI"] / 1000
    baselines = cues - 1    # baseline starts 1 second before cue

    df = pd.DataFrame({"threat": trial_types, "baseline": baselines,
                       "cue": cues, "stimulus": stimuli})
    df = df[(df["stimulus"] - df["cue"]) >= 6]    # exclude trials with anticipation windows shorter than 6 seconds
    df.to_csv(save_path, sep="\t", header=True, index=False, float_format="%.4f")
