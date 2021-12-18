"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import pandas as pd
from pathlib import Path
from mne.annotations import read_annotations
from pia_ptsd_prediction.utils.io_utils import individualize_filename
from pia_ptsd_prediction.config import ECG_SFREQ_ORIGINAL


def get_trial_info(subject, inputs, outputs, recompute, logpath):
    """Extract trial information from Presentation log files and BrainVision
    marker files.

    1. trial type {high threat, low threat}
    2. start baseline (seconds)
    3. cue (seconds)
    4. stimulus (seconds)
    """
    root = outputs["save_path"][0]
    filename = individualize_filename(outputs["save_path"][1], subject)
    save_path = Path(root).joinpath(f"{subject}/{filename}")
    computed = save_path.exists()   # boolean indicating if file already exists
    if computed and not recompute:    # only recompute if requested
        print(f"Not re-computing {save_path}")
        return

    root = inputs["log_path"][0]
    filename = inputs["log_path"][1]
    log_path = list(Path(root).joinpath(subject).glob(filename))

    root = inputs["marker_path"][0]
    filename = inputs["marker_path"][1]
    marker_path = list(Path(root).joinpath(subject).glob(filename))

    df_log = pd.read_csv(*log_path, sep="\t", usecols=["CSI", "shock"])
    markers = read_annotations(*marker_path, ECG_SFREQ_ORIGINAL)
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
