"""
@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

from pathlib import Path
from collections import Counter


def find_invalid_subjects(datadir, subjects, show_details=False):
    print("Validating input files.")
    invalid_subjects = set()

    for subject in subjects:
        subject_dir = Path(datadir).joinpath(subject)

        event_files = Counter(
            [
                file.suffix
                for file in subject_dir.joinpath("shootingtask/events").iterdir()
            ]
        )
        if event_files != Counter({".txt": 1}):
            if show_details:
                print(
                    f"{subject}: incorrect type and/or number of event files: {event_files}."
                )
            invalid_subjects.add(subject)

        physio_files_adr = Counter(
            [
                file.suffix
                for file in subject_dir.joinpath("shootingtask/physiology").iterdir()
            ]
        )
        if physio_files_adr != Counter({".eeg": 1, ".vhdr": 1, ".vmrk": 1}):
            if show_details:
                print(
                    f"{subject} incorrect type(s) and/or number of physio files adr {physio_files_adr}."
                )
            invalid_subjects.add(subject)

        physio_files_tonic = Counter(
            [
                file.suffix
                for file in subject_dir.joinpath("socialstress/physiology").iterdir()
            ]
        )
        if physio_files_tonic != Counter({".eeg": 3, ".vhdr": 3, ".vmrk": 3}):
            if show_details:
                print(
                    f"{subject} incorrect type(s) and/or number of physio files tonic {physio_files_tonic}."
                )
            invalid_subjects.add(subject)

    return invalid_subjects


# General ######################################################################
ROOTDIR = "data"
DATADIR_RAW = "data/raw"
DATADIR_PROCESSED = "data/processed"
subjects = {f"subj{str(i).zfill(3)}" for i in range(1, 428)}  # subjects 1 through 427
subjects_invalid = set()
if __name__ == "__main__":
    with open(
        Path(DATADIR_RAW).joinpath("invalid_subjects.txt"), "w"
    ) as f:  # overwrites file if it exists
        f.writelines(s + "\n" for s in find_invalid_subjects(DATADIR_RAW, subjects))
with open(
    Path(DATADIR_RAW).joinpath("invalid_subjects.txt"), "r"
) as f:  # let it fail if file isn't there
    subjects_invalid.update(f.read().splitlines())

# Manually add subjects who failed assertions or visual inspections during processing.
_subjects_invalid = {
    "headerfile_mismatch": [
        "subj095",
        "subj116",
        "subj117",
        "subj118",
        "subj136",
        "subj280",
    ],  # failed assertion
    "no_empty_bb_section": ["subj147"],  # failed assertion
    "unusable_ecg": [
        "subj091",
        "subj119",
        "subj235",
        "subj243",
    ],  # failed visual inspection
    "unequal_n_trials_bva_presentation": ["subj291"],
}  # failed assertion
subjects_invalid.update([s for l in _subjects_invalid.values() for s in l])
SUBJECTS = sorted(list(subjects - subjects_invalid))
CONDITIONS_TONIC = ["RS1", "RS2", "SECPT"]

# HEART ########################################################################
ECG_CHANNELS_ADR = ["ECG"]
PPG_CHANNELS_TONIC = ["HR"]

ECG_SFREQ_ORIGINAL_ADR = 2500  # Hz
PPG_SFREQ_ORIGINAL_TONIC = 5000  # Hz

ECG_SFREQ_DECIMATED = 500  # Hz
PPG_SFREQ_DECIMATED = 32  # Hz
# The sampling frequency of the heart period must not be too low in order to
# not loose too much temporal precision during event-related, and HRV analyses.
# See `demo_influcence_sfreq_event_timing` and https://doi.org/10.1088/1361-6579/aa5efa.
HEART_PERIOD_SFREQ = 32  # Hz
HR_MIN = 35  # bpm
HR_MAX = 185  # bpm
MAD_THRESHOLD_MULTIPLIER = 4
RUNNING_MEDIAN_KERNEL_SIZE = 120  # sec

# Balance-board  ###############################################################
BB_CHANNELS = ["BB1", "BB2", "BB3", "BB4"]
BB_SFREQ_ORIGINAL = 2500
BB_SFREQ_DECIMATED = 32  # Hz
BB_FILTER_CUTOFFS = [0.01, 10]  # lowcut (highpass) and highcut (lowpass) in Hz
BB_MIN_EMPTY = 10  # seconds
BB_BOARDLENGTH = 425  # TODO: verify board length!
BB_MOVING_WINDOW = 1  # seconds
