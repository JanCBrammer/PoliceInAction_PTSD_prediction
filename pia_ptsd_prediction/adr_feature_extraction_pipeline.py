# -*- coding: utf-8 -*-

from utils.io_utils import get_subjectpath, make_subjectpath
from feature_extraction import ecg
import numpy as np


"""
All directories must exist (they are not instantiated in the pipeline).
"""

# Subjects 1 trough 427.
# subjects = [f"subj{str(i).zfill(3)}" for i in np.arange(1, 428)]
subjects = ["subj001", "foo", "subj002"]


def task_ecgpreprocessing(subjects=None):
    """Step 1. Preprocess ECG"""

    rootdir = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data"
    basedir_read = r"raw"
    subdir_read = r"shootingtask\physiology"
    regex = r"*.vhdr"

    basedir_write = r"processed"
    subdir_write = r"adr\ecg"
    filename = r"ecg_clean.tsv"

    for subject in subjects:

        subjpath_read = get_subjectpath(rootdir, basedir_read, subject,
                                        subdir_read, regex, silent=False)

        # Skip subjects for whom no data were found.
        if not subjpath_read:
            print(f"Skipping {subject}.")
            continue

        subjpath_write = make_subjectpath(rootdir, basedir_write, subject,
                                          subdir_write, filename)

        ecg.preprocessing(subjpath_read, subjpath_write, show=True)


def task_ecgpeaks(subjects=None):
    """Step 2. Extract R-peaks from ECG"""

    rootdir = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data"
    basedir_read = r"processed"
    subdir_read = r"adr/ecg"
    regex = r"*ecg_clean.tsv"

    basedir_write = r"processed"
    subdir_write = r"adr\ecg"
    filename = r"ecg_peaks.tsv"

    for subject in subjects:

        subjpath_read = get_subjectpath(rootdir, basedir_read, subject,
                                        subdir_read, regex, silent=False)

        # Skip subjects for whom no data were found.
        if not subjpath_read:
            print(f"Skipping {subject}.")
            continue

        subjpath_write = make_subjectpath(rootdir, basedir_write, subject,
                                          subdir_write, filename)

        ecg.peaks(subjpath_read, subjpath_write, show=True)


def task_ecgperiod(subjects=None):
    """Step 3. Calculate instantaneous heart period"""
    pass
