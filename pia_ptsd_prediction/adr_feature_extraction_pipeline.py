# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:51:14 2020

@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

from utils.io_utils import search_subjectpath, make_subjectpath
from feature_extraction import ecg, bb
import numpy as np


"""
All directories must exist (they are not instantiated in the pipeline).
"""

def task_ecgpreprocessing(subjects=None):
    """Step 1. Preprocess ECG"""

    rootdir = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data"
    basedir_read = r"raw"
    subdir_read = r"shootingtask\physiology"
    regex = r"*.vhdr"

    basedir_write = r"processed"
    subdir_write = r"adr\ecg"
    filename = r"ecg_clean.tsv"

    print("Starting to pre-process ECG.")

    for subject in subjects:

        subjpath_read = search_subjectpath(rootdir, basedir_read, subject,
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
    subdir_read = r"adr\ecg"
    regex = r"*ecg_clean.tsv"

    basedir_write = r"processed"
    subdir_write = r"adr\ecg"
    filename = r"ecg_peaks.tsv"

    print("Starting to extract R-peaks.")

    for subject in subjects:

        subjpath_read = search_subjectpath(rootdir, basedir_read, subject,
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

    rootdir = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data"
    basedir_read = r"processed"
    subdir_read = r"adr\ecg"
    regex = r"*ecg_peaks.tsv"

    basedir_write = r"processed"
    subdir_write = r"adr\ecg"
    filename = r"ecg_period.tsv"

    print("Starting to extract heart period.")

    for subject in subjects:

        subjpath_read = search_subjectpath(rootdir, basedir_read, subject,
                                           subdir_read, regex, silent=False)

        # Skip subjects for whom no data were found.
        if not subjpath_read:
            print(f"Skipping {subject}.")
            continue

        subjpath_write = make_subjectpath(rootdir, basedir_write, subject,
                                          subdir_write, filename)

        ecg.period(subjpath_read, subjpath_write, show=True)


def task_bbpreprocessing(subjects=None):
    """Step X. Preprocess balance-board"""

    rootdir = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data"
    basedir_read = r"raw"
    subdir_read = r"shootingtask\physiology"
    regex = r"*.vhdr"

    basedir_write = r"processed"
    subdir_write = r"adr\balanceboard"
    filename = r"bb_clean.tsv"

    print("Starting to pre-process balance-board.")

    for subject in subjects:

        subjpath_read = search_subjectpath(rootdir, basedir_read, subject,
                                           subdir_read, regex, silent=False)

        # Skip subjects for whom no data were found.
        if not subjpath_read:
            print(f"Skipping {subject}.")
            continue

        subjpath_write = make_subjectpath(rootdir, basedir_write, subject,
                                          subdir_write, filename)

        bb.preprocessing(subjpath_read, subjpath_write, show=True)



if __name__ == "__main__":

    # Subjects 1 trough 427.
    # subjects = [f"subj{str(i).zfill(3)}" for i in np.arange(1, 428)]
    subjects = ["subj001", "foo", "subj002"]
    # Functions and their order of execution can be specified in a list.
    tasks = [task_ecgpreprocessing,
              task_ecgpeaks,
              task_ecgperiod,
              task_bbpreprocessing]
    # tasks = [task_bbpreprocessing]

    for task in tasks:
        task(subjects)
