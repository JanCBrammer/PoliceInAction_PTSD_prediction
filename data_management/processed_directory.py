"""
@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import os
import shutil
from pathlib import Path
from pia_ptsd_prediction.config import ROOTDIR, SUBJECTS

os.chdir(ROOTDIR)


def clean_logs_dir():
    if not Path.cwd().stem == ROOTDIR:
        return

    if not Path("processed/logs").is_dir():
        return

    os.chdir("processed")
    shutil.rmtree("logs")
    os.mkdir("logs")


def clean_processed_dir():
    if not Path.cwd().stem == ROOTDIR:
        return

    if not Path("processed").is_dir():
        return

    print("Removing existing directory 'processed'")
    shutil.rmtree("processed")
    build_processed_dir()


def build_processed_dir():
    if not Path.cwd().stem == ROOTDIR:
        return

    os.mkdir("processed")
    os.chdir("processed")

    os.mkdir("logs")

    # Define hierarchy of sub-directories in subject directories.
    level1 = ["shootingtask", "socialstress"]
    level2 = [["ecg", "balanceboard"], ["ppg"]]

    # Create sub-directory for each subject, containing the nested levels.
    for subject in SUBJECTS:
        print(f"Setting up directory for subject {subject}")
        for l1, l2 in zip(level1, level2):
            for i in l2:
                os.makedirs(os.path.join(subject, l1, i))
