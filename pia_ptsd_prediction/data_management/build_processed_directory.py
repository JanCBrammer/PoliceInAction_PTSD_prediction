# -*- coding: utf-8 -*-

import os
import numpy as np


# Change to top-level directory.
basedir = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data"
os.chdir(basedir)

# Create directory for processed data.
os.mkdir("processed")
os.chdir("processed")

# Define subjects 1 trough 427.
subjects = [f"subj{str(i).zfill(3)}" for i in np.arange(1, 428)]

# Define hierarchy of sub-directories in subject directory.
level1 = ["adr", "tonic", "pcl", "caps"]
level2 = [["ecg", "balanceboard"],
          ["ecg"],
          [],
          []]

# Create sub-directory for each subject, containing the nested levels.
for subject in subjects:
    for l1, l2 in zip(level1, level2):
        if l2:
            for i in l2:
                os.makedirs(os.path.join(subject, l1, i))
        else:
            os.makedirs(os.path.join(subject, l1))
