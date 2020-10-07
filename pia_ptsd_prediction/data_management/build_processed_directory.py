# -*- coding: utf-8 -*-
"""
@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import os
from pia_ptsd_prediction.config import ROOTDIR, SUBJECTS


# Change to top-level directory.
os.chdir(ROOTDIR)

# Create directory for processed data.
os.mkdir("processed")
os.chdir("processed")

# Define hierarchy of sub-directories in subject directory.
level1 = ["adr", "tonic", "pcl", "caps"]
level2 = [["ecg", "balanceboard"],
          ["ecg"],
          [],
          []]

# Create sub-directory for each subject, containing the nested levels.
for subject in SUBJECTS:
    print(f"Setting up directory for subject {subject}")
    for l1, l2 in zip(level1, level2):
        if l2:
            for i in l2:
                os.makedirs(os.path.join(subject, l1, i))
        else:
            os.makedirs(os.path.join(subject, l1))
