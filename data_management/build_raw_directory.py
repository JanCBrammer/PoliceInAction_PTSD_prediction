# -*- coding: utf-8 -*-
"""
@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import os
from pia_ptsd_prediction.config import ROOTDIR, SUBJECTS


# Change to top-level directory.
os.chdir(ROOTDIR)

# Create directory for raw data
os.mkdir("raw")
os.chdir("raw")

# Define tasks.
tasks = ["shootingtask", "socialstress", "questionnaires"]

# Define acquisitions.
acquisitions = [["physiology", "events"], [],
                ["caps", "pcl"]]

# Create sub-directory for each subject, containing the tasks and associated
# acquisitions.
for subject in SUBJECTS:
    for task, acquisition in zip(tasks, acquisitions):
        for a in acquisition:
            os.makedirs(os.path.join(subject, task, a))
