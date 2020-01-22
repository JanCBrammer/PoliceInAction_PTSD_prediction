# -*- coding: utf-8 -*-

import os
import numpy as np


# Change to top-level directory.
basedir = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data"
os.chdir(basedir)

# Create directory for raw data
os.mkdir("raw")
os.chdir("raw")

# Define subjects 1 trough 427.
subjects = [f"subj{str(i).zfill(3)}" for i in np.arange(1, 428)]

# Define tasks.
tasks = ["shootingtask", "socialstress", "questionnaires"]

# Define acquisitions.
acquisitions = [["physiology", "events"], ["physiology", "events"],
                ["caps", "pcl"]]

# Create sub-directory for each subject, containing the tasks and asociated
# acquisitions.
for subject in subjects:
    for task, acquisition in zip(tasks, acquisitions):
        for a in acquisition:
            os.makedirs(os.path.join(subject, task, a))
