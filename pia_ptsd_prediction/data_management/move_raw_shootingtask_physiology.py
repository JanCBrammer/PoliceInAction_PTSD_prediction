# -*- coding: utf-8 -*-

import shutil
import glob
import os
import numpy as np

# Define subjects 1 trough 427.
subjects = [f"subj{str(i).zfill(3)}" for i in np.arange(1, 428)]

irregular_subjects = []

old_dir = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data\raw_prelim\wave1"

for subject in subjects:

    # Find subject in old directory.
    old_paths = glob.glob(os.path.join(old_dir, f"*{subject}*"))
    new_dir = rf"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data\raw\{subject}\shootingtask\physiology"

    if old_paths:
        for old_path in old_paths:
            shutil.move(old_path, new_dir)
    if len(old_paths) != 3:
        print(f"found {len(old_paths)} files for {subject}")
        irregular_subjects.append(subject)
