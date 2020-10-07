# -*- coding: utf-8 -*-
"""
@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import shutil
from pathlib import Path
from pia_ptsd_prediction.config import DATADIR_RAW, SUBJECTS


irregular_subjects = []

old_dir = Path("...")

for subject in SUBJECTS:

    old_paths = list(old_dir.joinpath(f"{subject[4:]}").glob(f"*{subject}*"))
    old_paths = [p for p in old_paths if p.suffix in [".eeg", ".vhdr", ".vmrk"]]    # exclude .idf files
    new_dir = Path(DATADIR_RAW).joinpath(f"{subject}/socialstress/physiology")

    if old_paths:
        print(f"Copying {len(old_paths)} files for {subject}")
        for old_path in old_paths:
            shutil.copy(old_path, new_dir)
    if len(old_paths) != 9:
        print(f"found {len(old_paths)} files for {subject}")
        irregular_subjects.append(subject)
