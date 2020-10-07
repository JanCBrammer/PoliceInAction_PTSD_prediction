# -*- coding: utf-8 -*-
"""
@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import shutil
from pathlib import Path
from pia_ptsd_prediction.config import ROOTDIR, SUBJECTS


irregular_subjects = []

old_dir = Path("...")

for subject in SUBJECTS:

    # Find subject in old directory.
    old_paths = list(old_dir.glob(f"*{subject}*"))
    new_dir = Path(ROOTDIR).joinpath(f"raw/{subject}/shootingtask/physiology")

    if old_paths:
        print(f"Copying {len(old_paths)} files for {subject}")
        for old_path in old_paths:
            shutil.copy(old_path, new_dir)
    if len(old_paths) != 3:
        print(f"found {len(old_paths)} files for {subject}")
        irregular_subjects.append(subject)
