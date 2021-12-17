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

    # Find subject in old directory. Since numbers in subject IDs are nor zero padded,
    # only inlude matches that are followed by an underscore or a period.
    old_paths = list(old_dir.glob(f"*{subject}[_.]*"))
    subject_zeropad = f"subj{subject[4:].zfill(3)}"
    new_dir = Path(ROOTDIR).joinpath(f"raw/{subject_zeropad}/shootingtask/events")

    if old_paths:
        print(f"Copying {len(old_paths)} files for {subject}")
        for old_path in old_paths:
            shutil.copy(old_path, new_dir)
    if len(old_paths) != 1:
        print(f"Found {len(old_paths)} files for {subject}")
        irregular_subjects.append(subject)
