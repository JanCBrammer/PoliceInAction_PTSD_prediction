import shutil
import numpy as np
from pathlib import Path


# Define subjects 1 trough 427.
subjects = [f"subj{str(i).zfill(3)}" for i in np.arange(1, 428)]

irregular_subjects = []

old_dir = Path("...")

for subject in subjects:

    # Find subject in old directory.
    old_paths = list(old_dir.glob(f"*{subject}*"))
    new_dir = Path(f"...{subject}/shootingtask/physiology")

    if old_paths:
        print(f"Copying {len(old_paths)} files for {subject}")
        for old_path in old_paths:
            shutil.copy(old_path, new_dir)
    if len(old_paths) != 3:
        print(f"found {len(old_paths)} files for {subject}")
        irregular_subjects.append(subject)
