import shutil
import numpy as np
from pathlib import Path


# Define subjects 1 trough 427.
subjects = [f"subj{str(i)}" for i in np.arange(1, 428)]

irregular_subjects = []

old_dir = Path("C:/Users/JohnDoe/surfdrive/Beta/wave1_sh_logs")

for subject in subjects:
    
    # Find subject in old directory. Since numbers in subject IDs are nor zero padded,
    # only inlude matches that are followed by an underscore or a period.
    old_paths = list(old_dir.glob(f"*{subject}[_.]*"))
    subject_zeropad = f"subj{subject[4:].zfill(3)}"
    new_dir = Path(f"C:/Users/JohnDoe/surfdrive/Beta/PoliceInAction_PTSD_Prediction/data/raw/{subject_zeropad}/shootingtask/events")

    if old_paths:
        print(f"Copying {len(old_paths)} files for {subject}")
        for old_path in old_paths:
            shutil.copy(old_path, new_dir)
    if len(old_paths) != 1:
        print(f"Found {len(old_paths)} files for {subject}")
        irregular_subjects.append(subject)
