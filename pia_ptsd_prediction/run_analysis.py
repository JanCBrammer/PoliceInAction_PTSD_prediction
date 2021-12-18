"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

from datetime import datetime
from pathlib import Path
from pia_ptsd_prediction.config import DATADIR_RAW, DATADIR_PROCESSED, SUBJECTS
from pia_ptsd_prediction.preprocessing_adr.pipeline import pipeline as adr_preprocessing_pipeline


def run(pipeline):

    for task in pipeline:

        taskfunc = task["func"]
        subjects = task["subjects"]
        taskname = taskfunc.__name__
        n_subjects = len(subjects)

        for i, subject in enumerate(subjects):

            logdir = Path(DATADIR_PROCESSED).joinpath(f"logs/{taskname}")
            logdir.mkdir(parents=True, exist_ok=True)
            now = datetime.now()
            logpath = logdir.joinpath(f"{subject}_{now.strftime('%Y-%m-%d_%H-%M-%S')}")
            print(f"{now.strftime('%H:%M:%S')}: Running {taskname} on {subject} ({i + 1}/{n_subjects}).")
            taskfunc(subject,
                     task["inputs"],
                     task["outputs"],
                     task["recompute"],
                     logpath)

def main():
    """Command line entry point.
    Assumes that data directories are set up as specified in config.py.
    """
    selected_subjects = range(len(SUBJECTS)) # TODO: run on small, random subset during development (print seed)
    print("Running data processing pipeline.")
    run(adr_preprocessing_pipeline([SUBJECTS[i] for i in selected_subjects], DATADIR_RAW, DATADIR_PROCESSED))


if __name__ == "__main__":
    main()





