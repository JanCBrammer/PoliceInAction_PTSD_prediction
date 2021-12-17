#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from pia_ptsd_prediction.config import DATADIR_RAW, DATADIR_PROCESSED, SUBJECTS
from pia_ptsd_prediction.preprocessing_adr.pipeline import pipeline as adr_preprocessing_pipeline


def run(pipeline, logfile):

    for task in pipeline:

        taskfunc = task["func"]
        subjects = task["subjects"]

        for subject in subjects:

            print(f"Running {taskfunc.__name__} on {subject}.")

            taskfunc(subject,
                     task["inputs"],
                     task["outputs"],
                     task["recompute"],
                     logfile)

def main():
    """Command line entry point.
    Assumes that data directories are set up as specified in config.py.
    """
    selected_subjects = [2] # run on random subset (print seed)
    print("Running data processing pipeline.")
    t = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    logpath = Path(DATADIR_PROCESSED).joinpath(f"log_pia_ptsd_prediction_{t}.pdf")
    with PdfPages(logpath) as logpdf:
        run(adr_preprocessing_pipeline([SUBJECTS[i] for i in selected_subjects], DATADIR_RAW, DATADIR_PROCESSED), logpdf)

if __name__ == "__main__":
    main()





