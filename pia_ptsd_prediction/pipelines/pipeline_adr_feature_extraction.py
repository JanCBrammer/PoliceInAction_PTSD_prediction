"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>

All directories must exist (they are not instantiated in the pipeline).
"""

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from pia_ptsd_prediction.feature_extraction import ecg, bb, events
from pia_ptsd_prediction.config import DATADIR_RAW, DATADIR_PROCESSED, SUBJECTS


SUBJECTS = SUBJECTS[2:4]    # test pipeline on a subset

pipeline = [

    {"func": events.get_trial_info,
     "subjects": SUBJECTS,
     "inputs": {"log_path": [DATADIR_RAW, "shootingtask/events/PIA_w1_SH_dummy*.txt"],
                "marker_path": [DATADIR_RAW, "shootingtask/physiology/*.vmrk"]},
     "outputs": {"save_path": [DATADIR_PROCESSED, "adr/trial_info.tsv"]},
     "recompute": True},

    {"func": ecg.preprocess_ecg,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": [DATADIR_RAW, "shootingtask/physiology/*.vhdr"]},
     "outputs": {"save_path": [DATADIR_PROCESSED, "adr/ecg/ecg_clean.tsv"]},
     "recompute": False},

    {"func": ecg.get_peaks_ecg,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": [DATADIR_PROCESSED, "adr/ecg/*ecg_clean.tsv"]},
     "outputs": {"save_path": [DATADIR_PROCESSED, "adr/ecg/ecg_peaks.tsv"]},
     "recompute": False},

    {"func": ecg.get_period_ecg,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": [DATADIR_PROCESSED, "adr/ecg/*ecg_peaks.tsv"]},
     "outputs": {"save_path": [DATADIR_PROCESSED, "adr/ecg/ecg_period.tsv"]},
     "recompute": False},

    {"func": bb.preprocess_bb,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": [DATADIR_RAW, "shootingtask/physiology/*.vhdr"]},
     "outputs": {"save_path": [DATADIR_PROCESSED, "adr/balanceboard/bb_clean.tsv"]},
     "recompute": False},

    {"func": bb.get_cop_bb,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": [DATADIR_PROCESSED, "adr/balanceboard/*bb_clean.tsv"]},
     "outputs": {"save_path": [DATADIR_PROCESSED, "adr/balanceboard/bb_cop.tsv"]},
     "recompute": False},

    {"func": bb.get_sway_bb,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": [DATADIR_PROCESSED, "adr/balanceboard/*bb_cop.tsv"]},
     "outputs": {"save_path": [DATADIR_PROCESSED, "adr/balanceboard/bb_bodysway.tsv"]},
     "recompute": False}

]


def run_pipeline(pipeline, logfile):

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


if __name__ == "__main__":

    t = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    logpath = Path(DATADIR_PROCESSED).joinpath(f"log_adr_feature_extraction_{t}.pdf")
    with PdfPages(logpath) as logpdf:

        run_pipeline(pipeline, logpdf)
