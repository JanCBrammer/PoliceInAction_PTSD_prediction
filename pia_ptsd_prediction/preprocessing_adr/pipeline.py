"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

from pia_ptsd_prediction.preprocessing_adr import ecg, bb, events


def pipeline(SUBJECTS, DATADIR_RAW, DATADIR_PROCESSED):

    return [

    {"func": events.get_trial_info,
     "subjects": SUBJECTS,
     "inputs": {"log_path": f"{DATADIR_RAW}/shootingtask/events/PIA_w1_SH_dummy*.txt",
                "marker_path": f"{DATADIR_RAW}/shootingtask/physiology/*.vmrk"},
     "outputs": {"save_path": f"{DATADIR_PROCESSED}/adr/trial_info.tsv"},
     "recompute": True},

    {"func": ecg.preprocess_ecg,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": f"{DATADIR_RAW}/shootingtask/physiology/*.vhdr"},
     "outputs": {"save_path": f"{DATADIR_PROCESSED}/adr/ecg/ecg_clean.tsv"},
     "recompute": True},

    {"func": ecg.get_peaks_ecg,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": f"{DATADIR_PROCESSED}/adr/ecg/*ecg_clean.tsv"},
     "outputs": {"save_path": f"{DATADIR_PROCESSED}/adr/ecg/ecg_peaks.tsv"},
     "recompute": True},

    {"func": ecg.get_period_ecg,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": f"{DATADIR_PROCESSED}/adr/ecg/*ecg_peaks.tsv"},
     "outputs": {"save_path": f"{DATADIR_PROCESSED}/adr/ecg/ecg_period.tsv"},
     "recompute": True},

    {"func": ecg.remove_outliers_period_ecg,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": f"{DATADIR_PROCESSED}/adr/ecg/*ecg_period.tsv"},
     "outputs": {"save_path": f"{DATADIR_PROCESSED}/adr/ecg/ecg_period_clean.tsv"},
     "recompute": True},

    {"func": bb.preprocess_bb,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": f"{DATADIR_RAW}/shootingtask/physiology/*.vhdr"},
     "outputs": {"save_path": f"{DATADIR_PROCESSED}/adr/balanceboard/bb_clean.tsv"},
     "recompute": True},

    {"func": bb.get_cop_bb,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": f"{DATADIR_PROCESSED}/adr/balanceboard/*bb_clean.tsv"},
     "outputs": {"save_path": f"{DATADIR_PROCESSED}/adr/balanceboard/bb_cop.tsv"},
     "recompute": True},

    {"func": bb.get_sway_bb,
     "subjects": SUBJECTS,
     "inputs": {"physio_path": f"{DATADIR_PROCESSED}/adr/balanceboard/*bb_cop.tsv"},
     "outputs": {"save_path": f"{DATADIR_PROCESSED}/adr/balanceboard/bb_bodysway.tsv"},
     "recompute": True}

    ]
