"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

from pia_ptsd_prediction.utils.pipeline_utils import Task
from pia_ptsd_prediction.preprocessing import heart, bb, events
from pia_ptsd_prediction.config import (
    DATADIR_RAW,
    DATADIR_PROCESSED,
    SUBJECTS,
    ECG_CHANNELS_ADR,
    ECG_SFREQ_ORIGINAL_ADR,
    ECG_SFREQ_DECIMATED,
    HEART_PERIOD_SFREQ,
    HR_MIN,
    HR_MAX,
    RUNNING_MEDIAN_KERNEL_SIZE,
    MAD_THRESHOLD_MULTIPLIER,
    BB_CHANNELS,
    BB_SFREQ_ORIGINAL,
    BB_SFREQ_DECIMATED,
    BB_MIN_EMPTY,
    BB_BOARDLENGTH,
    BB_FILTER_CUTOFFS,
    BB_MOVING_WINDOW,
)
from biopeaks.heart import ecg_peaks

SUBJECTS = SUBJECTS[:2]


pipeline = [
    Task(
        events.get_trial_info,
        {
            "sfreq_original": ECG_SFREQ_ORIGINAL_ADR,
        },
        inputs={
            "log_path": f"{DATADIR_RAW}/<SUBJECT>/shootingtask/events/PIA_w1_SH_dummy*.txt",
            "marker_path": f"{DATADIR_RAW}/<SUBJECT>/shootingtask/physiology/*.vmrk",
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/<SUBJECT>_trial_info.tsv"
        },
        subjects=SUBJECTS,
    ),
    Task(
        heart.preprocess_heart_signal,
        {
            "chans": ECG_CHANNELS_ADR,
            "sfreq_original": ECG_SFREQ_ORIGINAL_ADR,
            "sfreq_decimated": ECG_SFREQ_DECIMATED,
            "invert_signal": True,
        },
        inputs={
            "physio_path": f"{DATADIR_RAW}/<SUBJECT>/shootingtask/physiology/*.vhdr"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/ecg/<SUBJECT>_ecg_clean.tsv"
        },
        subjects=SUBJECTS,
        logdir=f"{DATADIR_PROCESSED}/logs",
    ),
    Task(
        heart.get_heart_peaks,
        {"sfreq_decimated": ECG_SFREQ_DECIMATED, "detector": ecg_peaks},
        inputs={
            "physio_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/ecg/*_ecg_clean.tsv"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/ecg/<SUBJECT>_ecg_peaks.tsv"
        },
        subjects=SUBJECTS,
        logdir=f"{DATADIR_PROCESSED}/logs",
    ),
    Task(
        heart.get_heart_period,
        {
            "sfreq_period": HEART_PERIOD_SFREQ,
            "sfreq_decimated": ECG_SFREQ_DECIMATED,
        },
        inputs={
            "physio_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/ecg/*_ecg_peaks.tsv"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/ecg/<SUBJECT>_ecg_period.tsv"
        },
        subjects=SUBJECTS,
        logdir=f"{DATADIR_PROCESSED}/logs",
    ),
    Task(
        heart.remove_outliers_heart_period,
        {
            "sfreq_period": HEART_PERIOD_SFREQ,
            "hr_min": HR_MIN,
            "hr_max": HR_MAX,
            "running_median_kernel_size": RUNNING_MEDIAN_KERNEL_SIZE,
            "mad_threshold_multiplier": MAD_THRESHOLD_MULTIPLIER,
        },
        inputs={
            "physio_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/ecg/*_ecg_period.tsv"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/ecg/<SUBJECT>_ecg_period_clean.tsv"
        },
        subjects=SUBJECTS,
        logdir=f"{DATADIR_PROCESSED}/logs",
    ),
    Task(
        bb.preprocess_bb,
        {
            "chans": BB_CHANNELS,
            "sfreq_original": BB_SFREQ_ORIGINAL,
            "sfreq_decimated": BB_SFREQ_DECIMATED,
            "min_empty": BB_MIN_EMPTY,
            "boardlength": BB_BOARDLENGTH,
        },
        inputs={
            "physio_path": f"{DATADIR_RAW}/<SUBJECT>/shootingtask/physiology/*.vhdr"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/balanceboard/<SUBJECT>_bb_clean.tsv"
        },
        subjects=SUBJECTS,
        logdir=f"{DATADIR_PROCESSED}/logs",
    ),
    Task(
        bb.get_cop_bb,
        {
            "sfreq_decimated": BB_SFREQ_DECIMATED,
            "filter_cutoffs": BB_FILTER_CUTOFFS,
        },
        inputs={
            "physio_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/balanceboard/*bb_clean.tsv"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/balanceboard/<SUBJECT>_bb_cop.tsv"
        },
        subjects=SUBJECTS,
        logdir=f"{DATADIR_PROCESSED}/logs",
    ),
    Task(
        bb.get_sway_bb,
        {
            "sfreq_decimated": BB_SFREQ_DECIMATED,
            "moving_window": BB_MOVING_WINDOW,
        },
        inputs={
            "physio_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/balanceboard/*bb_cop.tsv"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/adr/balanceboard/<SUBJECT>_bb_bodysway.tsv"
        },
        subjects=SUBJECTS,
        logdir=f"{DATADIR_PROCESSED}/logs",
    ),
]
