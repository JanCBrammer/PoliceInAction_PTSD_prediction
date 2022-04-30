"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""
from pia_ptsd_prediction.utils.pipeline_utils import Task
from pia_ptsd_prediction.preprocessing import heart
from pia_ptsd_prediction.config import (
    DATADIR_RAW,
    DATADIR_PROCESSED,
    SUBJECTS,
    CONDITIONS_TONIC,
    PPG_CHANNELS_TONIC,
    PPG_SFREQ_ORIGINAL_TONIC,
    PPG_SFREQ_DECIMATED,
    HEART_PERIOD_SFREQ,
    HR_MIN,
    HR_MAX,
    RUNNING_MEDIAN_KERNEL_SIZE,
    MAD_THRESHOLD_MULTIPLIER,
)
from biopeaks.heart import ppg_peaks

SUBJECTS = SUBJECTS[:2]


pipeline = [
    Task(
        heart.preprocess_heart_signal,
        {
            "chans": PPG_CHANNELS_TONIC,
            "sfreq_original": PPG_SFREQ_ORIGINAL_TONIC,
            "sfreq_decimated": PPG_SFREQ_DECIMATED,
            "invert_signal": False,
        },
        inputs={
            "physio_path": f"{DATADIR_RAW}/<SUBJECT>/socialstress/physiology/*<CONDITION>*.vhdr"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/socialstress/ppg/<SUBJECT>_<CONDITION>_ppg_clean.tsv"
        },
        subjects=SUBJECTS,
        conditions=CONDITIONS_TONIC,
        logdir=f"{DATADIR_PROCESSED}/logs",
    ),
    Task(
        heart.get_heart_peaks,
        {"sfreq_decimated": PPG_SFREQ_DECIMATED, "detector": ppg_peaks},
        inputs={
            "physio_path": f"{DATADIR_PROCESSED}/<SUBJECT>/socialstress/ppg/*<CONDITION>_ppg_clean.tsv"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/socialstress/ppg/<SUBJECT>_<CONDITION>_ppg_peaks.tsv"
        },
        subjects=SUBJECTS,
        conditions=CONDITIONS_TONIC,
        logdir=f"{DATADIR_PROCESSED}/logs",
    ),
    Task(
        heart.get_heart_period,
        {
            "sfreq_period": HEART_PERIOD_SFREQ,
            "sfreq_decimated": PPG_SFREQ_DECIMATED,
        },
        inputs={
            "physio_path": f"{DATADIR_PROCESSED}/<SUBJECT>/socialstress/ppg/*<CONDITION>_ppg_peaks.tsv"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/socialstress/ppg/<SUBJECT>_<CONDITION>_ppg_period.tsv"
        },
        subjects=SUBJECTS,
        conditions=CONDITIONS_TONIC,
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
            "physio_path": f"{DATADIR_PROCESSED}/<SUBJECT>/socialstress/ppg/*<CONDITION>_ppg_period.tsv"
        },
        outputs={
            "save_path": f"{DATADIR_PROCESSED}/<SUBJECT>/socialstress/ppg/<SUBJECT>_<CONDITION>_ppg_period_clean.tsv"
        },
        subjects=SUBJECTS,
        conditions=CONDITIONS_TONIC,
        logdir=f"{DATADIR_PROCESSED}/logs",
    ),
]
