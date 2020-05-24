"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>

All directories must exist (they are not instantiated in the pipeline).
"""
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from pia_ptsd_prediction.utils.pipeline_utils import loop_subjects
from pia_ptsd_prediction.feature_extraction import ecg, bb


rootdir = r"C:\Users\JohnDoe\surfdrive\Beta\PoliceInAction_PTSD_Prediction\data"

taskmatrix = [
 
{"taskfunc": ecg.preprocess_ecg,
"subjects": ["subj001", "foo"],
"rootdir": rootdir,
"readpath": r"raw\subj\shootingtask\physiology\*.vhdr",    # generic "subj" placeholder in paths is automatically replaced with subject ID during processing
"writepath": r"processed\subj\adr\ecg\ecg_clean.tsv"},

{"taskfunc": ecg.get_peaks_ecg,
"subjects": ["foo", "bar", "subj001"],
"rootdir": rootdir,
"readpath": r"processed\subj\adr\ecg\*ecg_clean.tsv",
"writepath": r"processed\subj\adr\ecg\ecg_peaks.tsv"},

{"taskfunc": ecg.get_period_ecg,
"subjects": ["bar", "subj001", "foo"],
"rootdir": rootdir,
"readpath": r"processed\subj\adr\ecg\*ecg_peaks.tsv",
"writepath": r"processed\subj\adr\ecg\ecg_period.tsv"},

{"taskfunc": bb.preprocess_bb,
"subjects": ["subj001"],
"rootdir": rootdir,
"readpath": r"raw\subj\shootingtask\physiology\*.vhdr",
"writepath": r"processed\subj\adr\balanceboard\bb_clean.tsv"},

{"taskfunc": bb.get_cop_bb,
"subjects": ["subj001"],
"rootdir": rootdir,
"readpath": r"processed\subj\adr\balanceboard\*bb_clean.tsv",
"writepath": r"processed\subj\adr\balanceboard\bb_bodysway.tsv"}

]

logpath = Path("C:/Users/JohnDoe/surfdrive/Beta/PoliceInAction_PTSD_Prediction/data/tasklog.pdf")
with PdfPages(logpath) as logpdf:

    for task in taskmatrix:
        loop_subjects(task["taskfunc"],
                      task["subjects"],
                      task["rootdir"],
                      task["readpath"],
                      task["writepath"],
                      logpdf)
        