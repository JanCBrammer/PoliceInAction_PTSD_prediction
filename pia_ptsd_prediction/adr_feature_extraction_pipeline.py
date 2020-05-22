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
 
{"taskfunc": ecg.preprocess,
"subjects": ["subj001", "foo"],
"rootdir": rootdir,
"readcomponents": {"basedir": r"raw",
                "subdir": r"shootingtask\physiology",
                "regex": r"*.vhdr"},
"writecomponents": {"basedir": r"processed",
                    "subdir": r"adr\ecg",
                    "filename": r"ecg_clean.tsv"}
},

{"taskfunc": ecg.get_peaks,
"subjects": ["subj001", "foo"],
"rootdir": rootdir,
"readcomponents": {"basedir": r"processed",
                "subdir": r"adr\ecg",
                "regex": r"*ecg_clean.tsv"},
"writecomponents": {"basedir": r"processed",
                    "subdir": r"adr\ecg",
                    "filename": r"ecg_peaks.tsv"}
},

{"taskfunc": ecg.get_period,
"subjects": ["bar", "subj001", "foo"],
"rootdir": rootdir,
"readcomponents": {"basedir": r"processed",
                "subdir": r"adr\ecg",
                "regex": r"*ecg_peaks.tsv"},
"writecomponents": {"basedir": r"processed",
                    "subdir": r"adr\ecg",
                    "filename": r"ecg_period.tsv"}
},

# {"taskfunc": bb.preprocess,
# "subjects": ["foo", "subj001"],
# "rootdir": rootdir,
# "readcomponents": {"basedir": r"raw",
#                 "subdir": r"shootingtask\physiology",
#                 "regex": r"*.vhdr"},
# "writecomponents": {"basedir": r"processed",
#                     "subdir": r"adr\balanceboard",
#                     "filename": r"bb_clean.tsv"},
# "show": True},

# {"taskfunc": bb.get_bodysway,
# "subjects": [],
# "rootdir": rootdir,
# "readcomponents": {"basedir": r"processed",
#                 "subdir": r"adr\balanceboard",
#                 "regex": r"*bb_clean.tsv"},
# "writecomponents": {"basedir": r"processed",
#                     "subdir": r"adr\balanceboard",
#                     "filename": r"bb_bodysway.tsv"},
# "show": True},
]

logpath = Path("C:/Users/JohnDoe/surfdrive/Beta/PoliceInAction_PTSD_Prediction/data/tasklog.pdf")
with PdfPages(logpath) as logpdf:

    for task in taskmatrix:
        loop_subjects(task["taskfunc"],
                      task["subjects"],
                      task["rootdir"],
                      task["readcomponents"],
                      task["writecomponents"],
                      logpdf)
        