"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""
from pathlib import Path
from .io_utils import (search_subjectpath, individualize_subjectfilename,
                       individualize_subjectpath)


def loop_subjects(taskfunc, subjects, rootdir, readpath, writepath, logfile):
    """Execute a task for a set of subjects.

    Parameters
    ----------
    taskfunc : function
        A funtion from one of the feature_extraction sub-modules.
    subjects : list
        Each element is a string of the format "subj[0-9][0-9][0-9]".
    rootdir : string
        Directory containing the data directories "raw" and a "processed".
    readpath : dict
        Path to search the input data for taskfunc. Can contain a glob pattern
        in the file name.
    writepath : dict
        Path for writing the output of taskfunc.
    logfile: PdfPages
        An open PdfPages object.
    """
    print(f"Applying {taskfunc.__name__} to {len(subjects)} subjects")

    for subject in subjects:
        
        subj_readpath = individualize_subjectpath(readpath, subject)
        subj_readpath = Path(rootdir).joinpath(subj_readpath)
        subj_readpath_match = search_subjectpath(subj_readpath, silent=False)

        # Skip subjects for whom no data were found.
        if not subj_readpath_match:
            print(f"Skipping {subject}.")
            continue

        subj_writepath = individualize_subjectpath(writepath, subject)
        subj_writepath = individualize_subjectfilename(subj_writepath, subject)
        subj_writepath = Path(rootdir).joinpath(subj_writepath)

        taskfunc(subj_readpath_match, subj_writepath, logfile)
                