"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""
from .io_utils import search_subjectpath, make_subjectpath


def loop_subjects(taskfunc, subjects, rootdir, readcomponents, writecomponents,
                  logfile):
    """Execute a task for a set of subjects.

    Parameters
    ----------
    taskfunc : function
        A funtion from one of the feature_extraction sub-modules.
    subjects : list
        Each element is a string of the format "subj[0-9][0-9][0-9]".
    rootdir : string
        Directory containing the data directories "raw" and a "processed".
    readcomponents : dict
        Dictionary containing the arguments for io_utils.search_subjectpath.
    writecomponents : dict
        Dictionary containing the arguments for io_utils.make_subjectpath.
    logfile: PdfPages
        An open PdfPages object.
    """
    basedir_read = readcomponents["basedir"]
    subdir_read = readcomponents["subdir"]
    regex = readcomponents["regex"]

    basedir_write = writecomponents["basedir"]
    subdir_write = writecomponents["subdir"]
    filename = writecomponents["filename"]

    print(f"Applying {taskfunc.__name__} to {len(subjects)} subjects")

    for subject in subjects:

        subjpath_read = search_subjectpath(rootdir, basedir_read, subject,
                                           subdir_read, regex, silent=False)

        # Skip subjects for whom no data were found.
        if not subjpath_read:
            print(f"Skipping {subject}.")
            continue

        subjpath_write = make_subjectpath(rootdir, basedir_write, subject,
                                          subdir_write, filename)

        fig = taskfunc(subjpath_read, subjpath_write)
        
        logfile.savefig(fig)
