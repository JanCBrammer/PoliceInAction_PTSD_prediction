# -*- coding: utf-8 -*-

import os
import glob


def get_subjectpath(rootdir, subjdir, subdir, regex):
    """Search "rootdir\subjdir\subdir" for the file containing the regular
    expession "regex" and return the full path if exaclty one file is found.
    """

    searchpath = os.path.join(rootdir, subjdir, subdir, regex)
    path = glob.glob(searchpath)
    n_files = len(path)
    if n_files == 0:
        print(f"Could not find requested file for {subdir}.")
    elif n_files > 1:
        print(f"Found {n_files} files for {subjdir}.")
    elif n_files == 1:
        return path[0]


def make_subjectpath(rootdir, subjdir, subdir, filename):
    """Construct and return the path "rootdir\subjdir\subdir\filename".
    """

    path = os.path.join(rootdir, subjdir, subdir, filename)
    return path
