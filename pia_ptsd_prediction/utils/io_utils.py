# -*- coding: utf-8 -*-

import os
import glob


def get_subjectpath(root, base, subj, sub, regex, silent=True):
    """Search "root\base\subj\sub" for the file containing the regular
    expession "regex" and return the full path if exaclty one file is found.
    """

    searchpath = os.path.join(root, base, subj, sub, regex)
    paths = glob.glob(searchpath)
    n_files = len(paths)
    path = []

    if n_files == 0:
        if not silent:
            print(f"Could not find a file matching {searchpath}.")
    elif n_files > 1:
        if not silent:
            print(f"Found {n_files} files matching {searchpath}.")
    elif n_files == 1:
        if not silent:
            print(f"Found 1 file matching {searchpath}.")
        path = paths[0]

    return path


def make_subjectpath(root, base, subj, sub, filename):
    """Construct and return the path "root\base\subj\sub\filename".
    """

    filename = f"{subj}_{filename}"
    writepath = os.path.join(root, base, subj, sub, filename)
    return writepath
