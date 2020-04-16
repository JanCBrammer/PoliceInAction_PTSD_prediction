# -*- coding: utf-8 -*-

from pathlib import Path


def search_subjectpath(root, base, subj, sub, regex, silent=True):
    """Search "root\base\subj\sub" for the file containing "regex" and return
    the full path if exactly one file is found. Otherwise return False.
    """

    searchpath = Path(root).joinpath(base, subj, sub)
    paths = list(searchpath.glob(regex))
    n_files = len(paths)
    path = False

    if n_files == 0:
        status = f"Could not find a file matching {searchpath}."
    elif n_files > 1:
        status = f"Found {n_files} files matching {searchpath}."
    elif n_files == 1:
        status = f"Found 1 file matching {searchpath}."
        path = paths[0]

    if not silent:
        print(status)

    return path


def make_subjectpath(root, base, subj, sub, filename):
    """Construct and return the path "root\base\subj\sub\filename".
    """

    filename = f"{subj}_{filename}"
    writepath = Path(root).joinpath(base, subj, sub, filename)

    return writepath
