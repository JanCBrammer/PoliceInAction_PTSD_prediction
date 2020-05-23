"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""
from pathlib import Path


def search_subjectpath(searchpath, silent=True):
    """Search for a file.

    Parameters
    ----------
    searchpath : Path object
        The full search path with a glob pattern in the filename.
    silent : bool, optional
        Whether or not to print search status, by default True.

    Returns
    -------
    Path object, False
        Return the path to the matching file, only if exactly one file was
        found. Otherwise returns False.
    """
    filepattern = searchpath.name    # string
    searchdir = searchpath.parent    # Path object
    paths = list(searchdir.glob(filepattern))
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


def individualize_subjectfilename(path, subject):
    """Prepend a subject ID to the filename.

    Parameters
    ----------
    path : Path object
        The path in which to individualize the filename.
    subject : string
        The subject ID.

    Returns
    -------
    Path object
        The path with invidiualized filename.
    """
    filename = path.name
    subj_filename = f"{subject}_{filename}"
    subj_path = path.parent.joinpath(subj_filename)

    return subj_path


def individualize_subjectpath(pathstring, subject):
    """Replaces the generic subject placeholder with an individual subject ID.

    Parameters
    ----------
    pathstring : str
        Path containing a subdirectory called "subj" which serves as generic
        placeholder for a subject ID.
    subject : str
        The subject ID.

    Returns
    -------
    Path object
        The individualized subject paths.
    """
    components = list(Path(pathstring).parts)
    subj_idx = components.index("subj")    # throws ValueError if "subj" is not in components
    components[subj_idx] = subject
    subj_path = Path(*components)
    
    return subj_path
