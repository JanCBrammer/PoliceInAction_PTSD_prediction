"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

from pathlib import Path


def individualize_filename(path, subject):
    """Prepend a subject ID to the filename.

    Parameters
    ----------
    path : str
        The path in which to individualize the filename.
    subject : str
        The subject ID.

    Returns
    -------
    Path object
        The path with individualized filename.
    """
    path = Path(path)
    filename = path.name
    subj_filename = f"{subject}_{filename}"
    subj_path = path.parent.joinpath(subj_filename)

    return subj_path
