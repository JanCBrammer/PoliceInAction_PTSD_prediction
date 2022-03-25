"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

def individualize_path(path: str, subject: str, expand_name=False) -> str:
    """"""
    path_elements = path.split("/")
    path_elements.insert(2, subject)
    if expand_name:
        path_elements[-1] = f"{subject}_{path_elements[-1]}"

    return "/".join(path_elements)
