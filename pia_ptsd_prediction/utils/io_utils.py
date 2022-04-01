"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""


def individualize_path(path: str, subject: str, condition: str) -> str:
    """"""
    path = path.replace("<SUBJECT>", subject)
    path = path.replace("<CONDITION>", condition)

    return path
