"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""
from pia_ptsd_prediction.preprocessing.pipeline_adr import (
    pipeline as adr_preprocessing,
)


def run_pipeline(pipeline):
    for task in pipeline:
        task.run()


def main():
    """Command line entry point.
    Assumes that data directories are set up as specified in config.py.
    """
    run_pipeline(adr_preprocessing)


if __name__ == "__main__":
    main()
