"""
author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, AnyStr
from itertools import product
from pathlib import Path
from datetime import datetime


@dataclass
class Task:
    """Run a task."""

    task: Callable
    kwargs: Dict
    inputs: Dict = field(default_factory=dict)
    outputs: Dict = field(default_factory=dict)
    subjects: List = field(default_factory=lambda: ["None"])
    conditions: List = field(default_factory=lambda: ["None"])
    logdir: AnyStr = ""
    recompute: bool = True

    def __post_init__(self):
        self.name = self.task.__name__
        self.n_subjects = len(self.subjects)
        self.n_conditions = len(self.conditions)
        self.n_runs = self.n_subjects * self.n_conditions
        self.message = f"\nRunning {self.name} on {self.n_subjects} subjects in {self.n_conditions} conditions ({self.n_runs} runs)..."

    def run(self):
        print(self.message)
        print("-" * len(self.message))
        for i, (subject, condition) in enumerate(
            product(self.subjects, self.conditions)
        ):

            logdir = Path(self.logdir).joinpath(self.name)
            try:
                logdir.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                pass
            now = datetime.now()
            logpath = logdir.joinpath(
                f"{subject}_{condition}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
            )
            print(
                f"{now.strftime('%H:%M:%S')}: Running subject {subject} in condition {condition} (run {i + 1}/{self.n_runs})..."
            )
            self.task(
                subject,
                condition,
                self.inputs,
                self.outputs,
                self.recompute,
                logpath,
                **self.kwargs,
            )
