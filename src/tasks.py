from __future__ import annotations

from typing import List, Tuple

# Each task is a list of examples:
# ((input_1, input_2), target_output)

Task = List[Tuple[Tuple[float, float], float]]


def get_task_a() -> Task:
    """Task A:
    [1, 0] -> 1
    [0, 1] -> 0
    """
    return [
        ((1.0, 0.0), 1.0),
        ((0.0, 1.0), 0.0),
    ]


def get_task_b() -> Task:
    """Task B:
    [1, 0] -> 0
    [0, 1] -> 1
    """
    return [
        ((1.0, 0.0), 0.0),
        ((0.0, 1.0), 1.0),
    ]