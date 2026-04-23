import functools
from typing import Callable

task_registry = {}


def task(f: Callable):
    name = f.__name__

    if name in task_registry:
        raise ValueError(
            f"there is already a function with name '{name}' " "in the registry."
        )

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    task_registry[name] = wrapper

    return wrapper
