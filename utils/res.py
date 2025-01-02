import os
import importlib.resources as resources

from pathlib import Path
from . import conductor

class ResourceManager():
    _task_patterns = resources.contents(resources.files(conductor).joinpath('models/pattern'))
    _tasks = {os.path.splitext(f)[-2]: Path(f) for f in _task_patterns}
    
    @classmethod
    def get_legal_tasks(cls):
        return cls._tasks.keys()
    
    @classmethod
    def get_task_pattern(cls, task):
        assert task in cls._tasks.keys(), f"Unexpected task '{task}'."
        return cls._tasks[task]