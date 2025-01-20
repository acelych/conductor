from types import SimpleNamespace
from pathlib import Path

class ResourceManager():

    _files = SimpleNamespace(**{item.name: item for item in Path(__file__).parent.iterdir() if item.is_dir()})

    # task pattern
    _task_pattern: Path = _files.pattern
    _tasks = {item.stem: item for item in _task_pattern.iterdir()}
    
    @classmethod
    def get_legal_tasks(cls):
        return cls._tasks.keys()
    
    @classmethod
    def get_task_pattern(cls, task):
        assert task in cls._tasks.keys(), f"Unexpected task '{task}'."
        return cls._tasks[task] 