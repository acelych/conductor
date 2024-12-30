import os
import importlib.resources as resources

from pathlib import Path
from .. import conductor

class ResourceManager():
    _mission_patterns = resources.contents(resources.files(conductor).joinpath('models/pattern'))
    _missions = {os.path.splitext(f)[-2]: Path(f) for f in _mission_patterns}
    
    @classmethod
    def get_legal_missions(cls):
        return cls._missions.keys()
    
    @classmethod
    def get_mission_pattern(cls, mission):
        assert mission in cls._missions.keys(), f"Unexpected mission '{mission}'."
        return cls._missions[mission]