import yaml
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

class ClassifyDataset(Dataset):
    def __init__(self):
        super().__init__()

class UnifiedDataLoader:
    import pyarrow.parquet as pq
    def __init__(self, yaml_path: str, task: str):
        pass
            
    def parse_yaml(self, yaml_path: str, task: str):
        
        with open(yaml_path, 'r') as f:
            data_desc: dict = yaml.safe_load(f)
        assert data_desc.get('task') == task, f"task of dataset '{yaml_path}' ({data_desc.get('task')}) does not match required task ({task})"
            
        dataset_path = Path(data_desc.get('path'))
            
        if task == "classify":
            pass