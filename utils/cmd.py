from torch import nn, optim, cuda

class CommandDetails:
    def __init__(
        self, 
        model_yaml_path: str, 
        data_yaml_path: str,
        command: str,  # train, predict
        task: str,
        device = "cuda",  # cpu, cuda, [...]
        world = None,
        criterion: nn.Module = nn.CrossEntropyLoss,
        optimizer: optim.Optimizer = optim.AdamW,
        learn_rate: int = 0.001,
        batch_size: int = 16,
        epochs: int = 300
        ):
        self.model_yaml_path: str = model_yaml_path
        self.data_yaml_path: str = data_yaml_path
        self.command = command
        self.task = task
        
        device_count = cuda.device_count()
        assert device == "cuda" or device == "cpu", f"expect 'device' to be 'cuda' or 'cpu', got {device}"
        if device == "cuda" and device_count == 0:
            # TODO: warning, could not find any cuda device
            self.device = "cpu"
        else:
            self.device = device
            
        if isinstance(world, int):
            assert world <= device_count, f"expect using {device_count} gpu device(s) at most, got {world}"
            self.world = list(range(0, world))
        elif isinstance(world, list):
            assert all(n < device_count for n in world), f"expect device index less than {device_count}, got {world}"
        elif device == "cuda" and world is None:
            world = [0]
        self.world: list = None if device == "cpu" else world
        
        self.criterion = criterion
        self.optimizer: optim.Optimizer = optimizer
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
def is_using_ddp(cd: CommandDetails):
    return cd.world is not None and len(cd.world) > 1 