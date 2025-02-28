from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from .model import ModelManager
from .data import DataLoaderManager
from .utils import ConfigManager, ArtifactManager, MetricsManager, Calculate, Recorder, LogInterface, Plot, get_model_assessment
from .modules._utils import TensorCollector


class Tester:
    def __init__(self, cm: ConfigManager, am: ArtifactManager, log: LogInterface):
        self.cm = cm
        self.am = am
        self.log = log
        self.model_mng = ModelManager(self.cm)
        self.data_mng = DataLoaderManager(self.cm, self.log)
        self.met_mng = MetricsManager()
        self.device = torch.device(cm.device)
        
    def test(self):
        self.log.info("initializing testing")
        test_dataleader = self.data_mng.get_dataloader("test")
        self.model = self.model_mng.build_model().to(self.cm.device)
        self.criterion: nn.Module = self.cm.criterion()
        self.recorder: Recorder = Recorder(self.model_mng.model_desc.get("nc"))
        
        # basic assessment
        self.log.info(self.model.info(), fn=True)
        model_assessment, _ = get_model_assessment(self.model, self.cm.imgsz, lambda:self.model_mng.build_model().to(self.cm.device))
        self.log.info(model_assessment, fn=True)
        
        # testing
        best_epoch = self.load_state(self.am.ckpt)
        metrics = self.met_mng.get_metrics_holder(self.cm.task, best_epoch)
        test_report, precision, _ = self.test_epoch(test_dataleader, metrics, best_epoch)
        metrics.dummy_fill()
        self.log.metrics(vars(metrics), save=False)
        self.log.info(test_report)
        
        # latency
        self.latency()
        
        # sampling
        self.sampling("train")
        self.sampling("test")
        
        # focusing
        worst_category = precision.argmin()
        self.focusing("train", worst_category.item())
        self.focusing("test", worst_category.item())
    
    def test_epoch(self, dataloader: DataLoader, metrics: MetricsManager.Metrics, best_epoch: int = -1) -> Tuple[list, Tensor, Tensor]:
        self.model.eval()
        self.recorder.clear()

        self.log.bar_init(len(dataloader), f"(test) on epoch {best_epoch}", fn=True)

        # validate test
        with torch.no_grad():
            for x, label in dataloader:
                x, label = self.move_batch(x, label)
                # TensorCollector.enable()
                output = self.model(x)
                # self.am.vis_matrices(TensorCollector.get("expand"), "expand", 1)
                # self.am.vis_matrices(TensorCollector.get("input"), "input", 1)
                # self.am.vis_matrices(TensorCollector.get("norm"), "norm", 1)
                loss: Tensor = self.criterion(output, label)
                self.recorder(output, label, loss)
                self.log.bar_update()
        self.log.bar_close()
        self.recorder.converge()
        metrics.record_val(self.recorder)
        
        # test report
        precision = Calculate.precision(self.recorder.get_conf_mat())
        recall = Calculate.recall(self.recorder.get_conf_mat())
        head = [f"{'':>3}{'name':>20}{'precision':>12}{'recall':>12}"]
        typesinfo = [f"{i:>3}{name:>20}{precision[i].item():10.4f}{recall[i].item():10.4f}" for i, name in self.data_mng.names.items()]
        typesinfo.append(f"{'':>3}{'total':>20}{precision.mean().item():10.4f}{recall.mean().item():10.4f}")
        
        return head + typesinfo, precision, recall
    
    def latency(self, repeats: int = 500):
        import time
        with torch.no_grad():
            rand_tensor = torch.randn((1, 3, *self.cm.imgsz), device=torch.cuda.current_device())
            self.log.info("...testing latency...")
            start_time = time.time()
            for _ in range(repeats):
                self.model(rand_tensor)
            latency = (time.time() - start_time) / repeats
        self.log.info(f"model latency: {latency * 1e3}ms")
    
    def sampling(self, stage: str, amount: int = 25):
        self.log.info(f"...sampling {stage} dataset...")
        rand_indices = torch.randperm(len(self.data_mng.get_dataset(stage=stage)))[:amount]
        inputs = [self.data_mng.get_with_original(stage=stage, index=i.item()) for i in rand_indices]
        samples = [inp[0] for inp in inputs]
        
        x = torch.stack(tuple(inp[1] for inp in inputs)).to(device=self.device)
        out: Tensor = self.model(x)
        _, pred = out.max(1)
        
        label = [f"label: {self.data_mng.names[inp[2]]}({inp[2]})" for inp in inputs]
        pred = [f"pred: {self.data_mng.names[p.item()]}({p.item()})" for p in pred]
        self.am.plot_samples(stage, samples, label, pred)
        
    def focusing(self, stage: str, category_idx: int, amount: int = 25):
        self.log.info(f"...sampling {self.data_mng.names[category_idx]}({category_idx}) category from {stage} dataset...")
        nc = self.model_mng.model_desc.get("nc")
        assert 0 < category_idx < nc, f"expect category index between 0 & {nc}, got {category_idx}"
        
        inputs = []
        for i in range(len(self.data_mng.get_dataset(stage=stage))):
            inp = self.data_mng.get_with_original(stage=stage, index=i)
            if inp[2] == category_idx:
                inputs.append(inp)
            if len(inputs) >= amount:
                break
        samples = [inp[0] for inp in inputs]
        
        x = torch.stack(tuple(inp[1] for inp in inputs)).to(device=self.device)
        out: Tensor = self.model(x)
        _, pred = out.max(1)
        
        label = [f"label: {self.data_mng.names[inp[2]]}({inp[2]})" for inp in inputs]
        pred = [f"pred: {self.data_mng.names[p.item()]}({p.item()})" for p in pred]
        self.am.plot_samples(stage, samples, label, pred, category_name=self.data_mng.names[category_idx])

    def load_state(self, ckpt: dict) -> int:
        self.model.load_state_dict(ckpt.get('model_state_dict'))
        return ckpt.get('epoch')
        
    def move_batch(self, *args: Tensor) -> tuple:
        return (arg.to(self.device) for arg in args)