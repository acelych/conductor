from typing import Tuple

import torch
from torch import nn, Tensor
from torch import profiler

from .model import ModelManager
from .data import DataLoaderManager
from .utils import ConfigManager, ArtifactManager, MetricsManager, LogInterface, get_model_assessment

class Profiler:
    def __init__(self, cm: ConfigManager, am: ArtifactManager, log: LogInterface):
        self.cm = cm
        self.am = am
        self.log = log
        self.model_mng = ModelManager(self.cm)
        self.data_mng = DataLoaderManager(self.cm, self.log)
        self.met_mng = MetricsManager()
        self.device = torch.device(cm.device)

    def profile(self):
        self.log.info("initializing profiling")

        model = self.model_mng.build_model().to(self.cm.device)
        data = torch.randn(1, 3, *self.cm.imgsz).to(self.cm.device)  # Test random rgb tensor

        self.log.info(model.info(), fn=True)
        self.log.info(f"start inference through random tensor shape: {data.shape}...", fn=True)

        model.eval()

        # --------------------------
        # Step 1: PyTorch Profiling
        # --------------------------
        sches = [1, 1, 3]
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(wait=sches[0], warmup=sches[1], active=sches[2]),
            on_trace_ready=profiler.tensorboard_trace_handler(self.am.profiler_dir.absolute(), worker_name="profiler"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(sum(sches)):
                model(data)
                prof.step()

        self.log.info("profiling completed. results saved to TensorBoard.")
        self.log.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10), fn=True)

        # --------------------------
        # Step 2: Export to ONNX
        # --------------------------
        onnx_path = self.am.onnx_model_path  # 例如: "artifacts/model.onnx"
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

        torch.onnx.export(
            model,
            data,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes  # 支持不同 batch size
        )
        self.log.info(f"model exported to ONNX at {onnx_path}")

        # --------------------------
        # Step 3: ONNX Runtime Profiling
        # --------------------------
        import onnxruntime as ort
        from pathlib import Path

        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = True
        session = ort.InferenceSession(onnx_path, sess_options)

        input_name = session.get_inputs()[0].name
        input_data = data.cpu().numpy()  # ONNX Runtime 需要 numpy 输入

        self.log.info(f"start ONNX inference with input shape: {input_data.shape}", fn=True)

        for _ in range(sum(sches)):
            session.run(None, {input_name: input_data})

        # ONNX profiling 文件会自动保存到默认目录（当前工作目录）
        # 我们可以复制或移动它到 self.am.profiler_dir.absolute()
        profile_file = session.end_profiling()
        import shutil
        target_profile_file = self.am.profiler_dir / f"{Path(profile_file).name}"
        shutil.move(profile_file, target_profile_file)
        self.log.info(f"ONNX profiling completed. results saved to {target_profile_file}")