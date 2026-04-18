# CONDUCTOR

**Conductor** is a highly modular, configuration-driven computer vision (CV) deep learning training framework and experiment scheduler (Trainer) built on PyTorch.

Its purpose is to facilitate experiments on deep learning models by building YAML files, similar to the philosophy of [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO). However, while Ultralytics is heavily tailored for industrial deployment (making its internal structure extremely unfriendly for learning and mastering—e.g., hard-coded `if-else` parsing for module instantiation), Conductor is designed for **academic research, custom model development, and structural ablation studies**. 

It is incredibly easy to track the training process, modify network topologies, and integrate custom operators without ever touching the underlying Python `forward` functions.

## ✨ Key Features
* **Configuration-Driven**: All hyperparameters, data paths, and model topologies are centrally managed via YAML files.
* **Modular Assembly (NAS-Friendly)**: Build complex networks (including Skip Connections and Feature Fusions) directly through YAML lists `[from, repeats, module, args]`.
* **Native DDP & NAS**: Seamlessly enable Multi-GPU DistributedDataParallel (DDP) training or Differentiable Architecture Search (NAS) with just a single line of configuration.
* **Hardcore CUDA Operators**: Built-in custom C++/CUDA extensions (like `AdaptiveCrossHadamard` and In-place `DySoft`) to shatter PyTorch's VRAM bottlenecks.

---

## 📖 Documentation & Deep Dive

For a comprehensive guide on usage, DDP/NAS configuration, and underlying code design, please refer to our official architecture documentation:
* 🇬🇧 [English Architecture & User Manual](docs/ARCHITECTURE_EN.md)
* 🇨🇳 [中文架构深度解析与使用说明](docs/ARCHITECTURE.md)

---

## 🚀 Quick Start

### 1. Environment Setup

Please ensure that `nvcc` is available within your runtime environment, as Conductor relies on compiled CUDA kernels for specific high-performance modules.

**Option A: Automated Setup (Linux/macOS)**
We provide a setup script to automatically install dependencies and compile the CUDA extensions:
```bash
bash install.sh
```

**Option B: Manual Setup (Windows or Custom Envs)**
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Compile the CUDA kernels required by the framework:
   ```bash
   cd ./modules/cuda_modules
   python setup.py build_ext --inplace
   ```

### 2. Training Example

Conductor uses an instruction YAML file to orchestrate the entire training lifecycle. Create a file named `instruct.yaml` (or copy from our templates in `utils/resources/instruct_example/train.yaml`):

```yaml
# basics
model_yaml_path: utils/resources/model_example/mobilenetv4_s.yaml
data_yaml_path: utils/resources/dataset_example/cifar100.yaml
command: train  # command from cli, indicate the Conductor what to do
task: classify  # task of the model

# training details
epochs: 50
batch_size: 128
device: cuda
world: '[0]'             # Write '[0, 1]' to seamlessly enable DDP dual-card training
criterion: CrossEntropyLoss
optimizer: AdamW
scheduler: CosineAnnealingLR
learn_rate: 1e-3
momentum: 0.9
decay: 1e-4
best_metrics: top1_acc

# logging & ckpt details
output_dir: ./outputs
ckpt: null
```

*Note: You can easily swap `model_yaml_path` with a built-in reflection string like `mobilevit,{'num_classes':100,'mode':'xx_small'}` to train official models instantly.*

### 3. Run the Conductor

Finally, write a simple python script in your working directory:

```python
from conductor import Conductor

tar = Conductor('instruct.yaml')
tar.run()
```

Run this script to train! All your experimental artifacts (weights, metrics CSV, and diagnostic plots) will be neatly saved in the `outputs/task_0/` directory.
