# CONDUCTOR

Conductor is a trainer for Pytorch. Its purpose is to facilitate experiments on computer vision deep learning models by building yaml file, just like what [Ultralytics](https://github.com/ultralytics/ultralytics) doing. However, Ultralytics is made for industrial circle, and so the coding structure of Ultralytics is extremly unfriendly for learning and mastering. For example, how do the yaml descriptions of the layer transform into nn.Module instance is written by if-else statements, making it so hard to modifying module or using custom module on the latest YOLOs unless editing the original code.

In this different project, it's easy to track the training process and using custom module.

## Training Example

Create a yaml instruction file like this:

```yaml
# basics
model_yaml_path: path/to/model.yaml
data_yaml_path: path/to/dataset.yaml
command: train  # command from cli, indicate the Conductor what to do
task: classify  # task of the model

# training details
epochs: 1
batch_size: 64
device: cuda
world: '[0]'
criterion: CrossEntropyLoss  # below: within default
optimizer: AdamW
scheduler: CosineAnnealingLR
learn_rate: 1e-3
momentum: 0.9
decay: 1e-4
best_metrics: top1_acc

# logging & ckpt details
output_dir: path/to/output_dir
ckpt: null
```

Assuming this .yaml file named `instruct.yaml`.
Then, prepare your `model.yaml` like:

```yaml
nc: &nc 100
task: classify
backbone:
  - [-1, 1, ConvNormAct, [[16, 3, 2], BatchNorm2d, Hardswish]]  # [[c2, k, ...], BatchNorm2d, Hardswish]
  - [-1, 1, InvertedResidual, [16, 16, 3, 2, 1, true, ReLU]]  # -> C1
  - [-1, 1, InvertedResidual, [24, 72, 3, 2, 1, false, ReLU]]  # -> C2
  - [-1, 1, InvertedResidual, [40, 96, 5, 2, 1, true, Hardswish]]  # -> C3
  - [-1, 1, InvertedResidual, [96, 288, 5, 2, 1, true, Hardswish]]  # -> C4
  - [-1, 1, InvertedResidual, [96, 576, 5, 1, 1, true, Hardswish]]
  - [-1, 1, ConvNormAct, [[576, 1, 1], BatchNorm2d, Hardswish]]
head:
  - [-1, 1, Classifier, [*nc, 1024, 0.3]]
```
or direct get models from `./utils/resources/example/*`. For `data.yaml`, check `./temp/cifar100.yaml` as example. 
To use CIFAR100 dataset, you have to download them next to the `cifar100.yaml` file.

Finally, write a python script at `../` of current path:

```python
from conductor import Conductor

tar = Conductor('/path/to/instruct.yaml')
tar.run()
```

Run this script to train!