# CONDUCTOR

Conductor is a trainer for Pytorch. Its purpose is to facilitate experiments on computer vision deep learning models by build yaml file, just like what [Ultralytics](https://github.com/ultralytics/ultralytics) doing. However, Ultralytics is made for industrial circle, and so the coding structure of Ultralytics is extremly unfriendly for learning and mastering. For example, how do the yaml descriptions of the layer transform into nn.Module instance is written by if-else statements, making it so hard to modifying module or using custom module on the latest YOLOs unless editing the original code.

In this different project, it's easy to track the training process and using custom module.

//TODO