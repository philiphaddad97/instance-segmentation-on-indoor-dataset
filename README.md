# Instance Segmentation on Indoor Objects Datasets

## Description
The main objective of the project finetune and run inference on a model capable of performing instance-level segmentation on a pre-defined set of objects from coco2017 dataset [chair couch bed dining table toilet tv oven sink refrigerator]. The dataset should be trained on one or Two state-of-the-art models had been trained on the dataset Mask R-CNN and Mask Dino.

There are two directory one for each model and each one should be handled as a sperated project and run it on a different python environment.

To get the benfits of apptainer we installed this image from Nvidia which handels the Pytorch installtion with CUDA support:
```apptainer build --sandbox baseline docker://nvcr.io/nvidia/pytorch:22.12-py3```
