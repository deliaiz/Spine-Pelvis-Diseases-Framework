# Spine-Pelvis-Diseases
A gait analysis diagnostic Framework for spine and pelvis diseases in localization and pathological characterization.
The article is currently in the submission period, and the model code will be released after the article is accepted.
# Software Requirements
## Hardware requirements
The package development version is tested on Linux operating systems.
Linux: Ubuntu 16.04
window: window 10
CUDA/cudnn:10.1
## Python Dependencies
> - Python
> - PyTorch-cuda
> - torchvision
> - opencv
> - numpy
> - json
> - os
...
## Prepare dataset
1. Prepare an txt file containing video names(*.mp4) and the Label to complete video sequences as the data input for training.
   For example:
   P01_01.mp4 0 \\
   P03_1_01.mp4 3
# Training and Evaluation example
Training and evaluation are on a single GPU.
## Train
python train.py
## Evaluation
python test.py
