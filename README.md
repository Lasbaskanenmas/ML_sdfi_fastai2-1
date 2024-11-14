#SDFI_ml
Machine learning code used and maintained by sdfi to perform semantic segmentation on multichannel images. 

## Installation
```sh
mamba env create --file environment.yml
mamba activate ML_sdfi
pip install -e .
```
Check if you have cuda support
```sh
mamba activate ML_sdfi
python
import torch
>>> torch.cuda.is_available()
True
```
If torch.cuda.is_available() returns False you might have to install CUDA suported torch again acording to:
https://pytorch.org/get-started/locally/ 
example:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 





## train.py
configs/example_configs/train_example_dataset.ini holds parameters for training a model on a dataset
### Use it like this
```sh
python train.py --config configs/example_configs/train_example_dataset.ini
```
## infer.py
configs/example_configs/infer_example_dataset.ini holds parameters for doing inference on a dataset
### Use it like this
```sh
python infer.py --config configs/example_configs/example_infer.ini
```
## report.py
configs/example_configs/report_example_dataset.ini holds parameters for creating a report based on images, label-images and prediction-images
### Use it like this
```sh
python report.py --config configs/example_configs/report_example_dataset.ini
```

All example config files work with the example dataset that comes with the github https://github.com/SDFIdk/multi_channel_dataset_creation.git
