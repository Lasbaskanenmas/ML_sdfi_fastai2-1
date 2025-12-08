# SDFI_ml

Machine learning framework developed and maintained by **KDS** for performing **semantic segmentation** on **multichannel images**.

---

## Installation

```sh
mamba env create --file environment.yml
mamba activate ML_sdfi
pip install -e .
```

### Verify CUDA Support

```sh
python
>>> import torch
>>> torch.cuda.is_available()
True
```

If `torch.cuda.is_available()` returns `False`, you may need to reinstall a CUDA-compatible version of PyTorch following the instructions at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

For example:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Training

Configuration file:  
`configs/example_configs/train_example_dataset.ini`

Run:
```sh
python src/ML_sdfi_fastai2/train.py --config configs/example_configs/train_example_dataset.ini
```

---

### Inference

Configuration file:  
`configs/example_configs/infer_example_dataset.ini`

Run:
```sh
python src/ML_sdfi_fastai2/infer.py --config configs/example_configs/infer_example_dataset.ini
```

---

### Reporting

Configuration file:  
`configs/example_configs/report_example_dataset.ini`

Run:
```sh
In order to analyse how well the model does on a dataset we can create a .pdf report with loss plots , confusion matrixes and example iamges and predictions

cd ../multi_channel_dataset_creation
python src/multi_channel_dataset_creation/geopackage_to_label_v2.py --geopackage example_dataset/labels/example_dataset_buildings.gpkg --input_folder example_dataset/data/splitted/rgb/ --output_folder example_dataset/buildings/splitted_buildings/ --background_value 0 --value_used_for_all_polygons 1
cd ../ML_sdfi_fastai2
python src/ML_sdfi_fastai2/report.py --config configs/example_configs/report_example_dataset.ini

If you get confusing errors when creating report it is often casued by a faulty .csv file 
check ../logs_and_models/example_dataset_iter_1/iter_1/logs/iter_1.csv 
This file should look like this

epoch,train_loss,valid_loss,valid_accuracy,time,lr_0,lr_1,lr_2
0,2.671426296234131,2.261967182159424,0.15685869753360748,00:07,8e-05,8e-05,8e-05
1,2.4981231689453125,1.2677741050720215,0.6439558863639832,00:06
...
...
9,3.6438961029052734,1.2790676355361938,0.6442530751228333,00:06,8.64737438191444e-05,8.64737438191444e-05,8.64737438191444e-05

```

---

## Example Dataset

All example configuration files are compatible with the example dataset available at:  
👉 [https://github.com/SDFIdk/multi_channel_dataset_creation](https://github.com/SDFIdk/multi_channel_dataset_creation)


