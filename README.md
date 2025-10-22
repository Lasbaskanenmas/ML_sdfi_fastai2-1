# SDFI_ml

Machine learning framework developed and maintained by **SDFI** for performing **semantic segmentation** on **multichannel images**.

---

## Installation

```sh
mamba env create --file environment.yml
mamba activate ML_sdfi
pip install -e .
```

### Verify CUDA Support

```sh
mamba activate ML_sdfi
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
python src/ML_sdfi_fastai2/report.py --config configs/example_configs/report_example_dataset.ini
```

---

## Example Dataset

All example configuration files are compatible with the example dataset available at:  
👉 [https://github.com/SDFIdk/multi_channel_dataset_creation](https://github.com/SDFIdk/multi_channel_dataset_creation)
