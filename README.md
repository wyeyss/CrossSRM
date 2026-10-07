# CrossSRM

Official implementation of **CrossSRM** for cross-city few-shot traffic speed forecasting.

CrossSRM is pretrained on multiple source cities and adapted to an unseen target city using only **3 days of target-visible data**, including 2 days for adaptation and 1 day for validation.

## Requirements

Python 3.10 is recommended.

Install PyTorch according to your CUDA environment first.
The experiments in the paper were conducted with PyTorch 2.1.2 and CUDA 11.8.

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The processed traffic datasets used in this study are obtained via the data link provided in the TPB repository:

https://github.com/zhyliu00/TPB

Please download the datasets from TPB and organize them as follows:

```text
data/
├── metr-la/
│   ├── dataset_expand.npy
│   └── matrix.npy
├── pems-bay/
│   ├── dataset_expand.npy
│   └── matrix.npy
├── chengdu_m/
│   ├── dataset_expand.npy
│   └── matrix.npy
└── shenzhen/
    ├── dataset_expand.npy
    └── matrix.npy
```

For each city:

- `dataset_expand.npy`: traffic sequence with shape `[time, nodes, features]`, where the first channel is traffic speed.
- `matrix.npy`: adjacency matrix with shape `[nodes, nodes]`.

Chengdu and Shenzhen are converted from 10-minute to 5-minute resolution by linear interpolation in the data loader. METR-LA and PEMS-BAY use their original 5-minute resolution.

## Training and Evaluation

For example, to use **PEMS-BAY** as the target city:

```bash
python train.py \
  --config_filename configs/config_pems.yaml \
  --target_city pems-bay \
  --data_list chengdu_shenzhen_metr \
  --seed 7 \
  --gpu 0
```

The other leave-one-city-out settings can be run by changing the target city, source-city list, and configuration file accordingly.

Use CPU instead of GPU with:

```bash
python train.py --cpu
```

Additional options are available with:

```bash
python train.py --help
```

## Experimental Protocol

- Input length: 288 steps (24 hours at 5-minute resolution).
- Forecast horizon: 12 steps (5–60 minutes).
- Target-visible data: 3 days in total.
  - Days 1–2: target adaptation.
  - Day 3: validation and model selection.
- The target task graph is constructed from the target adaptation set.
- Target normalization statistics are estimated from the 3 target-visible days.
- Source-city data are chronologically divided into training and validation subsets.

## Citation

If you find this work useful, please cite:

> **Spatiotemporal Residual Modeling for Cross-City Few-Shot Traffic Speed Forecasting**

The complete citation will be updated after publication.

## License

This project is released under the MIT License. See `LICENSE` for details.
