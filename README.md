# TraffiCL

PyTorch implementation of TraffiCL for continual traffic forecasting.

<p align="center">
  <img src="fig/TraffiCL-framework.png" alt="TraffiCL framework" width="100%">
</p>

<p align="center">
  <em>The overall framework of TraffiCL.</em>
</p>

## 1. Environment Setup

### Create an environment

Create a dedicated Conda environment with Python 3.12:

```bash
conda create -n trafficl python=3.12 -y
conda activate trafficl
```

Clone the repository and update the package installer:

```bash
git clone https://github.com/Lilin-Chen/TraffiCL.git
cd TraffiCL
python -m pip install --upgrade pip
```

### Install PyTorch

For NVIDIA GPU execution, install the PyTorch build compatible with your driver
and hardware using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

For CPU execution:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Install project dependencies

```bash
python -m pip install -r requirements.txt
```

| Dependency | Purpose |
| --- | --- |
| PyTorch | Model training and inference |
| NumPy | Numerical arrays |
| Pandas | Tabular data and result files |
| PyYAML | Configuration loading |
| tqdm | Progress display |
| scikit-learn | Data utilities |
| SciPy | Scientific computing |

### Check the environment

```bash
python -m pip check
python -c "import torch, numpy, pandas, yaml, tqdm, sklearn, scipy; print('Dependencies imported successfully'); print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The default training device is `auto`: CUDA is selected when available;
otherwise, training runs on CPU. Set `training.device` to `cpu` or `cuda` in the
training configuration to choose a device explicitly.

## 2. Configuration

Configure the model in `configs/models/` and training in `configs/continual/`.
The training script loads the provided TraffiCL configuration by default.

## 3. Training and Evaluation

```bash
python scripts/run_trafficl_global.py \
  --cache-root "/path/to/cache" \
  --seed 42 \
  --datasets caltrans_bicycle caltrans_pedestrian dc_nova pvcgn_hangzhou pvcgn_shanghai shanghai_metroflow minneapolis_i94 mta_subway \
  --output-dir results/trafficl \
  --checkpoint-dir results/trafficl/checkpoints
```

Evaluation runs during training. Results are saved to `results/trafficl/`, and
checkpoints are saved to `results/trafficl/checkpoints/`.

## 4. Resume Training

Append `--resume` to the training command, retaining the original configuration
and checkpoint directory.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
