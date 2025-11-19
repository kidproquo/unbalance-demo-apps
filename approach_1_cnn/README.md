# Approach 1: Convolutional Neural Network on Raw Sensor Data

This directory contains a standalone Python script converted from the Jupyter notebook `3_Approach_1_CNN.ipynb`.

## Description

This approach uses a Convolutional Neural Network (CNN) to classify unbalance conditions directly from raw vibration sensor data. The CNN is trained with different numbers of convolutional layers (1-4) to evaluate the impact of model depth on classification accuracy.

## Python Version Requirements

**⚠️ IMPORTANT:** This script requires **Python 3.8-3.11** due to TensorFlow compatibility.

Python 3.12+ (including 3.13) is **NOT** currently supported by TensorFlow.

### If using Python 3.13

Create a new virtual environment with Python 3.11:

```bash
# Check if Python 3.11 is installed
python3.11 --version

# If not installed, install it first (macOS example):
# brew install python@3.11

# Create a new virtual environment
python3.11 -m venv venv_py311
source venv_py311/bin/activate  # On Windows: venv_py311\Scripts\activate
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script from this directory:

```bash
python approach_1_cnn.py
```

Or from the project root:

```bash
python apps/approach_1_cnn/approach_1_cnn.py
```

## Configuration

Edit the following variables in `approach_1_cnn.py` to customize behavior:

- `url`: Path to the dataset ZIP file (local or remote)
- `use_reference_models`: Set to `True` to use pre-trained models, `False` to train new models
- `model_path`: Directory containing model files

## Outputs

The script generates:

1. **Console output**: Training progress and evaluation metrics
2. **Figure files** (saved to `../../figures/`):
   - `Fig.5a_cnn_unbalance.png`: Classification accuracy vs. unbalance factor
   - `Fig.5b_cnn_rpm.png`: Classification accuracy vs. rotation speed

## Model Architecture

The CNN models consist of:
- 1-4 convolutional layers with batch normalization and LeakyReLU activation
- Max pooling layers
- Fully connected layers with ReLU activation
- Binary classification output (sigmoid activation)

## Dataset

The script expects a ZIP file containing CSV files for different unbalance conditions:
- Training sets: 0D, 1D, 2D, 3D, 4D
- Validation sets: 0E, 1E, 2E, 3E, 4E

Each dataset represents different levels of mechanical unbalance.

## Performance

Typical accuracies on validation data:
- 1 conv. layer: ~93.6%
- 2 conv. layers: ~90.9%
- 3 conv. layers: ~98.1%
- 4 conv. layers: ~97.6%

## Notes

- The script uses only the first vibration sensor (`Vibration_1`) for analysis
- The first 50,000 samples are skipped to avoid warm-up phase noise
- Class weights are applied during training to handle class imbalance
- For pairwise training experiments, refer to the original Jupyter notebook
