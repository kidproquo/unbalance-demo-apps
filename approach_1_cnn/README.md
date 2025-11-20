# Approach 1: Convolutional Neural Network on Raw Sensor Data

This directory contains a standalone Python script converted from the Jupyter notebook `3_Approach_1_CNN.ipynb`.

## Description

This approach uses a pre-trained 3-layer Convolutional Neural Network (CNN) - the most accurate configuration - to detect unbalance conditions in real-time from raw vibration sensor data.

### Key Features

1. **Pre-trained Model**: Uses reference model from `models/reference/cnn_3_layers.h5` (no training needed!)
2. **Real-time Detection**: Processes evaluation data as timeseries with 1-minute windows
3. **Anomaly Detection**: Detects unbalance anomalies using majority voting across 60 one-second windows
4. **Automatic Reporting**: Generates detailed figures for each detected unbalance with:
   - Full minute timeseries visualization
   - Prediction scores over time
   - Highest prediction window detail
   - Timestamp and row index information

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

### Basic Usage

Run the script with default settings (uses reference model, all datasets, maximum speed):

```bash
python approach_1_cnn.py
```

**Note:** The script uses the pre-trained reference model by default, so no training is required!

### Command-Line Arguments

```bash
python approach_1_cnn.py [OPTIONS]

Options:
  --dataset {all,0E,1E,2E,3E,4E}
                        Dataset to evaluate (default: all)
                        - all: Process all evaluation datasets
                        - 0E: No unbalance
                        - 1E: Unbalance level 1
                        - 2E: Unbalance level 2
                        - 3E: Unbalance level 3
                        - 4E: Maximum unbalance

  --speed FLOAT         Processing speed multiplier (default: 0)
                        - 0: Maximum speed (no delays)
                        - 1.0: Real-time (1 min data = 60s processing)
                        - 2.0: 2x real-time speed
                        - 0.5: Half real-time speed

  --model-path PATH     Path to trained model file
                        (default: ../../models/reference/cnn_3_layers.h5)

  --data-url PATH       Path or URL to dataset ZIP file
                        (default: ../../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip)

  --output-dir DIR      Directory to save detection figures
                        (default: ../../figures/detections)

  -h, --help           Show help message and exit
```

### Examples

```bash
# Process all datasets at maximum speed (uses reference model)
python approach_1_cnn.py

# Process only dataset 4E (maximum unbalance) at real-time speed
python approach_1_cnn.py --dataset 4E --speed 1.0

# Process dataset 2E at 2x real-time speed
python approach_1_cnn.py --dataset 2E --speed 2.0

# Process only dataset 0E (baseline, no unbalance)
python approach_1_cnn.py --dataset 0E

# Use a custom trained model
python approach_1_cnn.py --dataset 1E --model-path /path/to/custom_model.h5

# Process all datasets with custom output directory
python approach_1_cnn.py --dataset all --output-dir ./my_detections

# View help
python approach_1_cnn.py --help
```

## How It Works

1. **Load Pre-trained Model**: Loads reference model from `models/reference/cnn_3_layers.h5`
2. **Data Loading**: Loads vibration data from ZIP file (evaluation sets: 0E-4E)
3. **Model Validation**: Validates the model performance on the combined evaluation dataset
4. **Timeseries Processing**:

   **For Single Dataset (e.g., `--dataset 4E`):**
   - Processes the dataset sequentially minute-by-minute
   - Each minute: 245,760 samples @ 4096 Hz

   **For All Datasets (`--dataset all`):**
   - **Weighted Random Sampling** (simulates realistic conditions):
     - 75% probability: samples from 0E (no unbalance)
     - 25% probability: randomly samples from 1E-4E (unbalanced)
   - Each minute is randomly selected from the weighted pool
   - Simulates real-world scenario where most operation is normal

   **Detection Logic (Both Modes):**
   - Each minute divided into 60 one-second windows
   - CNN predicts unbalance for each window
   - Uses majority voting (>50%) to determine if the minute shows unbalance

5. **Detection & Logging**: When unbalance is detected:
   - Prints timestamp, row indices, and detection statistics
   - Shows which source dataset the data came from
   - Generates a comprehensive 3-panel figure
   - Saves figure with timestamp in filename

## Configuration

### Command-Line Arguments (Recommended)
- `--dataset`: Which dataset to process (0E, 1E, 2E, 3E, 4E, or all)
- `--speed`: Processing speed (0=max, 1.0=real-time, >1=faster)
- `--model-path`: Path to model file (default: reference model)
- `--output-dir`: Where to save detection figures

### Code Constants (Advanced)
Edit these in `approach_1_cnn.py` if needed:
- `UNBALANCE_THRESHOLD`: Prediction threshold for detection (default: 0.5)
- `MINUTE_WINDOW`: Size of processing window in samples (default: 60 * 4096)

## Outputs

### 1. Console Output
- Training progress and loss/accuracy metrics
- Real-time detection alerts with timestamps and row indices
- Summary statistics

### 2. Model Files
- Uses pre-trained model: `../../models/reference/cnn_3_layers.h5`

### 3. Detection Figures (saved to `../../figures/detections/`)
- Filename format: `unbalance_detection_{dataset}_{timestamp}_row{index}.png`
- Each figure contains:
  - **Panel 1**: Full minute of vibration data
  - **Panel 2**: Prediction scores per second with threshold line
  - **Panel 3**: Detailed view of highest prediction window

## Model Architecture

The 3-layer CNN consists of:
- **3 convolutional layers** with batch normalization and LeakyReLU activation
  - Layer 1: 10 filters, kernel size 9
  - Layer 2: 20 filters, kernel size 9
  - Layer 3: 30 filters, kernel size 9
- **Max pooling** after each conv layer (pool size 5, stride 2)
- **Fully connected layer** (128 units, ReLU)
- **Output layer** (1 unit, sigmoid) for binary classification

## Dataset

The script expects a ZIP file containing CSV files for different unbalance conditions:
- **Training sets**: 0D (no unbalance), 1D, 2D, 3D, 4D (increasing unbalance)
- **Evaluation sets**: 0E (no unbalance), 1E, 2E, 3E, 4E (increasing unbalance)

Each dataset represents different levels of mechanical unbalance from a rotating shaft.

## Performance

- **Validation Accuracy**: ~98.1% (3-layer CNN)
- **Training time**: ~100 epochs (varies with hardware)
- **Processing speed**:
  - Maximum speed (--speed 0): Processes as fast as hardware allows
  - Real-time (--speed 1.0): 1 minute of data = 60 seconds processing
  - Adjustable via `--speed` argument
- **Detection sensitivity**: Configurable via `UNBALANCE_THRESHOLD` constant

## Example Output

### Single Dataset Mode
```
⚠️  UNBALANCE DETECTED at minute 15
    Timestamp: 2020-01-01 00:15:00
    Row index: 3,686,400 - 3,932,160
    Detection ratio: 95.0%
    Mean prediction: 0.9234
    Figure saved: ../../figures/detections/unbalance_detection_4E_20200101_001500_row3686400.png
```

### Weighted Random Sampling Mode (All Datasets)
```
Mode: Weighted Random Sampling
  75% weight on 0E (no unbalance)
  25% weight on 1E-4E (unbalanced, randomly selected)

0E: 1,678 minutes available
1E: 1,678 minutes available
2E: 1,678 minutes available
3E: 1,678 minutes available
4E: 1,678 minutes available

Processing 1,678 minutes with weighted random sampling...

  ⚠️  UNBALANCE DETECTED at minute 42
      Source: 3E (Unbalance Level 3)
      Timestamp: 2020-01-01 00:42:00
      Row index: 614,400 - 860,160
      Detection ratio: 88.3%
      Mean prediction: 0.8521
      Figure saved: ../../figures/detections/...

Weighted Random Sampling Complete
Total minutes processed: 1,678
Total unbalance detections: 127

Dataset Selection Statistics:
  0E: 1,258 times (75.0%)
  1E: 105 times (6.3%)
  2E: 110 times (6.6%)
  3E: 98 times (5.8%)
  4E: 107 times (6.4%)
```

## Important Notes

### Pre-trained Reference Model
- **Uses pre-trained reference model** from `models/reference/cnn_3_layers.h5`
- **No training required** - ready to use immediately!
- Model was trained with ~98% validation accuracy
- If you want to use a different model, specify with `--model-path`

### Detection Behavior
- Uses only the first vibration sensor (`Vibration_1`) for analysis
- First 50,000 samples are skipped to avoid warm-up phase noise
- Detection uses majority voting across 60 one-second windows per minute
- Figures are only generated when >50% of windows in a minute detect unbalance

### Weighted Random Sampling (All Datasets Mode)
- **Simulates realistic operating conditions** where most time is normal operation
- **75% weight on 0E**: Most samples from normal (no unbalance) data
- **25% weight on 1E-4E**: Occasional samples from unbalanced conditions (randomly distributed)
- Each minute randomly selects a source dataset based on weights
- Provides statistics showing actual distribution at the end

### Workflow
```
Every Run:    Load Pre-trained Model → Validate → Detect Anomalies
              (Fast! No training needed!)

Single Dataset:  Sequential processing through one dataset
All Datasets:    Weighted random sampling (75% normal, 25% unbalanced)
```
