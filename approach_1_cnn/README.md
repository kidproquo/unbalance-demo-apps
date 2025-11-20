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
                        - 1.0: Real-time (window time = processing time)
                        - 2.0: 2x real-time speed
                        - 0.5: Half real-time speed

  --time-window INT     Time window size in seconds (default: 60)
                        - 60: 1 minute windows (default)
                        - 120: 2 minute windows
                        - 30: 30 second windows
                        - Adjusts how much data is analyzed at once

  --max-windows INT     Maximum number of windows to process (default: None)
                        - None: Infinite processing with rollover (Ctrl+C to stop)
                        - 100: Process exactly 100 windows then stop
                        - Useful for testing or limited runs

  --normal-weight FLOAT Weight for normal (0E) dataset in weighted random sampling (default: 0.9)
                        - Only applies when --dataset=all
                        - 0.9: 90% normal, 10% unbalanced (default)
                        - 0.75: 75% normal, 25% unbalanced
                        - 0.5: 50% normal, 50% unbalanced
                        - Range: 0.0-1.0

  --log-interval INT    Log performance metrics to console and write to CSV every N windows (default: 10)
                        - 10: Log/write every 10 windows (default)
                        - 1: Log/write after every window (verbose, real-time)
                        - 60: Log/write every 60 windows (if window=1s, this is every minute)
                        - 100: Log/write every 100 windows (less frequent updates)
                        - Note: Both console logging and CSV writing happen at the same interval

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
# Process all datasets at maximum speed with default 1-minute windows
python approach_1_cnn.py

# Process only dataset 4E (maximum unbalance) at real-time speed
python approach_1_cnn.py --dataset 4E --speed 1.0

# Process with 30-second windows instead of 1-minute
python approach_1_cnn.py --dataset all --time-window 30

# Process with 2-minute windows for longer-term analysis
python approach_1_cnn.py --dataset 2E --time-window 120

# Real-time processing with 1-minute windows
python approach_1_cnn.py --dataset all --time-window 60 --speed 1.0

# Process only dataset 0E (baseline, no unbalance)
python approach_1_cnn.py --dataset 0E

# Use a custom trained model
python approach_1_cnn.py --dataset 1E --model-path /path/to/custom_model.h5

# Process all datasets with custom output directory
python approach_1_cnn.py --dataset all --output-dir ./my_detections

# Process with custom normal weight (75% normal, 25% unbalanced)
python approach_1_cnn.py --dataset all --normal-weight 0.75

# Process with balanced sampling (50% normal, 50% unbalanced)
python approach_1_cnn.py --dataset all --normal-weight 0.5

# Process exactly 100 windows then stop
python approach_1_cnn.py --dataset all --max-windows 100

# Infinite processing with rollover (default, Ctrl+C to stop)
python approach_1_cnn.py --dataset all

# Log metrics every window (verbose mode, useful for debugging)
python approach_1_cnn.py --dataset all --max-windows 50 --log-interval 1

# Log metrics every 100 windows (less console output)
python approach_1_cnn.py --dataset all --log-interval 100

# View help
python approach_1_cnn.py --help
```

## How It Works

1. **Load Pre-trained Model**: Loads reference model from `models/reference/cnn_3_layers.h5`
2. **Data Loading**: Loads vibration data from ZIP file (evaluation sets: 0E-4E)
3. **Model Validation**: Validates the model performance on the combined evaluation dataset
4. **Timeseries Processing**:

   **For Single Dataset (e.g., `--dataset 4E`):**
   - Processes the dataset sequentially window-by-window
   - Default: 60-second windows (245,760 samples @ 4096 Hz)
   - Configurable via `--time-window` parameter

   **For All Datasets (`--dataset all`):**
   - **Weighted Random Sampling** (simulates realistic conditions):
     - 90% probability: samples from 0E (no unbalance) by default
     - 10% probability: randomly samples from 1E-4E (unbalanced) by default
     - Configurable via `--normal-weight` parameter
   - Each time window is randomly selected from the weighted pool
   - Simulates real-world scenario where most operation is normal

   **Detection Logic (Both Modes):**
   - Each time window divided into one-second sub-windows
   - CNN predicts unbalance for each second
   - Uses majority voting (>50%) to determine if the window shows unbalance

5. **Detection & Logging**: When unbalance is detected:
   - Prints system UTC timestamp, row indices, and detection statistics
   - Shows which source dataset the data came from
   - Generates a comprehensive 3-panel figure
   - Saves figure with UTC timestamp in filename

## Configuration

### Command-Line Arguments (Recommended)
- `--dataset`: Which dataset to process (0E, 1E, 2E, 3E, 4E, or all)
- `--time-window`: Time window size in seconds (default: 60)
- `--max-windows`: Maximum windows to process (default: None = infinite)
- `--normal-weight`: Weight for normal data in sampling (default: 0.9 = 90%)
- `--speed`: Processing speed (0=max, 1.0=real-time, >1=faster)
- `--model-path`: Path to model file (default: reference model)
- `--output-dir`: Where to save detection figures

### Code Constants (Advanced)
Edit these in `approach_1_cnn.py` if needed:
- `UNBALANCE_THRESHOLD`: Prediction threshold for detection (default: 0.5)
- `SAMPLES_PER_SECOND`: Sampling rate (4096 Hz, fixed)

## Outputs

### 1. Console Output
- Training progress and loss/accuracy metrics
- Real-time detection alerts with UTC timestamps and row indices
- Summary statistics

### 2. Model Files
- Uses pre-trained model: `../../models/reference/cnn_3_layers.h5`

### 3. Detection Figures (saved to `../../figures/detections/`)
- Filename format: `unbalance_detection_{dataset}_{UTC_timestamp}_window{idx}_row{index}.png`
- Timestamp format: `YYYYMMDD_HHMMSS` (UTC time)
- Each figure contains:
  - **Panel 1**: Full minute of vibration data
  - **Panel 2**: Prediction scores per second with threshold line
  - **Panel 3**: Detailed view of highest prediction window

### 4. Performance Report (saved to `../../figures/detections/`)
- Filename format: `performance_report_{UTC_timestamp}.txt`
- Timestamp format: `YYYYMMDD_HHMMSS` (UTC time when processing started)
- **Updated every N windows** (based on `--log-interval`)
- Each run creates its own report file with unique timestamp
- Maintained file that gets overwritten with latest metrics throughout the run
- Contains performance metrics table with:
  - Per-dataset statistics: TP, FP, TN, FN counts
  - Accuracy, Precision, Recall per dataset
  - Overall performance metrics across all datasets
- **Metric definitions**:
  - **TP (True Positive)**: Correctly detected unbalance in 1E-4E
  - **FP (False Positive)**: Incorrectly detected unbalance in 0E
  - **TN (True Negative)**: Correctly identified no unbalance in 0E
  - **FN (False Negative)**: Missed unbalance in 1E-4E

### 5. Performance Summary (Console Output)
- **Live metrics**: Logged periodically during processing
  - Default: every 10 windows (configurable with `--log-interval`)
  - Shows running counts of TP, FP, TN, FN per dataset
  - Displays current Accuracy, Precision, Recall for each dataset
  - Example: If window=60s and log-interval=60, updates every hour (60 minutes)
  - Useful for monitoring long-running processes
- **Final summary**: Printed at end of processing
  - Complete performance metrics table
  - Same format as the performance report file

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
    Timestamp: 2025-11-19 14:32:45 (UTC)
    Row index: 3,686,400 - 3,932,160
    Detection ratio: 95.0%
    Mean prediction: 0.9234
    Figure saved: ../../figures/detections/unbalance_detection_4E_20251119_143245_row3686400.png
```

### Weighted Random Sampling Mode (All Datasets)
```
Mode: Weighted Random Sampling
  90% weight on 0E (no unbalance)
  10% weight on 1E-4E (unbalanced, randomly selected)

0E: 1,678 minutes available
1E: 1,678 minutes available
2E: 1,678 minutes available
3E: 1,678 minutes available
4E: 1,678 minutes available

Processing 1,678 minutes with weighted random sampling...

================================================================================
Running Performance Metrics (after 10 windows, 42.3s elapsed)
================================================================================
Dataset    Processed  TP     FP     TN     FN     Accuracy
--------------------------------------------------------------------------------
0E         9          0      1      8      0      0.889
1E         1          1      0      0      0      1.000
--------------------------------------------------------------------------------
Overall    10         1      1      8      0      0.900
================================================================================

  ⚠️  UNBALANCE DETECTED at minute 42
      Source: 3E (Unbalance Level 3)
      Timestamp: 2025-11-19 14:45:12 (UTC)
      Row index: 614,400 - 860,160
      Detection ratio: 88.3%
      Mean prediction: 0.8521
      Figure saved: ../../figures/detections/unbalance_detection_3E_20251119_144512_row614400.png

Weighted Random Sampling Complete
Total minutes processed: 1,678
Total unbalance detections: 85

Dataset Selection Statistics:
  0E: 1,510 times (90.0%)
  1E: 42 times (2.5%)
  2E: 41 times (2.4%)
  3E: 43 times (2.6%)
  4E: 42 times (2.5%)

================================================================================
Performance Metrics
================================================================================
Dataset    Total    TP       FP       TN       FN       Accuracy   Precision  Recall
--------------------------------------------------------------------------------
0E         1510     0        35       1475     0        0.977      0.000      N/A
1E         42       40       0        0        2        0.952      1.000      0.952
2E         41       39       0        0        2        0.951      1.000      0.951
3E         43       41       0        0        2        0.953      1.000      0.953
4E         42       40       0        0        2        0.952      1.000      0.952
--------------------------------------------------------------------------------
Overall    1678     160      35       1475     8        0.974      0.821      0.952

Performance report saved to: ../../figures/detections/performance_report_20251119_145230.txt
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
- **Timestamps use system UTC time** for all detection events and filenames

### Weighted Random Sampling (All Datasets Mode)
- **Simulates realistic operating conditions** where most time is normal operation
- **90% weight on 0E (default)**: Most samples from normal (no unbalance) data
- **10% weight on 1E-4E (default)**: Occasional samples from unbalanced conditions (randomly distributed)
- **Configurable via `--normal-weight`**: Adjust the balance between normal and unbalanced data
- Each time window randomly selects a source dataset based on weights
- Provides statistics showing actual distribution at the end

### Workflow
```
Every Run:    Load Pre-trained Model → Validate → Detect Anomalies
              (Fast! No training needed!)

Single Dataset:  Sequential processing through one dataset
All Datasets:    Weighted random sampling (90% normal, 10% unbalanced by default)
                 Configurable via --normal-weight parameter
```
