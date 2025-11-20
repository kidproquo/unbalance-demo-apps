# Approach 2: Fully-Connected Neural Network on FFT-transformed Data

This directory contains a standalone Python script converted from the Jupyter notebook `3_Approach_2_FFT_FCN.ipynb`.

## Description

This approach uses a pre-trained 4-layer Fully-Connected Neural Network (FCN) operating on FFT-transformed vibration data to detect unbalance conditions in real-time.

### Key Features

1. **FFT Preprocessing**: Applies Fast Fourier Transform to convert time-domain signals to frequency domain
2. **RobustScaler**: Uses scikit-learn's RobustScaler for feature normalization
3. **Pre-trained Model**: Uses reference model from `models/reference/fft_fcn_4_layers.h5` (no training needed!)
4. **Real-time Detection**: Processes evaluation data as timeseries with 1-minute windows
5. **Anomaly Detection**: Detects unbalance anomalies using majority voting across 60 one-second windows
6. **Automatic Reporting**: Generates detailed figures for each detected unbalance with:
   - Full minute timeseries visualization
   - Prediction scores over time
   - FFT spectrum of highest prediction window
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

## MCP Server (Optional)

An MCP (Model Context Protocol) server is available for real-time monitoring and querying of the detection system's performance metrics and detection events.

**Integrated Mode (Recommended):**

```bash
# Install fastmcp
pip install fastmcp

# Run with MCP server enabled (runs in background thread with HTTP/SSE on port 8000)
python approach_2_fft.py --dataset all --enable-mcp

# Custom port
python approach_2_fft.py --dataset all --enable-mcp --mcp-port 8080
```

**Standalone Mode:**

```bash
# Run the MCP server separately (in a different terminal)
python mcp_server.py
```

**Features:**
- HTTP/SSE transport for easy integration with web apps and clients
- Query current performance metrics (TP/FP/TN/FN, Accuracy, Precision, Recall)
- Get recent unbalance detection events
- Check system status and running state
- Real-time monitoring while processing data in a loop
- Accessible at `http://localhost:8000/sse` (or custom port)

**See approach_1_cnn/MCP_SERVER.md for detailed documentation.**

## Usage

### Basic Usage

Run the script with default settings (uses reference model, all datasets, maximum speed):

```bash
python approach_2_fft.py
```

**Note:** The script uses the pre-trained reference model by default, so no training is required!

### Command-Line Arguments

```bash
python approach_2_fft.py [OPTIONS]

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

  --log-interval INT    Log performance metrics to console every N windows (default: 10)
                        - 10: Log every 10 windows (default)
                        - 1: Log after every window (verbose, real-time)
                        - 60: Log every 60 windows (if window=1s, this is every minute)
                        - 100: Log every 100 windows (less frequent updates)

  --enable-mcp          Enable MCP server for real-time monitoring
                        - Runs MCP server in background thread
                        - Uses HTTP/SSE transport (default port: 8000)
                        - Allows querying metrics and detections while processing
                        - Requires: pip install fastmcp

  --mcp-port INT        Port for MCP server HTTP transport (default: 8000)
                        - Only used when --enable-mcp is set
                        - HTTP/SSE endpoint: http://localhost:<port>/sse

  --model-path PATH     Path to trained model file
                        (default: ../../models/reference/fft_fcn_4_layers.h5)

  --data-url PATH       Path or URL to dataset ZIP file
                        (default: ../../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip)

  --output-dir DIR      Directory to save detection figures
                        (default: ../../figures/detections)

  -h, --help           Show help message and exit
```

### Examples

```bash
# Process all datasets at maximum speed with default 1-minute windows
python approach_2_fft.py

# Process only dataset 4E (maximum unbalance) at real-time speed
python approach_2_fft.py --dataset 4E --speed 1.0

# Process with 30-second windows instead of 1-minute
python approach_2_fft.py --dataset all --time-window 30

# Process with 2-minute windows for longer-term analysis
python approach_2_fft.py --dataset 2E --time-window 120

# Real-time processing with 1-minute windows
python approach_2_fft.py --dataset all --time-window 60 --speed 1.0

# Process only dataset 0E (baseline, no unbalance)
python approach_2_fft.py --dataset 0E

# Use a custom trained model
python approach_2_fft.py --dataset 1E --model-path /path/to/custom_model.h5

# Process all datasets with custom output directory
python approach_2_fft.py --dataset all --output-dir ./my_detections

# Process with custom normal weight (75% normal, 25% unbalanced)
python approach_2_fft.py --dataset all --normal-weight 0.75

# Process with balanced sampling (50% normal, 50% unbalanced)
python approach_2_fft.py --dataset all --normal-weight 0.5

# Process exactly 100 windows then stop
python approach_2_fft.py --dataset all --max-windows 100

# Infinite processing with rollover (default, Ctrl+C to stop)
python approach_2_fft.py --dataset all

# Log metrics every window (verbose mode, useful for debugging)
python approach_2_fft.py --dataset all --max-windows 50 --log-interval 1

# Log metrics every 100 windows (less console output)
python approach_2_fft.py --dataset all --log-interval 100

# Enable MCP server for real-time monitoring while processing (HTTP on port 8000)
python approach_2_fft.py --dataset all --enable-mcp

# Enable MCP server on custom port
python approach_2_fft.py --dataset all --enable-mcp --mcp-port 8080

# View help
python approach_2_fft.py --help
```

## How It Works

1. **Load Pre-trained Model**: Loads reference model from `models/reference/fft_fcn_4_layers.h5`
2. **Data Loading**: Loads vibration data from ZIP file (evaluation sets: 0E-4E)
3. **FFT Preprocessing**: Applies Fast Fourier Transform to convert time-domain signals to frequency domain
   - Input: 4096 samples (1 second @ 4096 Hz)
   - FFT Output: 2048 frequency features
   - DC component (first frequency bin) is zeroed out
4. **Scaling**: Applies RobustScaler normalization (fitted on subset of data)
5. **Model Validation**: Validates the model performance on the combined evaluation dataset
6. **Timeseries Processing**:

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
   - FFT applied to each second
   - FCN predicts unbalance for each second
   - Uses majority voting (>50%) to determine if the window shows unbalance

7. **Detection & Logging**: When unbalance is detected:
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
Edit these in `approach_2_fft.py` if needed:
- `UNBALANCE_THRESHOLD`: Prediction threshold for detection (default: 0.5)
- `SAMPLES_PER_SECOND`: Sampling rate (4096 Hz, fixed)
- `FFT_FEATURES`: Number of FFT features (2048, half of window size)

## Outputs

### 1. Console Output
- FFT transformation and scaling info
- Real-time detection alerts with UTC timestamps and row indices
- Summary statistics

### 2. Model Files
- Uses pre-trained model: `../../models/reference/fft_fcn_4_layers.h5`

### 3. Detection Figures (saved to `../../figures/detections/`)
- Filename format: `unbalance_detection_{dataset}_{UTC_timestamp}_window{idx}_row{index}.png`
- Timestamp format: `YYYYMMDD_HHMMSS` (UTC time)
- Each figure contains:
  - **Panel 1**: Full minute of vibration data (time domain)
  - **Panel 2**: Prediction scores per second with threshold line
  - **Panel 3**: FFT spectrum of highest prediction window

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

The 4-layer FCN consists of:
- **Input layer**: 2048 FFT features
- **4 hidden fully-connected layers** with LeakyReLU activation
  - Each layer: 1024 units
  - Activation: LeakyReLU (alpha=0.05)
- **Output layer**: 1 unit, sigmoid activation for binary classification

Total parameters: ~8.4 million

## Dataset

The script expects a ZIP file containing CSV files for different unbalance conditions:
- **Training sets**: 0D (no unbalance), 1D, 2D, 3D, 4D (increasing unbalance)
- **Evaluation sets**: 0E (no unbalance), 1E, 2E, 3E, 4E (increasing unbalance)

Each dataset represents different levels of mechanical unbalance from a rotating shaft.

## Performance

- **Validation Accuracy**: ~98.1% (4-layer FFT FCN)
- **FFT Preprocessing**: Fast Fourier Transform applied to each 1-second window
- **Processing speed**:
  - Maximum speed (--speed 0): Processes as fast as hardware allows
  - Real-time (--speed 1.0): 1 minute of data = 60 seconds processing
  - Adjustable via `--speed` argument
- **Detection sensitivity**: Configurable via `UNBALANCE_THRESHOLD` constant

## Example Output

### Weighted Random Sampling Mode (All Datasets)
```
Mode: Weighted Random Sampling
  90% weight on 0E (no unbalance)
  10% weight on 1E-4E (unbalanced, randomly selected)

0E: 1,678 windows available (1678.0 minutes)
1E: 1,678 windows available (1678.0 minutes)
2E: 1,678 windows available (1678.0 minutes)
3E: 1,678 windows available (1678.0 minutes)
4E: 1,678 windows available (1678.0 minutes)

Processing with weighted random sampling...

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

  ⚠️  UNBALANCE DETECTED at window 42
      Source: 3E (Unbalance Level 3)
      Timestamp: 2025-11-20 08:45:12 (UTC)
      Row index: 614,400 - 860,160
      Detection ratio: 88.3%
      Mean prediction: 0.8521
      Figure saved: ../../figures/detections/unbalance_detection_3E_20251120_084512_row614400.png

Weighted Random Sampling Complete
Total windows processed: 1,678
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

Performance report saved to: ../../figures/detections/performance_report_20251120_084530.txt
```

## Important Notes

### Pre-trained Reference Model
- **Uses pre-trained reference model** from `models/reference/fft_fcn_4_layers.h5`
- **No training required** - ready to use immediately!
- Model was trained with ~98% validation accuracy
- If you want to use a different model, specify with `--model-path`

### FFT Preprocessing
- **Fast Fourier Transform** applied to each 1-second window (4096 samples)
- **Frequency features**: 2048 (half of window size, from rfft)
- **DC component zeroed**: First frequency bin set to 0
- **RobustScaler normalization**: Fitted on subset of data, applied to all windows

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
Every Run:    Load Pre-trained Model → Apply FFT → Validate → Detect Anomalies
              (Fast! No training needed!)

Single Dataset:  Sequential processing through one dataset
All Datasets:    Weighted random sampling (90% normal, 10% unbalanced by default)
                 Configurable via --normal-weight parameter
```

## Differences from Approach 1 (CNN)

| Feature | Approach 1 (CNN) | Approach 2 (FFT FCN) |
|---------|------------------|----------------------|
| Input | Raw time-domain signal (4096 samples) | FFT frequency features (2048 features) |
| Preprocessing | None (minimal normalization in CNN) | FFT + RobustScaler |
| Model Type | 3-layer Convolutional Neural Network | 4-layer Fully-Connected Network |
| Model Size | ~3.1M parameters | ~8.4M parameters |
| Accuracy | ~98.1% | ~98.1% |
| Speed | Faster (no FFT overhead) | Slightly slower (FFT computation) |
| Interpretability | Learns features from raw signal | Works on explicit frequency features |

Both approaches achieve similar accuracy but operate on different representations of the data!
