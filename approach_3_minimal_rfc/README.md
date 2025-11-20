# Approach 3: Random Forest Classifier on Minimal Statistical Features

Machine Learning Based Unbalance Detection using only **7 hand-crafted statistical features**.

## Overview

This approach uses a **Random Forest Classifier** trained on a minimal set of statistical features extracted from vibration sensor data:

- **1 feature**: Mean RPM (rotation speed)
- **3 features**: Standard deviation of each vibration sensor
- **3 features**: Kurtosis of each vibration sensor

**Total**: 7 features per 1-second window

### Why "Minimal Features"?

The original Notebook 4 presents two variants:
1. **TSFRESH variant**: 748 automatically-extracted time series features (complex, slow)
2. **Minimal features variant**: 7 hand-crafted statistical features (simple, fast) ← **This implementation**

We chose minimal features because:
- ✅ **Self-contained**: Can train model from scratch
- ✅ **Fast**: Millisecond feature extraction
- ✅ **Interpretable**: Features have clear physical meaning
- ✅ **Lightweight**: No TensorFlow, no TSFRESH library needed
- ⚠️ **Lower accuracy**: 82% vs 94% (TSFRESH) or 99% (Deep Learning)

See [APPROACH_DECISION.md](APPROACH_DECISION.md) for detailed analysis of why we chose this variant.

## Model Architecture

### Random Forest Classifier

**Type**: Classical Machine Learning (not Deep Learning!)

**Hyperparameters** (from GridSearch):
- `n_estimators`: 30 trees
- `max_depth`: 20
- `min_samples_leaf`: 1
- `scoring`: balanced_accuracy

**Training Data**:
- 32,166 windows (1-second each)
  - 6,438 normal (0D dataset)
  - 25,728 unbalanced (1D-4D datasets)

### Features Explained

Each 1-second window (4096 samples) produces 7 features:

1. **RPM Mean**: Average rotation speed
   - Physical meaning: Machine operating speed
   - Relevance: Unbalance effects vary with RPM

2. **Vibration Sensor 1 Std**: Standard deviation
   - Physical meaning: Vibration amplitude
   - Relevance: Unbalance increases vibration

3. **Vibration Sensor 2 Std**: Standard deviation
   - Same as above for sensor 2

4. **Vibration Sensor 3 Std**: Standard deviation
   - Same as above for sensor 3

5. **Vibration Sensor 1 Kurtosis**: Distribution "tailedness"
   - Physical meaning: How spiky/impulsive the vibration is
   - Relevance: Unbalance creates periodic spikes

6. **Vibration Sensor 2 Kurtosis**
   - Same as above for sensor 2

7. **Vibration Sensor 3 Kurtosis**
   - Same as above for sensor 3

## Performance

### Evaluation Accuracy (0E-4E datasets)

| Dataset | Unbalance Level | Accuracy |
|---------|----------------|----------|
| 0E      | No unbalance   | **93.5%** |
| 1E      | Level 1 (weak) | 45.4% ⚠️ |
| 2E      | Level 2        | 70.9% |
| 3E      | Level 3        | **99.4%** |
| 4E      | Level 4 (strong) | **99.9%** |
| **Mean** |                | **81.8%** |

### Key Observations

**Strengths**:
- ✅ Excellent at detecting strong unbalance (levels 3-4): 99%+
- ✅ Good at identifying normal operation (0E): 93.5%
- ✅ Very fast feature extraction (<1ms per window)

**Weaknesses**:
- ⚠️ Poor at detecting weak unbalance (level 1): 45.4%
- ⚠️ Overall accuracy (82%) significantly lower than deep learning approaches (99%)

**Comparison**:
- Approach 1 (CNN): **99.6%** mean accuracy
- Approach 2 (FFT FCN): **99.7%** mean accuracy
- Approach 3 (Minimal RFC): **81.8%** mean accuracy ← This approach

## Installation

### Prerequisites

- Python 3.8+ (compatible with any Python version, unlike Approaches 1 & 2 which require 3.8-3.11)
- **No TensorFlow required!**

### Install Dependencies

```bash
# Activate virtual environment (if using)
source ../../venv_py311/bin/activate

# Install requirements
pip install -r requirements.txt
```

**Note**: This approach is much lighter than Approaches 1 & 2:
- No TensorFlow (saves ~500MB)
- No GPU needed
- Faster installation

## Usage

### Basic Usage

Process all evaluation datasets with default settings:

```bash
python approach_3_minimal_rfc.py --dataset all
```

### Command-Line Arguments

```
--dataset DATASET          Dataset to process: 0E, 1E, 2E, 3E, 4E, or "all" (default: all)
--time-window SECONDS      Time window duration in seconds (default: 10)
--max-windows N            Maximum number of windows to process (default: infinite)
--speed MULTIPLIER         Playback speed (0=max, 1=real-time, 2=2x, etc.) (default: 0)
--normal-weight WEIGHT     Probability of selecting normal (0E) data (default: 0.9)
--log-interval N           How often to print metrics in windows (default: 10)
--model-path PATH          Path to RFC model (default: ../../models/reference/minimal_rfc.joblib)
--output-dir PATH          Directory to save figures (default: ../../figures/detections)
--enable-mcp               Enable MCP server for real-time monitoring
--mcp-port PORT            Port for MCP server (default: 8000)
```

### Examples

**Process specific dataset**:
```bash
python approach_3_minimal_rfc.py --dataset 3E
```

**Process 100 windows with 60-second time windows**:
```bash
python approach_3_minimal_rfc.py --dataset all --max-windows 100 --time-window 60
```

**Continuous mode with custom normal weight**:
```bash
python approach_3_minimal_rfc.py --dataset all --normal-weight 0.8
```

**Enable MCP server for real-time monitoring**:
```bash
python approach_3_minimal_rfc.py --dataset all --enable-mcp --mcp-port 8000
```

## How It Works

### Step 1: Load Pre-trained Model

The script loads a pre-trained Random Forest Classifier from:
```
models/reference/minimal_rfc.joblib
```

**Training the model yourself** (optional):
```bash
cd ../..
python train_minimal_rfc.py
```

This will:
- Extract features from training data (0D-4D)
- Run GridSearch for hyperparameter tuning
- Save the best model to `models/reference/minimal_rfc.joblib`

### Step 2: Load Datasets

Load evaluation datasets (0E-4E) with weighted random sampling:
- **90% probability**: Select normal (0E) dataset
- **10% probability**: Select unbalanced (1E-4E) dataset

This simulates realistic conditions where normal operation is more common.

### Step 3: Feature Extraction Pipeline

For each time window (e.g., 10 seconds):

1. **Window the data**: Split into 1-second windows (4096 samples each)
2. **Extract features**: Compute 7 statistical features per window
   ```python
   features = [
       np.mean(window[:, 0]),              # RPM mean
       np.std(window[:, 1]),               # Sensor 1 std
       np.std(window[:, 2]),               # Sensor 2 std
       np.std(window[:, 3]),               # Sensor 3 std
       scipy.stats.kurtosis(window[:, 1]), # Sensor 1 kurtosis
       scipy.stats.kurtosis(window[:, 2]), # Sensor 2 kurtosis
       scipy.stats.kurtosis(window[:, 3])  # Sensor 3 kurtosis
   ]
   ```
3. **Predict**: Random Forest outputs probability of unbalance
4. **Majority voting**: If >50% of 1-second windows predict unbalance, flag the entire time window

### Step 4: Detection and Visualization

When unbalance is detected, the system:

1. **Logs the event** (UTC timestamp, dataset, row indices, prediction scores)
2. **Generates a figure** with 4 panels:
   - Panel 1: Raw vibration signals (all 3 sensors)
   - Panel 2: RPM over time
   - Panel 3: Extracted features (7 features per 1-second window)
   - Panel 4: RFC predictions with threshold line
3. **Saves to disk**: `figures/detections/unbalance_detection_*.png`
4. **Updates performance metrics**: TP/FP/TN/FN tracking

### Step 5: Performance Tracking

The system continuously tracks:
- **True Positives (TP)**: Correctly detected unbalance in 1E-4E
- **True Negatives (TN)**: Correctly identified normal in 0E
- **False Positives (FP)**: Incorrectly flagged normal as unbalanced
- **False Negatives (FN)**: Missed unbalance in 1E-4E

Metrics are updated every N windows (configurable with `--log-interval`).

## Output Files

### 1. Detection Figures

**Location**: `figures/detections/unbalance_detection_*.png`

Each figure contains:
- Raw vibration signals (3 sensors)
- RPM profile
- Extracted features (7 per 1-second window)
- RFC predictions with detection threshold

**Example filename**:
```
unbalance_detection_3E_20251120_091530_row614400.png
```

### 2. Performance Report

**Location**: `figures/detections/performance_report_YYYYMMDD_HHMMSS.txt`

Contains:
- Per-dataset metrics (TP, FP, TN, FN, Accuracy, Precision, Recall)
- Overall metrics
- Timestamp and processing stats

**Example**:
```
Dataset    Total    TP       FP       TN       FN       Accuracy   Precision  Recall
-------------------------------------------------------------------------------------
0E         49       0        14       35       0        0.714      0.000      0.000
1E         1        1        0        0        0        1.000      1.000      1.000
2E         3        1        0        0        2        0.333      1.000      0.333
3E         2        2        0        0        0        1.000      1.000      1.000
4E         1        1        0        0        0        1.000      1.000      1.000
-------------------------------------------------------------------------------------
Overall    56       5        14       35       2        0.714      0.263      0.714
```

### 3. Detections Log (JSONL)

**Location**: `figures/detections/detections.jsonl`

JSON Lines format (one detection event per line) for MCP server integration.

**Example event**:
```json
{
  "timestamp": "2025-11-20 17:30:45",
  "window_idx": 42,
  "dataset": "3E",
  "dataset_name": "Unbalance Level 3",
  "start_idx": 614400,
  "end_idx": 655360,
  "detection_ratio": 0.883,
  "mean_prediction": 0.852,
  "max_prediction": 0.923,
  "figure_file": "unbalance_detection_3E_20251120_173045_row614400.png"
}
```

## MCP Server Integration

The Minimal RFC approach includes an **MCP (Model Context Protocol) server** for real-time monitoring.

### Features

The MCP server provides three tools:

1. **`get_performance_metrics()`**: Latest accuracy, precision, recall
2. **`get_recent_detections(limit=10)`**: Most recent unbalance events
3. **`get_system_status()`**: Overall system health

### Running with MCP Server

**Option 1: Integrated Mode** (Recommended)

Run detection with MCP server in background:
```bash
python approach_3_minimal_rfc.py --dataset all --enable-mcp
```

The MCP server will be available at:
- HTTP/SSE endpoint: `http://localhost:8000/sse`

**Option 2: Standalone Mode**

Run MCP server separately:
```bash
# Terminal 1: Run detection
python approach_3_minimal_rfc.py --dataset all

# Terminal 2: Run MCP server
python mcp_server.py
```

**Option 3: Claude Desktop Integration**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "minimal-rfc-monitor": {
      "command": "/path/to/venv_py311/bin/python",
      "args": ["/path/to/apps/approach_3_minimal_rfc/mcp_server.py"]
    }
  }
}
```

### MCP Server Queries

Once connected to Claude Desktop, you can ask:

- "What's the current accuracy of the minimal RFC detection system?"
- "Show me the latest unbalance detections"
- "Is the detection system running?"
- "How many false positives have occurred?"
- "What was the detection ratio for the last event?"

## Comparison with Other Approaches

| Feature | Approach 1 (CNN) | Approach 2 (FFT FCN) | Approach 3 (Minimal RFC) |
|---------|------------------|---------------------|--------------------------|
| **Input** | Raw signal (4096) | FFT spectrum (2048) | 7 statistical features |
| **Preprocessing** | None | FFT + RobustScaler | Feature extraction |
| **Model Type** | Deep Learning (CNN) | Deep Learning (FCN) | Classical ML (RFC) |
| **Model File** | `.h5` (TensorFlow) | `.h5` (TensorFlow) | `.joblib` (scikit-learn) |
| **Dependencies** | TensorFlow + numpy | TensorFlow + numpy | scikit-learn only |
| **Training Time** | Long (GPU beneficial) | Long (GPU beneficial) | Moderate (CPU-friendly) |
| **Inference Speed** | Fast (~1ms) | Fast (~1ms) | **Very Fast (<1ms)** |
| **Interpretability** | Low (black box) | Low (black box) | **High** (feature importance) |
| **Accuracy** | **99.6%** | **99.7%** | 81.8% |
| **Level 1 Detection** | Excellent | Excellent | **Poor (45%)** |
| **Model Size** | 12 MB | 60 MB | **4 MB** |
| **Memory Usage** | High | High | **Low** |

## When to Use This Approach

### ✅ **Good For:**

1. **Educational Purposes**
   - Learning classical ML approaches
   - Understanding feature engineering
   - Comparing ML vs. Deep Learning

2. **Resource-Constrained Environments**
   - Limited computational power
   - Cannot use TensorFlow/GPU
   - Need minimal memory footprint

3. **Interpretability Required**
   - Must explain predictions
   - Regulatory compliance
   - Safety-critical contexts
   - Feature importance analysis

4. **Strong Unbalance Detection Only**
   - Only need to detect severe unbalance (levels 3-4)
   - 99%+ accuracy on these levels
   - Acceptable to miss weak unbalance

5. **Real-time Embedded Systems**
   - Ultra-fast inference (<1ms)
   - Minimal dependencies
   - Small model size (4MB)

### ⚠️ **Not Good For:**

1. **Production/Critical Applications**
   - Need highest accuracy (99%+)
   - Use Approach 1 or 2 instead

2. **Weak Unbalance Detection**
   - Only 45% accuracy on level 1
   - Use Approach 1 or 2 instead

3. **All-Level Performance**
   - Need consistent accuracy across all levels
   - Use Approach 1 or 2 instead

## Limitations

1. **Lower Accuracy**: 82% mean vs 99% for deep learning
2. **Poor Level 1 Detection**: Only 45% accuracy on weak unbalance
3. **Hand-crafted Features**: May not capture all relevant patterns
4. **Fixed Feature Set**: Cannot adapt/learn new features

## Future Improvements

1. **Feature Engineering**:
   - Add frequency-domain features (peak frequencies)
   - Add time-domain features (zero-crossing rate)
   - Add cross-sensor features (correlation, phase difference)

2. **Model Ensemble**:
   - Combine RFC with other classifiers (SVM, Gradient Boosting)
   - Majority voting across multiple models

3. **Adaptive Thresholding**:
   - Different thresholds for different RPM ranges
   - Adjust threshold based on sensor noise levels

4. **Feature Selection**:
   - Analyze feature importance
   - Remove redundant features
   - Add domain-specific features

## Troubleshooting

### Model File Not Found

**Error**: `Model file not found at ../../models/reference/minimal_rfc.joblib`

**Solution**: Train the model:
```bash
cd ../..
python train_minimal_rfc.py
```

### Poor Detection Performance

**Issue**: Many false positives or false negatives

**Solutions**:
1. Check if using correct datasets (0E-4E for evaluation)
2. Adjust `--normal-weight` (default 0.9 may not match your use case)
3. Consider using Approach 1 or 2 for better accuracy

### MCP Server Not Starting

**Error**: `Could not start MCP server: No module named 'fastmcp'`

**Solution**: Install fastmcp:
```bash
pip install fastmcp
```

## References

- Notebook 4: `notebooks/3_Approach_3_TSFRESH_RFC.ipynb`
- Paper: "Machine Learning Based Unbalance Detection of a Rotating Shaft Using Vibration Data"
- Conference: IEEE ETFA 2020

## License

Same as parent project.

## Contact

See parent project README for contact information.
