# Shared Utilities Module

This module provides shared utilities for loading and processing vibration sensor data across different approaches (CNN, FFT, Random Forest, HMM).

**Location:** `apps/utils/`

## Installation

No additional installation required. The module uses standard dependencies already available in the project.

## Usage

### Basic Data Loading

```python
import sys
from pathlib import Path

# Add apps directory to path (from within an app like approach_1_cnn)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_utils import load_data, skip_warmup

# Load all evaluation datasets
data = load_data('../../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip')

# Or load specific datasets
data = load_data('path/to/data.zip', datasets_to_load='4E')
data = load_data('path/to/data.zip', datasets_to_load=['0E', '1E', '4E'])

# Skip warm-up phase
data_clean = skip_warmup(data, skip=50000)
```

### Using the DataGenerator

The `DataGenerator` class provides weighted random sampling with automatic rollover:

```python
from utils.data_utils import load_data, skip_warmup, DataGenerator

# Load and prepare data
data = load_data('path/to/data.zip')
data = skip_warmup(data)

# Create generator
generator = DataGenerator(
    data=data,
    time_window_seconds=60,      # 60-second windows
    normal_weight=0.9,            # 90% normal, 10% unbalanced
    sensor='Vibration_1',
    samples_per_second=4096
)

# Generate windows (infinite by default)
for window_info in generator.generate_windows(max_windows=100):
    window_data = window_info['data']         # numpy array of sensor data
    source_label = window_info['label']       # e.g., '0E', '4E'
    source_name = window_info['name']         # e.g., 'No Unbalance'
    start_idx = window_info['start_idx']      # starting sample index
    end_idx = window_info['end_idx']          # ending sample index
    window_idx = window_info['window_idx']    # sequential window number

    # Process the window data...
    # Your detection/analysis code here

# Print statistics
generator.print_statistics()
```

### Example: CNN Detection

```python
from utils.data_utils import load_data, skip_warmup, DataGenerator
import numpy as np

# Load data
data = load_data('../../data/dataset.zip', datasets_to_load='all')
data = skip_warmup(data)

# Create generator with 90% normal weight
generator = DataGenerator(
    data=data,
    time_window_seconds=60,
    normal_weight=0.9
)

# Process windows
detections = 0
try:
    for window_info in generator.generate_windows():
        # Split into 1-second sub-windows
        window_data = window_info['data']
        n_seconds = len(window_data) // 4096

        # Your model prediction here
        # predictions = model.predict(...)

        # if unbalance_detected:
        #     detections += 1
        #     print(f"Unbalance in {window_info['label']} at {window_info['start_idx']}")

        # Stop after processing some windows
        if window_info['window_idx'] >= 1000:
            break

except KeyboardInterrupt:
    print("\nStopped by user")

# Show statistics
generator.print_statistics()
```

## API Reference

### Functions

#### `load_data(url, datasets_to_load='all')`

Load measurement data from ZIP file.

**Parameters:**
- `url` (str): Path to the ZIP file containing the dataset
- `datasets_to_load` (str or list): Which datasets to load
  - `'all'`: Load all evaluation datasets (0E-4E)
  - `'4E'`: Load specific dataset
  - `['0E', '1E', '4E']`: Load list of datasets

**Returns:**
- Dictionary with dataset labels as keys and DataFrames as values

#### `skip_warmup(data, skip=50000)`

Skip warm-up phase of measurements.

**Parameters:**
- `data` (dict): Dictionary of dataframes from `load_data()`
- `skip` (int): Number of samples to skip (default: 50000)

**Returns:**
- Dictionary of dataframes with warm-up phase removed

### DataGenerator Class

#### `__init__(data, time_window_seconds=60, normal_weight=0.9, sensor='Vibration_1', samples_per_second=4096)`

Initialize the data generator.

**Parameters:**
- `data` (dict): Dictionary of datasets from `load_data()` + `skip_warmup()`
- `time_window_seconds` (int): Time window size in seconds (default: 60)
- `normal_weight` (float): Weight for 0E dataset, 0.0-1.0 (default: 0.9)
- `sensor` (str): Sensor column name (default: 'Vibration_1')
- `samples_per_second` (int): Sampling rate in Hz (default: 4096)

#### `generate_windows(max_windows=None)`

Generate windows indefinitely or up to max_windows.

**Parameters:**
- `max_windows` (int or None): Maximum windows to generate (None = infinite)

**Yields:**
- Dictionary with keys: `data`, `label`, `name`, `start_idx`, `end_idx`, `window_idx`

#### `get_statistics()`

Get generation statistics as a dictionary.

#### `print_statistics()`

Print generation statistics to console.

#### `reset()`

Reset all positions and statistics.

## Constants

- `SAMPLES_PER_SECOND = 4096`: Sampling rate in Hz
- `SKIP_WARMUP = 50000`: Default samples to skip for warm-up
- `DEFAULT_SENSOR = 'Vibration_1'`: Default sensor column name

## Features

### Weighted Random Sampling

The DataGenerator implements weighted random sampling to simulate realistic operating conditions:
- Configurable weight for normal (0E) vs. unbalanced (1E-4E) data
- Default 90% normal, 10% unbalanced
- Random selection among unbalanced datasets (1E-4E)

### Automatic Rollover

When a dataset is exhausted:
- Position automatically resets to 0
- Generation continues seamlessly
- Rollover count is tracked for statistics

### Sequential Reading

Within each dataset:
- Windows are read sequentially (not randomly)
- Position advances after each read
- Maintains realistic temporal patterns

## Integration with Existing Apps

### Approach 1: CNN

The CNN approach (`apps/approach_1_cnn/`) already uses this module.

### Future Approaches

Other approaches can import and use these utilities:

```python
# In apps/approach_2_fft/approach_2_fft.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.data_utils import load_data, skip_warmup, DataGenerator
```

## Notes

- All timestamps and labels are preserved from the original data
- Memory efficient - only loads requested datasets
- Thread-safe for single-threaded use (not designed for multi-threading)
- Generator state can be reset with `reset()` method
