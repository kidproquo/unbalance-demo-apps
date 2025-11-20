# Migration Guide: Shared Data Utilities

## Overview

The data loading and generation logic has been extracted into a shared `utils` module that can be used across all approaches (CNN, FFT, Random Forest, HMM).

**Location:** `apps/utils/`

## What Was Changed

### New Files Created

```
apps/utils/
├── __init__.py           # Package initialization
├── data_utils.py         # Core data utilities
├── README.md             # Documentation
├── example_usage.py      # Usage examples
└── MIGRATION_GUIDE.md    # This file
```

### Updated Files

- `apps/approach_1_cnn/approach_1_cnn.py`: Now imports from `utils.data_utils`

## What Was Extracted

### Functions

1. **`load_data(url, datasets_to_load='all')`**
   - Loads datasets from ZIP file
   - Supports selective loading (all, specific dataset, list of datasets)
   - Returns dictionary of DataFrames

2. **`skip_warmup(data, skip=50000)`**
   - Removes warm-up phase samples
   - Returns cleaned dataset dictionary

### Classes

3. **`DataGenerator`**
   - Weighted random sampling (90% normal, 10% unbalanced by default)
   - Automatic rollover when datasets are exhausted
   - Sequential reading within each dataset
   - Statistics tracking

### Constants

- `SAMPLES_PER_SECOND = 4096`
- `SKIP_WARMUP = 50000`
- `DEFAULT_SENSOR = 'Vibration_1'`

## How to Use in Other Apps

### Step 1: Add Import Path

```python
import sys
from pathlib import Path

# Add apps directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Step 2: Import Utilities

```python
from utils.data_utils import (
    load_data,
    skip_warmup,
    DataGenerator,
    SAMPLES_PER_SECOND,
    SKIP_WARMUP,
    DEFAULT_SENSOR
)
```

### Step 3: Use in Your Code

```python
# Load data
data = load_data('../../data/dataset.zip', datasets_to_load='all')
data = skip_warmup(data)

# Use DataGenerator for weighted sampling
generator = DataGenerator(
    data=data,
    time_window_seconds=60,
    normal_weight=0.9
)

for window_info in generator.generate_windows(max_windows=100):
    # Your processing code
    process_window(window_info['data'])
```

## Example: Migrating Approach 2 (FFT)

Here's how you would use the shared utilities in `apps/approach_2_fft/`:

```python
# apps/approach_2_fft/approach_2_fft.py

import sys
from pathlib import Path
import numpy as np

# Add apps directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared utilities
from utils.data_utils import (
    load_data,
    skip_warmup,
    DataGenerator,
    SAMPLES_PER_SECOND
)

# FFT-specific constants
FFT_WINDOW = 4096

def main():
    # Load data using shared utility
    data = load_data('../../data/dataset.zip')
    data = skip_warmup(data)

    # Use DataGenerator
    generator = DataGenerator(
        data=data,
        time_window_seconds=60,
        normal_weight=0.9
    )

    # Process windows with FFT
    for window_info in generator.generate_windows():
        window_data = window_info['data']

        # Your FFT processing here
        fft_result = np.fft.rfft(window_data)
        # ... detection logic ...

        if window_info['window_idx'] >= 100:
            break

    generator.print_statistics()

if __name__ == '__main__':
    main()
```

## Benefits

### Code Reuse
- No duplication of data loading logic across approaches
- Consistent behavior across all methods
- Single source of truth for data handling

### Maintainability
- Bug fixes and improvements benefit all approaches
- Easier to add new datasets or formats
- Centralized documentation

### Flexibility
- DataGenerator supports configurable weights
- Easy to extend with new features
- Compatible with all analysis approaches

## Backward Compatibility

The `approach_1_cnn.py` has been updated to use the shared module, but maintains the same external API and behavior. No changes are needed to how you run the script.

## Testing

Run the example script to verify everything works:

```bash
cd utils
python example_usage.py
```

Or run the CNN approach as before:

```bash
cd apps/approach_1_cnn
python approach_1_cnn.py --dataset all --speed 0
```

## Future Work

The shared utilities can be extended with:

1. Support for training datasets (0D-4D)
2. Different sampling strategies
3. Data augmentation utilities
4. Feature extraction helpers
5. Common preprocessing functions

## Questions?

See `utils/README.md` for detailed API documentation and more examples.
