"""
Data Loading and Generation Utilities for Unbalance Detection

This module provides shared utilities for loading and processing vibration sensor data
across different approaches (CNN, FFT, Random Forest, HMM).
"""

import pandas as pd
import numpy as np
import zipfile


# Constants
SAMPLES_PER_SECOND = 4096  # Sampling rate in Hz
SKIP_WARMUP = 50000  # Number of samples to skip (warm-up phase)
DEFAULT_SENSOR = 'Vibration_1'  # Default sensor to use


def load_data(url, datasets_to_load='all'):
    """
    Load measurement data from ZIP file.

    Args:
        url: Path to the ZIP file containing the dataset
        datasets_to_load: Which datasets to load - 'all', or specific like '4E', '0E', etc.
                         Can also be a list of dataset labels like ['0E', '4E']

    Returns:
        Dictionary containing requested datasets (key=label, value=DataFrame)

    Example:
        >>> data = load_data('path/to/data.zip', datasets_to_load='4E')
        >>> data = load_data('path/to/data.zip', datasets_to_load=['0E', '1E'])
        >>> data = load_data('path/to/data.zip')  # loads all evaluation datasets
    """
    print("Loading measurement data...")
    data = {}

    # Determine which datasets to load
    if isinstance(datasets_to_load, list):
        # List of specific datasets provided
        labels_to_load = [label.upper() for label in datasets_to_load]
    elif datasets_to_load.upper() == 'ALL':
        # Load all evaluation datasets for validation and processing
        labels_to_load = ['0E', '1E', '2E', '3E', '4E']
    elif datasets_to_load.upper() in ['0E', '1E', '2E', '3E', '4E']:
        # Load only the specific evaluation dataset requested
        labels_to_load = [datasets_to_load.upper()]
    else:
        # Default: load all evaluation datasets
        labels_to_load = ['0E', '1E', '2E', '3E', '4E']

    with zipfile.ZipFile(url, 'r') as f:
        for label in labels_to_load:
            with f.open(f'{label}.csv', 'r') as c:
                data[label] = pd.read_csv(c)
                print(f"  Loaded {label}.csv: {len(data[label])} rows")

    return data


def skip_warmup(data, skip=SKIP_WARMUP):
    """
    Skip warm-up phase of each measurement.

    The first samples are noisy due to the warm-up phase of the measuring device,
    so they should be skipped.

    Args:
        data: Dictionary of dataframes
        skip: Number of samples to skip (default: 50000)

    Returns:
        Dictionary of dataframes with warm-up phase removed

    Example:
        >>> data = load_data('path/to/data.zip')
        >>> data_clean = skip_warmup(data, skip=50000)
    """
    print(f"\nSkipping first {skip} samples (warm-up phase)...")
    processed_data = {}

    for label, df in data.items():
        processed_data[label] = df.iloc[skip:, :]
        print(f"  {label}: {len(processed_data[label])} rows remaining")

    return processed_data


class DataGenerator:
    """
    Generator for weighted random sampling from multiple datasets.

    This class enables continuous data generation with configurable weights,
    automatic rollover when datasets are exhausted, and sequential reading
    within each dataset.

    Example:
        >>> data = load_data('path/to/data.zip')
        >>> data = skip_warmup(data)
        >>>
        >>> generator = DataGenerator(
        ...     data=data,
        ...     time_window_seconds=60,
        ...     normal_weight=0.9,
        ...     sensor='Vibration_1'
        ... )
        >>>
        >>> # Generate infinite windows
        >>> for window_info in generator.generate_windows():
        ...     window_data = window_info['data']
        ...     source_label = window_info['label']
        ...     start_idx = window_info['start_idx']
        ...     # Process window_data...
        ...     if some_condition:
        ...         break  # Stop generation
    """

    def __init__(self, data, time_window_seconds=60, normal_weight=0.9,
                 sensor=DEFAULT_SENSOR, samples_per_second=SAMPLES_PER_SECOND):
        """
        Initialize the data generator.

        Args:
            data: Dictionary of datasets (output from load_data + skip_warmup)
            time_window_seconds: Time window size in seconds (default: 60)
            normal_weight: Weight for 0E (normal) dataset, 0.0-1.0 (default: 0.9)
            sensor: Name of the sensor column to extract (default: 'Vibration_1')
            samples_per_second: Sampling rate in Hz (default: 4096)
        """
        self.data = data
        self.time_window_seconds = time_window_seconds
        self.normal_weight = normal_weight
        self.sensor = sensor
        self.samples_per_second = samples_per_second

        # Calculate time window in samples
        self.time_window_samples = int(time_window_seconds * samples_per_second)

        # Calculate available windows in each dataset
        self.dataset_windows = {}
        for label in ['0E', '1E', '2E', '3E', '4E']:
            if label in data:
                total_samples = len(data[label][sensor].values)
                n_windows = int(np.floor(total_samples / self.time_window_samples))
                self.dataset_windows[label] = n_windows

        # Initialize position tracking
        self.dataset_positions = {label: 0 for label in self.dataset_windows.keys()}

        # Statistics tracking
        self.dataset_selection_counts = {label: 0 for label in ['0E', '1E', '2E', '3E', '4E']}
        self.dataset_rollover_counts = {label: 0 for label in ['0E', '1E', '2E', '3E', '4E']}
        self.total_windows_generated = 0

    def _select_dataset(self):
        """
        Select a dataset based on weighted random sampling.

        Returns:
            Tuple of (label, name)
        """
        rand_val = np.random.random()

        if rand_val < self.normal_weight and '0E' in self.data:
            # Select 0E (no unbalance)
            selected_label = '0E'
            selected_name = 'No Unbalance'
        else:
            # Select randomly from 1E-4E (unbalanced datasets)
            unbalanced_labels = [label for label in ['1E', '2E', '3E', '4E'] if label in self.data]
            if not unbalanced_labels:
                # If no unbalanced data available, use 0E
                selected_label = '0E'
                selected_name = 'No Unbalance'
            else:
                selected_label = np.random.choice(unbalanced_labels)
                level = ['1E', '2E', '3E', '4E'].index(selected_label) + 1
                selected_name = f'Unbalance Level {level}'

        return selected_label, selected_name

    def _get_window_data(self, label):
        """
        Get the next window from the specified dataset with rollover support.

        Args:
            label: Dataset label (e.g., '0E', '4E')

        Returns:
            Tuple of (window_data, start_idx, end_idx)
        """
        # Get current position
        current_pos = self.dataset_positions[label]

        # Check for rollover
        if current_pos >= self.dataset_windows[label]:
            current_pos = 0
            self.dataset_positions[label] = 0
            self.dataset_rollover_counts[label] += 1

        # Extract window data
        sensor_data = self.data[label][self.sensor].values
        start_idx = current_pos * self.time_window_samples
        end_idx = start_idx + self.time_window_samples
        window_data = sensor_data[start_idx:end_idx]

        # Move to next window
        self.dataset_positions[label] += 1

        return window_data, start_idx, end_idx

    def generate_windows(self, max_windows=None):
        """
        Generate windows indefinitely (or up to max_windows).

        Args:
            max_windows: Maximum number of windows to generate (None = infinite)

        Yields:
            Dictionary with keys:
                - 'data': numpy array of window data
                - 'label': source dataset label (e.g., '0E')
                - 'name': human-readable name (e.g., 'No Unbalance')
                - 'start_idx': starting sample index in source dataset
                - 'end_idx': ending sample index in source dataset
                - 'window_idx': sequential window number (0-indexed)
        """
        num_windows_to_generate = float('inf') if max_windows is None else max_windows
        window_idx = 0

        while window_idx < num_windows_to_generate:
            # Select dataset
            selected_label, selected_name = self._select_dataset()
            self.dataset_selection_counts[selected_label] += 1

            # Get window data
            window_data, start_idx, end_idx = self._get_window_data(selected_label)

            # Update counter
            self.total_windows_generated += 1

            # Yield window info
            yield {
                'data': window_data,
                'label': selected_label,
                'name': selected_name,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'window_idx': window_idx
            }

            window_idx += 1

    def get_statistics(self):
        """
        Get generation statistics.

        Returns:
            Dictionary with statistics about dataset selection and rollovers
        """
        stats = {
            'total_windows_generated': self.total_windows_generated,
            'selection_counts': self.dataset_selection_counts.copy(),
            'rollover_counts': self.dataset_rollover_counts.copy(),
            'available_windows': self.dataset_windows.copy()
        }
        return stats

    def print_statistics(self):
        """
        Print generation statistics to console.
        """
        print(f"\n{'='*80}")
        print(f"Data Generation Statistics")
        print(f"{'='*80}")
        print(f"Total windows generated: {self.total_windows_generated}")
        print(f"\nDataset Selection Statistics:")
        for label in ['0E', '1E', '2E', '3E', '4E']:
            if label in self.dataset_selection_counts and self.dataset_selection_counts[label] > 0:
                percentage = (self.dataset_selection_counts[label] / self.total_windows_generated) * 100 if self.total_windows_generated > 0 else 0
                print(f"  {label}: {self.dataset_selection_counts[label]} times ({percentage:.1f}%)")

        total_rollovers = sum(self.dataset_rollover_counts.values())
        if total_rollovers > 0:
            print(f"\nDataset Rollover Statistics:")
            for label in ['0E', '1E', '2E', '3E', '4E']:
                if self.dataset_rollover_counts[label] > 0:
                    print(f"  {label}: {self.dataset_rollover_counts[label]} rollover(s)")
        print(f"{'='*80}")

    def reset(self):
        """
        Reset all positions and statistics.
        """
        self.dataset_positions = {label: 0 for label in self.dataset_windows.keys()}
        self.dataset_selection_counts = {label: 0 for label in ['0E', '1E', '2E', '3E', '4E']}
        self.dataset_rollover_counts = {label: 0 for label in ['0E', '1E', '2E', '3E', '4E']}
        self.total_windows_generated = 0
