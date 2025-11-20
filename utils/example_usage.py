#!/usr/bin/env python3
"""
Example usage of the data_utils module.

This script demonstrates how to use the shared data utilities
for loading data and generating windows with weighted random sampling.
"""

import sys
from pathlib import Path

# Add apps directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_utils import load_data, skip_warmup, DataGenerator


def example_basic_loading():
    """Example: Basic data loading."""
    print("="*80)
    print("Example 1: Basic Data Loading")
    print("="*80)

    # Path to dataset
    data_path = '../../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'

    # Load all evaluation datasets
    print("\n1. Loading all evaluation datasets...")
    data = load_data(data_path, datasets_to_load='all')

    # Skip warm-up phase
    data = skip_warmup(data, skip=50000)

    print(f"\nLoaded {len(data)} datasets")
    for label, df in data.items():
        print(f"  {label}: {len(df)} samples ({len(df)/4096/60:.1f} minutes)")


def example_specific_loading():
    """Example: Loading specific datasets."""
    print("\n" + "="*80)
    print("Example 2: Loading Specific Datasets")
    print("="*80)

    data_path = '../../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'

    # Load only dataset 4E
    print("\n1. Loading only dataset 4E...")
    data = load_data(data_path, datasets_to_load='4E')
    data = skip_warmup(data)

    # Load multiple specific datasets
    print("\n2. Loading datasets 0E, 1E, and 4E...")
    data = load_data(data_path, datasets_to_load=['0E', '1E', '4E'])
    data = skip_warmup(data)

    print(f"\nLoaded {len(data)} datasets")


def example_data_generator():
    """Example: Using DataGenerator for weighted random sampling."""
    print("\n" + "="*80)
    print("Example 3: Using DataGenerator")
    print("="*80)

    data_path = '../../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'

    # Load and prepare data
    print("\nLoading data...")
    data = load_data(data_path, datasets_to_load='all')
    data = skip_warmup(data)

    # Create generator with 90% normal weight
    print("\nCreating DataGenerator (90% normal, 10% unbalanced)...")
    generator = DataGenerator(
        data=data,
        time_window_seconds=60,
        normal_weight=0.9,
        sensor='Vibration_1'
    )

    # Generate 10 windows
    print("\nGenerating 10 windows...")
    for window_info in generator.generate_windows(max_windows=10):
        print(f"  Window {window_info['window_idx']:2d}: "
              f"{window_info['label']} ({window_info['name']:20s}) | "
              f"Samples {window_info['start_idx']:8,} - {window_info['end_idx']:8,} | "
              f"Data shape: {window_info['data'].shape}")

    # Print statistics
    print("\nGeneration statistics:")
    generator.print_statistics()


def example_continuous_generation():
    """Example: Continuous generation with rollover."""
    print("\n" + "="*80)
    print("Example 4: Continuous Generation with Rollover")
    print("="*80)

    data_path = '../../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'

    # Load data
    print("\nLoading data...")
    data = load_data(data_path, datasets_to_load='all')
    data = skip_warmup(data)

    # Create generator
    print("\nCreating DataGenerator...")
    generator = DataGenerator(
        data=data,
        time_window_seconds=60,
        normal_weight=0.9
    )

    # Generate windows until we see a rollover
    print("\nGenerating windows until rollover occurs...")
    print("(This may take a while - limiting to 2000 windows for demo)")

    try:
        for window_info in generator.generate_windows(max_windows=2000):
            # Check for rollovers
            stats = generator.get_statistics()
            total_rollovers = sum(stats['rollover_counts'].values())

            if total_rollovers > 0:
                print(f"\n✓ Rollover detected at window {window_info['window_idx']}!")
                print(f"  Dataset: {window_info['label']}")
                print(f"\nRollover counts:")
                for label, count in stats['rollover_counts'].items():
                    if count > 0:
                        print(f"  {label}: {count} rollover(s)")
                break

            # Print progress every 100 windows
            if window_info['window_idx'] % 100 == 0 and window_info['window_idx'] > 0:
                print(f"  Processed {window_info['window_idx']} windows...")

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")

    # Final statistics
    generator.print_statistics()


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Data Utils Module - Usage Examples")
    print("="*80)

    # Check if data file exists
    data_path = Path(__file__).parent.parent / 'data' / 'fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'

    if not data_path.exists():
        print(f"\n⚠️  Warning: Dataset not found at {data_path}")
        print("Please download the dataset first.")
        print("\nThese examples demonstrate the API, but won't run without the dataset.")
        return

    # Run examples
    example_basic_loading()
    example_specific_loading()
    example_data_generator()
    example_continuous_generation()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
