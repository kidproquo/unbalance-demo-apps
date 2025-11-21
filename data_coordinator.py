#!/usr/bin/env python3
"""
Data Coordinator for Synchronized Unbalance Detection

Coordinates window processing across multiple detection approaches by:
1. Performing weighted random sampling of datasets
2. Publishing window selections to Redis Stream
3. Ensuring all approaches process the same windows

This enables direct comparison of CNN, FFT FCN, and Random Forest approaches
on identical data in real-time.
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Add current directory to path for utils
sys.path.insert(0, str(Path(__file__).parent))
from utils.redis_client import RedisConfig, WindowPublisher, test_redis_connection
from utils.data_utils import load_data, skip_warmup

# Sensor columns to extract: RPM and 3 vibration sensors (skip V_in at index 0)
SENSOR_COLUMNS = ['Measured_RPM', 'Vibration_1', 'Vibration_2', 'Vibration_3']


# Constants
SAMPLES_PER_SECOND = 4096


def calculate_available_windows(data_lengths: dict, time_window_seconds: int) -> dict:
    """
    Calculate how many windows are available in each dataset.

    Args:
        data_lengths: Dictionary of {label: total_samples}
        time_window_seconds: Duration of each window in seconds

    Returns:
        Dictionary of {label: num_windows}
    """
    time_window_samples = time_window_seconds * SAMPLES_PER_SECOND
    available_windows = {}

    for label, total_samples in data_lengths.items():
        if total_samples >= time_window_samples:
            n_windows = total_samples // time_window_samples
            available_windows[label] = n_windows

    return available_windows


def main():
    parser = argparse.ArgumentParser(
        description='Data Coordinator for Synchronized Unbalance Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Publish 100 windows
  python data_coordinator.py --max-windows 100

  # Continuous mode (infinite)
  python data_coordinator.py

  # Custom sampling weights
  python data_coordinator.py --normal-weight 0.8 --max-windows 500

  # Connect to remote Redis
  python data_coordinator.py --redis-host redis.example.com --redis-port 6379
        """)

    parser.add_argument('--max-windows', type=int, default=None,
                       help='Maximum number of windows to publish (default: infinite)')
    parser.add_argument('--time-window', type=int, default=10,
                       help='Time window duration in seconds (default: 10)')
    parser.add_argument('--normal-weight', type=float, default=0.9,
                       help='Probability of selecting normal (0E) data (default: 0.9)')
    parser.add_argument('--publish-rate', type=float, default=0,
                       help='Publish rate in windows/second (0=maximum speed) (default: 0)')

    parser.add_argument('--redis-host', type=str, default=None,
                       help='Redis host (default: localhost or REDIS_HOST env var)')
    parser.add_argument('--redis-port', type=int, default=None,
                       help='Redis port (default: 6379 or REDIS_PORT env var)')
    parser.add_argument('--redis-stream', type=str, default='windows',
                       help='Redis stream name (default: windows)')

    parser.add_argument('--data-url', type=str,
                       default='/app/data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip',
                       help='Path to dataset ZIP file')

    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print("Data Coordinator for Synchronized Unbalance Detection")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Time Window: {args.time_window} seconds")
    print(f"  Max Windows: {'Infinite (Ctrl+C to stop)' if args.max_windows is None else args.max_windows}")
    print(f"  Normal Weight (0E): {args.normal_weight*100:.0f}%")
    print(f"  Unbalanced Weight (1E-4E): {(1-args.normal_weight)*100:.0f}%")
    print(f"  Publish Rate: {'Maximum' if args.publish_rate == 0 else f'{args.publish_rate} windows/s'}")
    print(f"  Redis Stream: {args.redis_stream}")
    print()

    # Load dataset
    print("=" * 80)
    print("Loading Dataset")
    print("=" * 80)
    print(f"Data source: {args.data_url}")
    data = load_data(args.data_url, datasets_to_load='all')
    data = skip_warmup(data)

    # Get actual data lengths from loaded data
    data_lengths = {label: len(df) for label, df in data.items()}

    time_window_samples = args.time_window * SAMPLES_PER_SECOND

    # Calculate available windows per dataset
    available_windows = calculate_available_windows(data_lengths, args.time_window)

    print("Available Windows per Dataset:")
    for label, n_windows in sorted(available_windows.items()):
        duration_minutes = n_windows * args.time_window / 60
        print(f"  {label}: {n_windows:,} windows ({duration_minutes:.1f} minutes)")
    print()

    # Test Redis connection first
    print("=" * 80)
    print("Testing Redis Connection")
    print("=" * 80)
    redis_config = RedisConfig(host=args.redis_host, port=args.redis_port)
    if not test_redis_connection(redis_config):
        print("\nError: Cannot connect to Redis server")
        print(f"  Host: {redis_config.host}")
        print(f"  Port: {redis_config.port}")
        print("\nPlease ensure Redis is running:")
        print("  docker run -p 6379:6379 redis:7-alpine")
        print("Or check your --redis-host and --redis-port arguments")
        return
    print()

    # Initialize Redis publisher
    print("Connecting to Redis...")
    publisher = WindowPublisher(redis_config, stream_name=args.redis_stream)

    # Track dataset selection and rollover
    dataset_selection_counts = {label: 0 for label in available_windows.keys()}
    dataset_rollover_counts = {label: 0 for label in available_windows.keys()}
    dataset_current_window = {label: 0 for label in available_windows.keys()}

    # Publish windows
    num_windows_to_publish = float('inf') if args.max_windows is None else args.max_windows

    if args.max_windows is None:
        print(f"Starting continuous publishing (Ctrl+C to stop)...")
    else:
        print(f"Publishing {args.max_windows} windows...")
    print()

    start_time = time.time()
    window_idx = 0

    try:
        while window_idx < num_windows_to_publish:
            # Weighted random selection
            if np.random.random() < args.normal_weight:
                # Select normal (0E)
                selected_label = '0E'
            else:
                # Select randomly from unbalanced datasets (1E-4E)
                unbalanced_labels = [l for l in available_windows.keys() if l != '0E']
                selected_label = np.random.choice(unbalanced_labels)

            dataset_selection_counts[selected_label] += 1

            # Get current window index for this dataset
            current_window = dataset_current_window[selected_label]

            # Check if we need to rollover
            if current_window >= available_windows[selected_label]:
                current_window = 0
                dataset_rollover_counts[selected_label] += 1
                if args.max_windows is None:  # Only print rollover in infinite mode
                    print(f"  🔄 Dataset {selected_label} rolled over to beginning")

            # Calculate row indices
            start_idx = current_window * time_window_samples
            end_idx = start_idx + time_window_samples

            # Extract sensor data for this window
            # Columns: RPM, Vibration_1, Vibration_2, Vibration_3
            sensor_data = data[selected_label][SENSOR_COLUMNS].iloc[start_idx:end_idx].values

            # Update current window for next selection
            dataset_current_window[selected_label] = current_window + 1

            # Publish to Redis with sensor data
            message_id = publisher.publish_window(
                dataset=selected_label,
                window_idx=current_window,
                start_idx=start_idx,
                end_idx=end_idx,
                sensor_data=sensor_data
            )

            # Log progress
            if (window_idx + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                rate = (window_idx + 1) / elapsed_time if elapsed_time > 0 else 0
                stream_length = publisher.get_stream_length()
                print(f"Published {window_idx + 1} windows "
                      f"(Rate: {rate:.1f} windows/s, Queue: {stream_length})")

            window_idx += 1

            # Rate limiting
            if args.publish_rate > 0:
                time.sleep(1.0 / args.publish_rate)

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Publishing interrupted by user (Ctrl+C)")
        print(f"Published {window_idx} windows before interruption")

    # Final summary
    total_time = time.time() - start_time
    avg_rate = window_idx / total_time if total_time > 0 else 0

    print(f"\n{'='*80}")
    print(f"Publishing Complete")
    print(f"{'='*80}")
    print(f"Total windows published: {window_idx}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average rate: {avg_rate:.1f} windows/second")

    print(f"\nDataset Selection Statistics:")
    total_selections = sum(dataset_selection_counts.values())
    if total_selections > 0:
        for label in sorted(dataset_selection_counts.keys()):
            count = dataset_selection_counts[label]
            percentage = (count / total_selections) * 100
            print(f"  {label}: {count} times ({percentage:.1f}%)")

    # Print rollover statistics (only if any rollovers occurred)
    if any(count > 0 for count in dataset_rollover_counts.values()):
        print(f"\nDataset Rollover Statistics:")
        for label in sorted(dataset_rollover_counts.keys()):
            if dataset_rollover_counts[label] > 0:
                print(f"  {label}: {dataset_rollover_counts[label]} rollover(s)")

    print(f"\nStream status:")
    print(f"  Stream name: {args.redis_stream}")
    print(f"  Messages in stream: {publisher.get_stream_length()}")
    print(f"  ✓ Ready for consumers to process")
    print(f"{'='*80}")

    publisher.close()


if __name__ == "__main__":
    main()
