#!/usr/bin/env python3
"""
Machine Learning Based Unbalance Detection of a Rotating Shaft Using Vibration Data
Notebook 4 of 5 - Approach 3: Random Forest Classifier on Minimal Statistical Features

Oliver Mey, Willi Neudeck, André Schneider and Olaf Enge-Rosenblatt
Fraunhofer IIS/EAS, Fraunhofer Institute for Integrated Circuits,
Division Engineering of Adaptive Systems, Dresden, Germany

This Python script is converted from the Jupyter Notebook that is part of a paper
submission to the 25th IEEE International Conference on Emerging Technologies and
Factory Automation, ETFA 2020.

Modified to:
1. Use pre-trained Random Forest Classifier model
2. Process evaluation data as timeseries with configurable time windows
3. Detect unbalance anomalies in real-time
4. Generate and save figures for detected unbalances
5. Extract minimal statistical features (7 features: RPM mean + sensor std/kurtosis)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import os
import argparse
import time
import json
from datetime import datetime, timezone
import scipy.stats
from joblib import load

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_utils import load_data, SKIP_WARMUP, SAMPLES_PER_SECOND
from utils.redis_client import RedisConfig, WindowConsumer, test_redis_connection

# Constants
WINDOW = 4096  # Samples per second (1 second windows for feature extraction)
UNBALANCE_THRESHOLD = 0.5  # Prediction threshold (RFC outputs 0 or 1, but we use predict_proba)

# Dataset names for display
DATASET_NAMES = {
    '0E': 'No Unbalance',
    '1E': 'Unbalance Level 1',
    '2E': 'Unbalance Level 2',
    '3E': 'Unbalance Level 3',
    '4E': 'Unbalance Level 4'
}


def extract_minimal_features(window_data):
    """
    Extract 7 minimal statistical features from a time window.

    Features:
    1. Mean of RPM
    2-4. Standard deviation of each vibration sensor (Vibration_1, 2, 3)
    5-7. Kurtosis of each vibration sensor

    Args:
        window_data: Array of shape (n_samples, 3) with columns [Vibration_1, Vibration_2, Vibration_3]

    Returns:
        Array of shape (7,) containing the extracted features
    """
    features = np.array([
        np.mean(window_data[:, 0]),                    # Sensor 1 mean
        np.std(window_data[:, 0]),                      # Sensor 1 std
        np.std(window_data[:, 1]),                      # Sensor 2 std
        np.std(window_data[:, 2]),                      # Sensor 3 std
        scipy.stats.kurtosis(window_data[:, 0]),        # Sensor 1 kurtosis
        scipy.stats.kurtosis(window_data[:, 1]),        # Sensor 2 kurtosis
        scipy.stats.kurtosis(window_data[:, 2])         # Sensor 3 kurtosis
    ])
    return features


def prepare_datasets_minimal(data):
    """
    Prepare validation datasets with minimal feature extraction.

    No separate training set needed since we're using a pre-trained model.
    Just verifies the data can be loaded properly.

    Args:
        data: Dictionary of {label: DataFrame} from load_all_datasets()

    Returns:
        data: Original data dictionary (unchanged)
    """
    print("\nValidating datasets...")
    for label, df in data.items():
        n_samples = len(df)
        n_windows = n_samples // WINDOW
        print(f"  {label}: {n_samples:,} samples = {n_windows:,} windows ({n_windows} seconds)")

    print(f"\nMinimal features will be extracted in real-time:")
    print(f"  - 1x RPM mean")
    print(f"  - 3x Vibration sensor standard deviations")
    print(f"  - 3x Vibration sensor kurtosis values")
    print(f"  - Total: 7 features per window")

    return data


def process_with_weighted_sampling(model, data, output_dir, speed, time_window_seconds,
                                     max_windows, normal_weight, log_interval):
    """
    Process datasets using weighted random sampling with minimal feature extraction.

    Similar to approaches 1 and 2, but extracts 7 statistical features instead of
    using raw signals or FFT.

    Args:
        model: Pre-trained Random Forest Classifier
        data: Dictionary of {label: DataFrame}
        output_dir: Directory to save detection figures
        speed: Playback speed multiplier (0 = maximum speed)
        time_window_seconds: Duration of each analysis window in seconds
        max_windows: Maximum number of windows to process (None = infinite)
        normal_weight: Probability of selecting normal (0E) data
        log_interval: How often to print performance metrics (in windows)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate samples per time window
    time_window_samples = time_window_seconds * SAMPLES_PER_SECOND

    # Calculate how many windows available in each dataset
    available_windows = {}
    for label, df in data.items():
        total_samples = len(df)
        if total_samples >= time_window_samples:
            n_windows = total_samples // time_window_samples
            print(f"{label}: {n_windows} windows available ({n_windows * time_window_seconds / 60:.1f} minutes)")
            available_windows[label] = n_windows
        else:
            print(f"{label}: Skipped (insufficient data)")

    if not available_windows:
        print("No data available to process")
        return

    # Setup weighted random sampling
    print(f"\nWeighted Random Sampling Configuration:")
    print(f"  Normal (0E) weight: {normal_weight*100:.0f}%")
    print(f"  Unbalanced (1E-4E) weight: {(1-normal_weight)*100:.0f}%")

    # Determine how many windows to process
    if max_windows is None:
        print(f"\nContinuous processing mode (infinite with rollover)")
        print(f"Press Ctrl+C to stop")
        num_windows_to_process = float('inf')
    else:
        num_windows_to_process = max_windows
        print(f"\nProcessing {max_windows} windows ({time_window_seconds}s each, {max_windows * time_window_seconds / 60:.1f} minutes total) with weighted random sampling...")
    print()

    # Create timestamped report filename
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    report_filename = os.path.join(output_dir, f"performance_report_{timestamp}.txt")

    # Track performance metrics per dataset
    dataset_stats = {label: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'total': 0}
                     for label in available_windows.keys()}

    # Track dataset selection and rollover counts
    dataset_selection_counts = {label: 0 for label in available_windows.keys()}
    dataset_rollover_counts = {label: 0 for label in available_windows.keys()}
    dataset_current_window = {label: 0 for label in available_windows.keys()}

    # Track total detections
    total_detections = 0

    # Performance tracking
    start_time = time.time()

    # Log file for detections (JSONL format for MCP server)
    detections_log = os.path.join(output_dir, "detections.jsonl")

    print(f"Performance report: {report_filename}")
    print(f"Updating report every {log_interval} windows")
    print(f"  (Time interval: {log_interval * time_window_seconds} seconds)\n")

    try:
        window_idx = 0
        while window_idx < num_windows_to_process:
            # Weighted random selection
            if np.random.random() < normal_weight:
                # Select normal (0E)
                selected_label = '0E'
            else:
                # Select randomly from unbalanced datasets (1E-4E)
                unbalanced_labels = [l for l in available_windows.keys() if l != '0E']
                selected_label = np.random.choice(unbalanced_labels)

            dataset_selection_counts[selected_label] += 1
            selected_name = DATASET_NAMES.get(selected_label, selected_label)

            # Get current window index for this dataset
            current_window = dataset_current_window[selected_label]

            # Check if we need to rollover
            if current_window >= available_windows[selected_label]:
                current_window = 0
                dataset_rollover_counts[selected_label] += 1
                if max_windows is None:  # Only print rollover message in infinite mode
                    print(f"\n  🔄 Dataset {selected_label} rolled over to beginning")

            # Extract the time window
            start_idx = current_window * time_window_samples
            end_idx = start_idx + time_window_samples
            # Select columns: Measured_RPM (index 1) and Vibration_1/2/3 (indices 2-4)
            # Skip V_in (index 0)
            window_data = data[selected_label].iloc[start_idx:end_idx, 1:].values

            # Update current window for next selection
            dataset_current_window[selected_label] = current_window + 1

            # Extract 1-second windows for feature extraction
            # Each time window contains multiple 1-second windows
            n_second_windows = len(window_data) // WINDOW
            window_data_windowed = window_data[:n_second_windows * WINDOW].reshape(n_second_windows, WINDOW, 3)

            # Extract minimal features for each 1-second window
            window_features = np.array([extract_minimal_features(w) for w in window_data_windowed])

            # Predict on all 1-second windows
            # RFC outputs probabilities via predict_proba
            predictions = model.predict_proba(window_features)[:, 1]  # Probability of class 1 (unbalance)

            # Detection logic: majority voting
            # If >50% of 1-second windows predict unbalance, flag the entire time window
            detection_ratio = np.mean(predictions > UNBALANCE_THRESHOLD)

            # Determine if unbalance detected
            if detection_ratio > 0.5:
                total_detections += 1

                # Get current UTC time
                current_time = datetime.now(timezone.utc)

                print(f"\n  ⚠️  UNBALANCE DETECTED at window {window_idx}")
                print(f"      Source: {selected_label} ({selected_name})")
                print(f"      Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"      Row index: {start_idx:,} - {end_idx:,}")
                print(f"      Detection ratio: {detection_ratio*100:.1f}%")
                print(f"      Mean prediction: {np.mean(predictions):.4f}")

                # Create detection figure
                fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                fig.suptitle(f'Unbalance Detection - {selected_label} ({selected_name})\\n'
                             f'Window {window_idx} | Rows {start_idx:,}-{end_idx:,} | '
                             f'{current_time.strftime("%Y-%m-%d %H:%M:%S")} UTC',
                             fontsize=14)

                # Panel 1: All sensor signals
                time_axis = np.arange(len(window_data)) / SAMPLES_PER_SECOND
                axes[0].plot(time_axis, window_data[:, 0], label='Vibration_1', alpha=0.7, linewidth=0.5)
                axes[0].plot(time_axis, window_data[:, 1], label='Vibration_2', alpha=0.7, linewidth=0.5)
                axes[0].plot(time_axis, window_data[:, 2], label='Vibration_3', alpha=0.7, linewidth=0.5)
                axes[0].set_ylabel('Amplitude')
                axes[0].set_title('Raw Vibration Signals')
                axes[0].legend(loc='upper right')
                axes[0].grid(True, alpha=0.3)

                # Panel 2: Vibration_1 zoomed
                axes[1].plot(time_axis, window_data[:, 0], color='green', linewidth=1)
                axes[1].set_ylabel('Amplitude')
                axes[1].set_title('Vibration Sensor 1')
                axes[1].grid(True, alpha=0.3)

                # Panel 3: Feature visualization (7 features per 1-second window)
                feature_names = ['V1\\nmean', 'V1\\nstd', 'V2\\nstd', 'V3\\nstd', 'V1\\nkurt', 'V2\\nkurt', 'V3\\nkurt']
                feature_time_axis = np.arange(len(window_features))
                for i in range(7):
                    axes[2].plot(feature_time_axis, window_features[:, i], marker='o', label=feature_names[i], markersize=3)
                axes[2].set_ylabel('Feature Value')
                axes[2].set_title('Extracted Minimal Features (7 per 1-second window)')
                axes[2].legend(loc='upper right', ncol=7, fontsize=8)
                axes[2].grid(True, alpha=0.3)

                # Panel 4: RFC predictions over time
                axes[3].plot(feature_time_axis, predictions, marker='o', color='red', markersize=4)
                axes[3].axhline(y=UNBALANCE_THRESHOLD, color='orange', linestyle='--',
                                label=f'Threshold ({UNBALANCE_THRESHOLD})')
                axes[3].fill_between(feature_time_axis, 0, 1,
                                      where=(predictions > UNBALANCE_THRESHOLD),
                                      alpha=0.3, color='red', label='Detected')
                axes[3].set_xlabel('1-Second Window Index')
                axes[3].set_ylabel('Probability')
                axes[3].set_title(f'Random Forest Predictions (Detection Ratio: {detection_ratio*100:.1f}%)')
                axes[3].set_ylim([-0.1, 1.1])
                axes[3].legend(loc='upper right')
                axes[3].grid(True, alpha=0.3)

                plt.tight_layout()

                # Save figure with timestamp
                timestamp_str = current_time.strftime('%Y%m%d_%H%M%S')
                filename = f"unbalance_detection_{selected_label}_{timestamp_str}_row{start_idx}.png"
                output_path = os.path.join(output_dir, filename)
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"      Figure saved: {output_path}")

                # Log detection event to JSONL file for MCP server
                detections_log_file = os.path.join(output_dir, "detections.jsonl")
                detection_event = {
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'window_idx': window_idx,
                    'dataset': selected_label,
                    'dataset_name': selected_name,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'detection_ratio': detection_ratio if isinstance(detection_ratio, (int, float)) else detection_ratio.item(),
                    'mean_prediction': np.mean(predictions).item(),
                    'max_prediction': np.max(predictions).item(),
                    'figure_file': filename
                }
                with open(detections_log_file, 'a') as f:
                    f.write(json.dumps(detection_event) + '\n')

            # Track performance metrics
            # Ground truth: 0E = negative (no unbalance), 1E-4E = positive (unbalance)
            ground_truth_positive = selected_label != '0E'  # True if 1E-4E
            predicted_positive = detection_ratio > 0.5       # True if detected

            if ground_truth_positive and predicted_positive:
                # True Positive: Correctly detected unbalance in 1E-4E
                dataset_stats[selected_label]['TP'] += 1
            elif not ground_truth_positive and not predicted_positive:
                # True Negative: Correctly identified no unbalance in 0E
                dataset_stats[selected_label]['TN'] += 1
            elif not ground_truth_positive and predicted_positive:
                # False Positive: Incorrectly detected unbalance in 0E
                dataset_stats[selected_label]['FP'] += 1
            else:
                # False Negative: Missed unbalance in 1E-4E
                dataset_stats[selected_label]['FN'] += 1

            dataset_stats[selected_label]['total'] += 1

            # Log performance metrics periodically
            if (window_idx + 1) % log_interval == 0:
                elapsed_time = time.time() - start_time

                # Print to console
                print(f"\n{'='*80}")
                print(f"Running Performance Metrics (after {window_idx + 1} windows, {elapsed_time:.1f}s elapsed)")
                print(f"{'='*80}")
                print(f"{'Dataset':<10} {'Processed':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Accuracy':<8}")
                print(f"{'-'*80}")

                for label in sorted(dataset_stats.keys()):
                    stats = dataset_stats[label]
                    total = stats['total']
                    if total > 0:
                        accuracy = (stats['TP'] + stats['TN']) / total
                        print(f"{label:<10} {total:<10} {stats['TP']:<6} {stats['FP']:<6} {stats['TN']:<6} {stats['FN']:<6} {accuracy:<8.3f}")

                # Calculate overall metrics
                overall_tp = sum(s['TP'] for s in dataset_stats.values())
                overall_fp = sum(s['FP'] for s in dataset_stats.values())
                overall_tn = sum(s['TN'] for s in dataset_stats.values())
                overall_fn = sum(s['FN'] for s in dataset_stats.values())
                overall_total = sum(s['total'] for s in dataset_stats.values())
                overall_accuracy = (overall_tp + overall_tn) / overall_total if overall_total > 0 else 0.0

                print(f"{'-'*80}")
                print(f"{'Overall':<10} {overall_total:<10} {overall_tp:<6} {overall_fp:<6} {overall_tn:<6} {overall_fn:<6} {overall_accuracy:<8.3f}")
                print(f"{'='*80}\n")

                # Write to report file
                with open(report_filename, 'w') as f:
                    f.write(f"Performance Report - Approach 3: Minimal RFC\n")
                    f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
                    f.write(f"Windows Processed: {window_idx + 1}\n")
                    f.write(f"Elapsed Time: {elapsed_time:.1f} seconds\n")
                    f.write(f"\n{'='*80}\n")
                    f.write(f"{'Dataset':<10} {'Total':<8} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}\n")
                    f.write(f"{'-'*80}\n")

                    for label in sorted(dataset_stats.keys()):
                        stats = dataset_stats[label]
                        total = stats['total']
                        if total > 0:
                            accuracy = (stats['TP'] + stats['TN']) / total
                            precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0.0
                            recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0.0
                            f.write(f"{label:<10} {total:<8} {stats['TP']:<8} {stats['FP']:<8} {stats['TN']:<8} {stats['FN']:<8} "
                                    f"{accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}\n")

                    if overall_total > 0:
                        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
                        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
                        f.write(f"{'-'*80}\n")
                        f.write(f"{'Overall':<10} {overall_total:<8} {overall_tp:<8} {overall_fp:<8} {overall_tn:<8} {overall_fn:<8} "
                                f"{overall_accuracy:<10.3f} {overall_precision:<10.3f} {overall_recall:<10.3f}\n")

            # Simulate real-time processing speed
            if speed > 0:
                time.sleep(time_window_seconds / speed)

            window_idx += 1

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Processing interrupted by user (Ctrl+C)")
        print(f"Processed {window_idx} windows before interruption")

    # Final summary
    total_processing_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Weighted Random Sampling Complete")
    print(f"Total windows processed: {window_idx} ({time_window_seconds}s each)")
    print(f"Total time analyzed: {window_idx * time_window_seconds / 60:.1f} minutes")
    print(f"Total unbalance detections: {total_detections}")
    print(f"\nDataset Selection Statistics:")
    total_selections = sum(dataset_selection_counts.values())
    if total_selections > 0:
        for label in sorted(dataset_selection_counts.keys()):
            count = dataset_selection_counts[label]
            percentage = (count / total_selections) * 100
            print(f"  {label}: {dataset_selection_counts[label]} times ({percentage:.1f}%)")

    # Print rollover statistics (only if any rollovers occurred)
    if any(count > 0 for count in dataset_rollover_counts.values()):
        print(f"\nDataset Rollover Statistics:")
        for label in sorted(dataset_rollover_counts.keys()):
            if dataset_rollover_counts[label] > 0:
                print(f"  {label}: {dataset_rollover_counts[label]} rollover(s)")

    # Final performance metrics
    print(f"\n{'='*80}")
    print(f"Performance Metrics")
    print(f"{'='*80}")
    print(f"{'Dataset':<10} {'Total':<8} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    print(f"{'-'*80}")

    for label in sorted(dataset_stats.keys()):
        stats = dataset_stats[label]
        total = stats['total']
        if total > 0:
            accuracy = (stats['TP'] + stats['TN']) / total
            precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0.0
            recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0.0

            print(f"{label:<10} {total:<8} {stats['TP']:<8} {stats['FP']:<8} {stats['TN']:<8} {stats['FN']:<8} "
                  f"{accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}")

    # Overall metrics
    overall_tp = sum(s['TP'] for s in dataset_stats.values())
    overall_fp = sum(s['FP'] for s in dataset_stats.values())
    overall_tn = sum(s['TN'] for s in dataset_stats.values())
    overall_fn = sum(s['FN'] for s in dataset_stats.values())
    overall_total = sum(s['total'] for s in dataset_stats.values())

    if overall_total > 0:
        overall_accuracy = (overall_tp + overall_tn) / overall_total
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
        print(f"{'-'*80}")
        print(f"{'Overall':<10} {overall_total:<8} {overall_tp:<8} {overall_fp:<8} {overall_tn:<8} {overall_fn:<8} "
              f"{overall_accuracy:<10.3f} {overall_precision:<10.3f} {overall_recall:<10.3f}")

    print(f"\nTotal processing time: {total_processing_time:.2f} seconds")
    if window_idx > 0:
        speed = (window_idx * time_window_seconds) / total_processing_time
        print(f"Average speed: {speed:.2f}x real-time")
    print(f"Detection figures saved to: {output_dir}")
    print(f"Performance report saved to: {report_filename}")
    print(f"{'='*80}")


def process_from_redis(model, output_dir, redis_config,
                      stream_name='windows', consumer_group='detectors',
                      consumer_name='rfc', log_interval=10, data=None):
    """
    Process windows from Redis stream for synchronized detection with minimal feature extraction.

    Args:
        model: Pre-trained Random Forest Classifier
        output_dir: Directory to save detection figures
        redis_config: RedisConfig instance
        stream_name: Redis stream name
        consumer_group: Consumer group name
        consumer_name: Consumer name for this approach
        log_interval: Log metrics to console every N windows
        data: Optional dictionary of {label: DataFrame} for fallback
    """
    print(f"\n{'='*80}")
    print("Redis Consumer Mode - Synchronized Window Processing")
    print(f"{'='*80}")
    print(f"Stream: {stream_name}")
    print(f"Consumer Group: {consumer_group}")
    print(f"Consumer Name: {consumer_name}")
    print(f"Log Interval: {log_interval} windows")
    print()

    # Initialize Redis consumer
    consumer = WindowConsumer(redis_config, stream_name, consumer_group, consumer_name)

    # Wait for stream to be created by coordinator
    print("Waiting for Redis stream to be created by data coordinator...")
    if not consumer.wait_for_stream(timeout_s=30):
        print("❌ Timeout waiting for stream. Make sure data_coordinator.py is running.")
        return

    # Clear and recreate output directory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"  Cleared existing output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamped report filename
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    report_filename = os.path.join(output_dir, f"performance_report_{timestamp}.txt")

    # Track performance metrics per dataset
    dataset_stats = {'0E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'total': 0},
                     '1E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'total': 0},
                     '2E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'total': 0},
                     '3E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'total': 0},
                     '4E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'total': 0}}

    # Track total detections
    total_detections = 0

    # Log file for detections (JSONL format for MCP server)
    detections_log = os.path.join(output_dir, "detections.jsonl")

    print(f"Performance report: {report_filename}")
    print(f"Updating report every {log_interval} windows\n")

    # Performance tracking
    start_time = time.time()
    window_idx = 0

    print(f"✓ Connected and ready to process windows")
    print(f"Waiting for windows from data coordinator...")
    print(f"(Press Ctrl+C to stop)\n")

    try:
        while True:
            # Read next window from Redis (blocking)
            window_msg = consumer.read_window(block_ms=5000)

            if window_msg is None:
                # Timeout - no new messages, continue waiting
                continue

            # Extract window information
            dataset_label = window_msg['dataset']
            start_idx = window_msg['start_idx']
            end_idx = window_msg['end_idx']
            redis_timestamp = window_msg['timestamp']

            # Get window data from Redis message (if available) or from loaded data
            if 'sensor_data' in window_msg:
                # Sensor data from Redis: columns are [RPM, Vibration_1, Vibration_2, Vibration_3]
                # RFC uses all 4 columns
                window_data = window_msg['sensor_data']
            elif data is not None and dataset_label in data:
                # Fallback to loaded data
                # Select columns: Measured_RPM (index 1) and Vibration_1/2/3 (indices 2-4)
                # Skip V_in (index 0)
                window_data = data[dataset_label].iloc[start_idx:end_idx, 1:].values
            else:
                print(f"⚠️  No sensor data available for window, skipping")
                consumer.acknowledge(window_msg['message_id'])
                continue

            # Extract 1-second windows for feature extraction
            n_second_windows = len(window_data) // WINDOW
            if n_second_windows == 0:
                print(f"⚠️  Warning: Window too small ({len(window_data)} samples), skipping")
                consumer.acknowledge(window_msg['message_id'])
                continue

            window_data_windowed = window_data[:n_second_windows * WINDOW].reshape(n_second_windows, WINDOW, 3)

            # Extract minimal features for each 1-second window
            window_features = np.array([extract_minimal_features(w) for w in window_data_windowed])

            # Predict on all 1-second windows
            predictions = model.predict_proba(window_features)[:, 1]  # Probability of class 1 (unbalance)

            # Detection logic: majority voting
            detection_ratio = np.mean(predictions > UNBALANCE_THRESHOLD)

            # Get current UTC time
            current_time = datetime.now(timezone.utc)
            selected_name = DATASET_NAMES.get(dataset_label, dataset_label)

            # Determine if unbalance detected
            if detection_ratio > 0.5:
                total_detections += 1

                print(f"\n  ⚠️  UNBALANCE DETECTED at window {window_idx}")
                print(f"      Source: {dataset_label} ({selected_name})")
                print(f"      Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"      Row index: {start_idx:,} - {end_idx:,}")
                print(f"      Detection ratio: {detection_ratio*100:.1f}%")
                print(f"      Mean prediction: {np.mean(predictions):.4f}")

                # Create detection figure
                fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                time_window_seconds = len(window_data) / SAMPLES_PER_SECOND
                fig.suptitle(f'Unbalance Detection - {dataset_label} ({selected_name})\n'
                             f'Window {window_idx} | Rows {start_idx:,}-{end_idx:,} | '
                             f'{current_time.strftime("%Y-%m-%d %H:%M:%S")} UTC',
                             fontsize=14, fontweight='bold')

                # Plot 1: Vibration Sensor 1
                time_axis = np.arange(len(window_data)) / SAMPLES_PER_SECOND
                axes[0].plot(time_axis, window_data[:, 0], lw=0.5, color='blue')
                axes[0].set_ylabel('Amplitude')
                axes[0].set_title('Vibration Sensor 1')
                axes[0].grid(True, alpha=0.3)

                # Plot 2-3: Vibration sensors 2-3
                axes[1].plot(time_axis, window_data[:, 1], lw=0.5, color='green')
                axes[1].set_ylabel('Amplitude')
                axes[1].set_title('Vibration Sensor 2')
                axes[1].grid(True, alpha=0.3)

                axes[2].plot(time_axis, window_data[:, 2], lw=0.5, color='orange')
                axes[2].set_ylabel('Amplitude')
                axes[2].set_title('Vibration Sensor 3')
                axes[2].grid(True, alpha=0.3)

                # Plot 4: All sensors combined
                for j, color in enumerate(['blue', 'green', 'orange']):
                    axes[3].plot(time_axis, window_data[:, j], lw=0.3, color=color, alpha=0.7)
                axes[3].set_ylabel('Amplitude')
                axes[3].set_title('All Sensors Combined')
                axes[3].grid(True, alpha=0.3)

                axes[-1].set_xlabel('Time (seconds)')
                plt.tight_layout()

                # Save figure
                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                filename = f"unbalance_detection_{dataset_label}_{timestamp_str}_window{window_idx}_row{start_idx}.png"
                output_path = os.path.join(output_dir, filename)
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"      Figure saved: {output_path}")

                # Log detection event to JSONL file
                detection_event = {
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'window_idx': window_idx,
                    'dataset': dataset_label,
                    'dataset_name': selected_name,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'detection_ratio': detection_ratio if isinstance(detection_ratio, (int, float)) else detection_ratio.item(),
                    'mean_prediction': np.mean(predictions).item() if hasattr(np.mean(predictions), 'item') else float(np.mean(predictions)),
                    'max_prediction': np.max(predictions).item() if hasattr(np.max(predictions), 'item') else float(np.max(predictions)),
                    'figure_file': filename
                }
                with open(detections_log, 'a') as f:
                    f.write(json.dumps(detection_event) + '\n')

            # Track performance metrics
            ground_truth_positive = dataset_label != '0E'
            predicted_positive = detection_ratio > 0.5

            if ground_truth_positive and predicted_positive:
                dataset_stats[dataset_label]['TP'] += 1
            elif ground_truth_positive and not predicted_positive:
                dataset_stats[dataset_label]['FN'] += 1
            elif not ground_truth_positive and predicted_positive:
                dataset_stats[dataset_label]['FP'] += 1
            else:
                dataset_stats[dataset_label]['TN'] += 1

            dataset_stats[dataset_label]['total'] += 1

            # Acknowledge message
            consumer.acknowledge(window_msg['message_id'])

            # Log running metrics every N windows
            if (window_idx + 1) % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"\n{'='*80}")
                print(f"Running Performance Metrics (after {window_idx + 1} windows, {elapsed_time:.1f}s elapsed)")
                print(f"{'='*80}")
                print(f"{'Dataset':<10} {'Processed':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Accuracy':<8}")
                print(f"{'-'*80}")

                overall_tp, overall_fp, overall_tn, overall_fn = 0, 0, 0, 0

                for label in sorted(dataset_stats.keys()):
                    stats = dataset_stats[label]
                    total = stats['total']
                    if total > 0:
                        accuracy = (stats['TP'] + stats['TN']) / total
                        print(f"{label:<10} {total:<10} {stats['TP']:<6} {stats['FP']:<6} {stats['TN']:<6} {stats['FN']:<6} {accuracy:<8.3f}")
                        overall_tp += stats['TP']
                        overall_fp += stats['FP']
                        overall_tn += stats['TN']
                        overall_fn += stats['FN']

                overall_total = overall_tp + overall_fp + overall_tn + overall_fn
                if overall_total > 0:
                    overall_accuracy = (overall_tp + overall_tn) / overall_total
                    print(f"{'-'*80}")
                    print(f"{'Overall':<10} {overall_total:<10} {overall_tp:<6} {overall_fp:<6} {overall_tn:<6} {overall_fn:<6} {overall_accuracy:<8.3f}")
                print(f"{'='*80}\n")

                # Write performance report to file
                with open(report_filename, 'w') as f:
                    f.write(f"{'='*80}\n")
                    f.write(f"Performance Metrics - Minimal RFC Approach (Redis Consumer Mode)\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"{'Dataset':<10} {'Total':<8} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}\n")
                    f.write(f"{'-'*80}\n")

                    for label in sorted(dataset_stats.keys()):
                        stats = dataset_stats[label]
                        total = stats['total']
                        if total > 0:
                            accuracy = (stats['TP'] + stats['TN']) / total
                            precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0.0
                            recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0.0

                            f.write(f"{label:<10} {total:<8} {stats['TP']:<8} {stats['FP']:<8} {stats['TN']:<8} {stats['FN']:<8} "
                                    f"{accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}\n")

                    # Write overall metrics
                    overall_total = overall_tp + overall_fp + overall_tn + overall_fn
                    if overall_total > 0:
                        overall_accuracy = (overall_tp + overall_tn) / overall_total
                        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
                        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
                        f.write(f"{'-'*80}\n")
                        f.write(f"{'Overall':<10} {overall_total:<8} {overall_tp:<8} {overall_fp:<8} {overall_tn:<8} {overall_fn:<8} "
                                f"{overall_accuracy:<10.3f} {overall_precision:<10.3f} {overall_recall:<10.3f}\n")

            window_idx += 1

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Processing interrupted by user (Ctrl+C)")
        print(f"Processed {window_idx} windows before interruption")

    # Final summary
    total_processing_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Redis Consumer Processing Complete")
    print(f"{'='*80}")
    print(f"Total windows processed: {window_idx}")
    print(f"Total unbalance detections: {total_detections}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")

    # Final performance metrics
    print(f"\n{'='*80}")
    print(f"Performance Metrics")
    print(f"{'='*80}")
    print(f"{'Dataset':<10} {'Total':<8} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    print(f"{'-'*80}")

    for label in sorted(dataset_stats.keys()):
        stats = dataset_stats[label]
        total = stats['total']
        if total > 0:
            accuracy = (stats['TP'] + stats['TN']) / total
            precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0.0
            recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0.0

            print(f"{label:<10} {total:<8} {stats['TP']:<8} {stats['FP']:<8} {stats['TN']:<8} {stats['FN']:<8} "
                  f"{accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}")

    # Overall metrics
    overall_tp = sum(s['TP'] for s in dataset_stats.values())
    overall_fp = sum(s['FP'] for s in dataset_stats.values())
    overall_tn = sum(s['TN'] for s in dataset_stats.values())
    overall_fn = sum(s['FN'] for s in dataset_stats.values())
    overall_total = sum(s['total'] for s in dataset_stats.values())

    if overall_total > 0:
        overall_accuracy = (overall_tp + overall_tn) / overall_total
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
        print(f"{'-'*80}")
        print(f"{'Overall':<10} {overall_total:<8} {overall_tp:<8} {overall_fp:<8} {overall_tn:<8} {overall_fn:<8} "
              f"{overall_accuracy:<10.3f} {overall_precision:<10.3f} {overall_recall:<10.3f}")

    print(f"\nDetection figures saved to: {output_dir}")
    print(f"Performance report saved to: {report_filename}")
    print(f"{'='*80}")

    consumer.close()


def main():
    parser = argparse.ArgumentParser(
        description='Approach 3: Random Forest Classifier on Minimal Statistical Features for Unbalance Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets with default settings
  python approach_3_minimal_rfc.py --dataset all

  # Process only dataset 0E (no unbalance)
  python approach_3_minimal_rfc.py --dataset 0E

  # Process 100 windows with 60-second windows
  python approach_3_minimal_rfc.py --dataset all --max-windows 100 --time-window 60

  # Continuous mode with 90% normal weight
  python approach_3_minimal_rfc.py --dataset all --normal-weight 0.9

  # Enable MCP server for real-time monitoring
  python approach_3_minimal_rfc.py --dataset all --enable-mcp
        """)

    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset to process: 0E, 1E, 2E, 3E, 4E, or "all" (default: all)')
    parser.add_argument('--time-window', type=int, default=10,
                        help='Time window duration in seconds (default: 10)')
    parser.add_argument('--max-windows', type=int, default=None,
                        help='Maximum number of windows to process (default: infinite, use Ctrl+C to stop)')
    parser.add_argument('--speed', type=float, default=0,
                        help='Playback speed multiplier (0=maximum, 1=real-time, 2=2x, etc.) (default: 0)')
    parser.add_argument('--normal-weight', type=float, default=0.9,
                        help='Probability of selecting normal (0E) data (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='How often to print performance metrics in number of windows (default: 10)')
    parser.add_argument('--model-path', type=str,
                        default='../models/minimal_rfc.joblib',
                        help='Path to pre-trained RFC model (default: ../models/minimal_rfc.joblib)')
    parser.add_argument('--data-url', type=str,
                        default='../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip',
                        help='Path or URL to dataset ZIP file')
    parser.add_argument('--output-dir', type=str,
                        default='../figures/detections',
                        help='Directory to save detection figures (default: ../figures/detections)')
    parser.add_argument('--enable-mcp', action='store_true',
                        help='Enable MCP server for real-time monitoring')
    parser.add_argument('--mcp-port', type=int, default=8000,
                        help='Port for MCP server (default: 8000)')

    # Redis synchronization arguments
    parser.add_argument('--redis-mode', action='store_true',
                       help='Enable Redis consumer mode for synchronized processing')
    parser.add_argument('--redis-host', type=str, default=None,
                       help='Redis host (default: localhost or REDIS_HOST env var)')
    parser.add_argument('--redis-port', type=int, default=None,
                       help='Redis port (default: 6379 or REDIS_PORT env var)')
    parser.add_argument('--redis-stream', type=str, default='windows',
                       help='Redis stream name (default: windows)')
    parser.add_argument('--consumer-group', type=str, default='detectors',
                       help='Redis consumer group (default: detectors)')
    parser.add_argument('--consumer-name', type=str, default='rfc',
                       help='Consumer name for this approach (default: rfc)')

    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print("Machine Learning Based Unbalance Detection - Approach 3: Minimal RFC")
    print("Real-time Anomaly Detection with Random Forest (7 Statistical Features)")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    if args.dataset == 'all':
        print(f"  Normal Weight (0E): {args.normal_weight*100:.0f}%")
    print(f"  Time Window: {args.time_window} seconds")
    print(f"  Max Windows: {'Infinite (Ctrl+C to stop)' if args.max_windows is None else args.max_windows}")
    print(f"  Speed: {'Maximum' if args.speed == 0 else f'{args.speed}x real-time'}")
    print(f"  Model Path: {args.model_path}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  MCP Server: {'Enabled' if args.enable_mcp else 'Disabled'}")
    print()

    # Test Redis connection if in Redis mode
    if args.redis_mode:
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

    # Start MCP server if enabled
    if args.enable_mcp:
        try:
            import threading
            from mcp_server import mcp, set_detections_dir, run_mcp_server

            # Set the detections directory for MCP server
            set_detections_dir(args.output_dir)

            # Run MCP server in background thread
            mcp_thread = threading.Thread(
                target=run_mcp_server,
                args=("streamable-http", args.mcp_port, "0.0.0.0"),
                daemon=True
            )
            mcp_thread.start()

            print("✓ MCP server started in background thread")
            print(f"  HTTP endpoint: http://0.0.0.0:{args.mcp_port}")
            print("  Available for real-time monitoring queries")
            print()
        except ImportError as e:
            print(f"⚠️  Warning: Could not start MCP server: {e}")
            print("  Install fastmcp to enable: pip install fastmcp")
            print()
        except Exception as e:
            print(f"⚠️  Warning: MCP server failed to start: {e}")
            print()

    # Check model file exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        print("\nPlease ensure the reference model exists, or specify a different model with --model-path")
        sys.exit(1)

    # Determine which datasets to load
    if args.dataset == 'all':
        datasets_to_load = ['0E', '1E', '2E', '3E', '4E']
    else:
        datasets_to_load = [args.dataset]

    print("=" * 80)
    print("STEP 1: Load Pre-trained Model")
    print("=" * 80)
    print(f"Loading reference model from: {args.model_path}")

    model = load(args.model_path)
    print(f"✓ Model loaded successfully")
    print(f"  Type: {type(model).__name__}")
    if hasattr(model, 'best_params_'):
        print(f"  Best parameters: {model.best_params_}")
    if hasattr(model, 'n_estimators'):
        print(f"  Number of trees: {model.n_estimators}")
    print()

    if args.dataset == 'all':
        print("\n" + "=" * 80)
        print("STEP 2: Model Validation on Combined Evaluation Dataset")
        print("=" * 80)
        print("Skipped (using pre-trained model with known performance)")
        print("See training script output for validation metrics")

    # STEP 3: Load datasets (skip in Redis mode - data comes from coordinator)
    if args.redis_mode:
        print("\n" + "=" * 80)
        print("Redis Mode - Skipping Data Loading")
        print("=" * 80)
        print("Data will be received from coordinator via Redis stream")
        data_prepared = None
    else:
        print("\n" + "=" * 80)
        print(f"STEP {'3' if args.dataset == 'all' else '2'}: Load Datasets")
        print("=" * 80)

        # Load data from ZIP file
        data_zip_path = args.data_url
        data = load_data(data_zip_path, datasets_to_load=datasets_to_load)
        data_prepared = prepare_datasets_minimal(data)

    # STEP 4: Process data
    print("\n" + "=" * 80)
    print(f"STEP {'4' if args.dataset == 'all' else '3'}: Real-time Unbalance Detection")
    print("=" * 80)

    if args.redis_mode:
        # Redis consumer mode - synchronized processing
        print("Mode: Redis Consumer (Synchronized)")
        print(f"  Redis Stream: {args.redis_stream}")
        print(f"  Consumer Group: {args.consumer_group}")
        print(f"  Consumer Name: {args.consumer_name}")
        print()

        # Signal that we're ready (for health checks)
        with open('/tmp/ready', 'w') as f:
            f.write('ready\n')
        print("✓ Application ready - health check enabled")
        print()

        redis_config = RedisConfig(host=args.redis_host, port=args.redis_port)
        process_from_redis(
            model=model,
            output_dir=args.output_dir,
            redis_config=redis_config,
            stream_name=args.redis_stream,
            consumer_group=args.consumer_group,
            consumer_name=args.consumer_name,
            log_interval=args.log_interval,
            data=data_prepared  # Optional fallback if sensor data not in Redis
        )
    else:
        # Standalone mode - weighted random sampling
        print("Mode: Standalone (Weighted Random Sampling)")
        print()

        process_with_weighted_sampling(
            model=model,
            data=data_prepared,
            output_dir=args.output_dir,
            speed=args.speed,
            time_window_seconds=args.time_window,
            max_windows=args.max_windows,
            normal_weight=args.normal_weight,
            log_interval=args.log_interval
        )


if __name__ == "__main__":
    main()
