#!/usr/bin/env python3
"""
Machine Learning Based Unbalance Detection of a Rotating Shaft Using Vibration Data
Notebook 3 of 5 - Approach 2: Fully-Connected Neural Network on FFT-transformed Data

Oliver Mey, Willi Neudeck, André Schneider and Olaf Enge-Rosenblatt
Fraunhofer IIS/EAS, Fraunhofer Institute for Integrated Circuits,
Division Engineering of Adaptive Systems, Dresden, Germany

This Python script is converted from the Jupyter Notebook that is part of a paper
submission to the 25th IEEE International Conference on Emerging Technologies and
Factory Automation, ETFA 2020.

Modified to:
1. Use pre-trained 4-layer FFT FCN model
2. Process evaluation data as timeseries with configurable time windows
3. Detect unbalance anomalies in real-time
4. Generate and save figures for detected unbalances
5. Apply FFT transformation and RobustScaler preprocessing
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import os
import argparse
import time
import json
import threading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import pickle

from tensorflow.keras.models import load_model
from datetime import datetime

# Add apps directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared data utilities
from utils.data_utils import (
    load_data,
    skip_warmup,
    DataGenerator,
    SAMPLES_PER_SECOND,
    SKIP_WARMUP as SKIP,
    DEFAULT_SENSOR as SENSOR
)
from utils.redis_client import RedisConfig, WindowConsumer, test_redis_connection

# Constants
WINDOW = 4096  # Samples per second (1-second window)
FFT_FEATURES = 2048  # Half of window size (rfft output)
UNBALANCE_THRESHOLD = 0.5  # Prediction threshold for detection


def prepare_datasets_fft(data):
    """
    Prepare training and validation datasets with FFT transformation.

    Args:
        data: Dictionary of pandas DataFrames (0E-4E)

    Returns:
        X_val_fft: FFT-transformed validation data
        y_val: Validation labels
        scaler: Fitted RobustScaler for preprocessing
    """
    print("\nPreparing datasets with FFT transformation...")

    # Labels
    labels = {'no_unbalance': 0, 'unbalance': 1}

    def get_features(df, label):
        """Extract features from dataframe."""
        n = int(np.floor(len(df) / WINDOW))
        data_array = df[:int(n) * WINDOW]
        X = data_array.values.reshape((n, WINDOW))
        y = np.ones(n) * labels[label]
        return X, y

    # Prepare validation data (0E-4E)
    X_val_list = []
    y_val_list = []

    for label in ['0E', '1E', '2E', '3E', '4E']:
        if label in data:
            lbl = 'no_unbalance' if label == '0E' else 'unbalance'
            X, y = get_features(data[label][SENSOR], lbl)
            X_val_list.append(X)
            y_val_list.append(y)
            print(f"  {label}: {len(X)} samples")

    X_val = np.concatenate(X_val_list)
    y_val = np.concatenate(y_val_list)

    print(f"\nValidation set: {X_val.shape}")

    # Apply FFT transformation
    print("Applying FFT transformation...")
    X_val_fft = np.abs(np.fft.rfft(X_val, axis=1))[:, :FFT_FEATURES]
    X_val_fft[:, 0] = 0  # Zero out DC component

    print(f"FFT features: {X_val_fft.shape}")

    # For scaling, we need some training data - use a subset of 0E and 1E-4E
    print("\nPreparing scaler using subset of data...")
    X_train_subset = X_val[:5000]  # Use first 5000 samples for scaler fitting
    X_train_fft = np.abs(np.fft.rfft(X_train_subset, axis=1))[:, :FFT_FEATURES]
    X_train_fft[:, 0] = 0

    # Fit scaler
    scaler = RobustScaler(quantile_range=(5, 95)).fit(X_train_fft)
    print("Scaler fitted")

    # Transform validation data
    X_val_fft_scaled = scaler.transform(X_val_fft)

    return X_val_fft_scaled, y_val, scaler


def process_with_weighted_sampling(model, data, scaler, output_dir, speed=0,
                                   time_window_seconds=60, max_windows=None,
                                   normal_weight=0.9, log_interval=10):
    """
    Process data using weighted random sampling with FFT preprocessing.

    Args:
        model: Trained FFT FCN model
        data: Dictionary of evaluation datasets (0E-4E)
        scaler: Fitted RobustScaler
        output_dir: Directory to save detection figures
        speed: Processing speed multiplier (0 = max, 1.0 = real-time)
        time_window_seconds: Time window size in seconds
        max_windows: Maximum number of windows to process (None = infinite with rollover)
        normal_weight: Weight for 0E dataset (0.0-1.0), default 0.9 (90%)
        log_interval: Log metrics to console every N windows (default: 10)
    """
    total_detections = 0
    processing_start = time.time()

    # Calculate time window in samples
    time_window_samples = int(time_window_seconds * SAMPLES_PER_SECOND)

    # Calculate how many windows are available in each dataset (for rollover)
    dataset_windows = {}
    for label in ['0E', '1E', '2E', '3E', '4E']:
        if label in data:
            total_samples = len(data[label][SENSOR].values)
            n_windows = int(np.floor(total_samples / time_window_samples))
            dataset_windows[label] = n_windows
            print(f"{label}: {n_windows} windows available ({n_windows * time_window_seconds / 60:.1f} minutes)")

    if not dataset_windows:
        print("No data available to process")
        return

    # Determine how many windows to process
    print(f"\nWeighted Random Sampling Configuration:")
    print(f"  Normal (0E) weight: {normal_weight*100:.0f}%")
    print(f"  Unbalanced (1E-4E) weight: {(1-normal_weight)*100:.0f}%")

    if max_windows is None:
        # Infinite mode - process until user stops (Ctrl+C)
        print(f"\nContinuous processing mode (infinite with rollover)")
        print(f"Press Ctrl+C to stop")
        num_windows_to_process = float('inf')
    else:
        num_windows_to_process = max_windows
        print(f"\nProcessing {max_windows} windows ({time_window_seconds}s each, {max_windows * time_window_seconds / 60:.1f} minutes total) with weighted random sampling...")
    print()

    # Track stats
    dataset_selection_counts = {label: 0 for label in ['0E', '1E', '2E', '3E', '4E']}
    dataset_rollover_counts = {label: 0 for label in ['0E', '1E', '2E', '3E', '4E']}

    # Performance tracking: TP, FP, TN, FN per dataset
    performance_stats = {
        '0E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},  # Ground truth: no unbalance
        '1E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},  # Ground truth: unbalance
        '2E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},  # Ground truth: unbalance
        '3E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},  # Ground truth: unbalance
        '4E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},  # Ground truth: unbalance
    }

    # Create performance report file with start timestamp
    start_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    report_filename = os.path.join(output_dir, f"performance_report_{start_timestamp}.txt")
    print(f"Performance report: {report_filename}")
    print(f"Updating report every {log_interval} windows")
    print(f"  (Time interval: {log_interval * time_window_seconds} seconds)\n")

    # Keep track of current position in each dataset (with rollover)
    dataset_positions = {label: 0 for label in dataset_windows.keys()}

    window_idx = 0
    try:
        while window_idx < num_windows_to_process:
            window_process_start = time.time()

            # Weighted random selection
            rand_val = np.random.random()

            if rand_val < normal_weight and '0E' in data:
                # Select 0E (no unbalance)
                selected_label = '0E'
                selected_name = 'No Unbalance'
            else:
                # Select randomly from 1E-4E (unbalanced datasets)
                unbalanced_labels = [label for label in ['1E', '2E', '3E', '4E'] if label in data]
                if not unbalanced_labels:
                    selected_label = '0E'
                    selected_name = 'No Unbalance'
                else:
                    selected_label = np.random.choice(unbalanced_labels)
                    level = ['1E', '2E', '3E', '4E'].index(selected_label) + 1
                    selected_name = f'Unbalance Level {level}'

            # Track selection
            dataset_selection_counts[selected_label] += 1

            # Get the current window from the selected dataset (with rollover)
            current_pos = dataset_positions[selected_label]

            # Rollover if we've reached the end
            if current_pos >= dataset_windows[selected_label]:
                current_pos = 0
                dataset_positions[selected_label] = 0
                dataset_rollover_counts[selected_label] += 1
                if dataset_rollover_counts[selected_label] == 1:
                    print(f"\n  🔄 Dataset {selected_label} rolled over to beginning")

            sensor_data = data[selected_label][SENSOR].values

            start_idx = current_pos * time_window_samples
            end_idx = start_idx + time_window_samples
            window_data = sensor_data[start_idx:end_idx]

            # Move to next window in this dataset
            dataset_positions[selected_label] += 1

            # Split into 1-second windows for prediction
            n_second_windows = int(np.floor(len(window_data) / WINDOW))
            window_data_windowed = window_data[:n_second_windows * WINDOW].reshape((n_second_windows, WINDOW))

            # Apply FFT transformation
            window_data_fft = np.abs(np.fft.rfft(window_data_windowed, axis=1))[:, :FFT_FEATURES]
            window_data_fft[:, 0] = 0  # Zero out DC component

            # Apply scaling
            window_data_fft_scaled = scaler.transform(window_data_fft)

            # Predict on all 1-second windows in this time window
            predictions = model.predict(window_data_fft_scaled, verbose=0)
            predictions_binary = (predictions > UNBALANCE_THRESHOLD).astype(int).flatten()

            # Check if unbalance detected (majority voting)
            unbalance_detections = np.sum(predictions_binary)
            detection_ratio = unbalance_detections / len(predictions_binary)

            # Calculate timestamp (use system UTC time)
            current_time = datetime.utcnow()
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")

            # If significant unbalance detected (>50% of windows), save figure
            if detection_ratio > 0.5:
                total_detections += 1

                print(f"\n  ⚠️  UNBALANCE DETECTED at window {window_idx}")
                print(f"      Source: {selected_label} ({selected_name})")
                print(f"      Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"      Row index: {start_idx:,} - {end_idx:,}")
                print(f"      Detection ratio: {detection_ratio*100:.1f}%")
                print(f"      Mean prediction: {np.mean(predictions):.4f}")

                # Generate and save figure
                fig = plt.figure(figsize=(15, 10))

                # Plot 1: Full time window timeseries
                ax1 = plt.subplot(3, 1, 1)
                time_axis = np.arange(len(window_data)) / SAMPLES_PER_SECOND
                ax1.plot(time_axis, window_data, lw=0.5)
                ax1.set_title(f"UNBALANCE DETECTED - Dataset {selected_label} ({selected_name})\n"
                             f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                             f"Rows: {start_idx:,}-{end_idx:,} | Window: {time_window_seconds}s",
                             fontsize=14, fontweight='bold')
                ax1.set_xlabel("Time (seconds)")
                ax1.set_ylabel("Vibration Amplitude")
                ax1.grid(True, alpha=0.3)

                # Plot 2: Predictions over time
                ax2 = plt.subplot(3, 1, 2)
                pred_times = np.arange(len(predictions))
                ax2.plot(pred_times, predictions, marker='o', markersize=3, linewidth=1)
                ax2.axhline(y=UNBALANCE_THRESHOLD, color='r', linestyle='--',
                           label=f'Threshold ({UNBALANCE_THRESHOLD})')
                ax2.fill_between(pred_times, 0, 1, where=(predictions.flatten() > UNBALANCE_THRESHOLD),
                                alpha=0.3, color='red', label='Unbalance Detected')
                ax2.set_title(f"Predictions per Second (Detection Ratio: {detection_ratio*100:.1f}%)")
                ax2.set_xlabel("Second")
                ax2.set_ylabel("Prediction Score")
                ax2.set_ylim([-0.05, 1.05])
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Plot 3: FFT spectrum of highest prediction window
                ax3 = plt.subplot(3, 1, 3)
                max_pred_idx = np.argmax(predictions)
                sample_data = window_data_windowed[max_pred_idx]
                # Show FFT spectrum
                fft_spectrum = np.abs(np.fft.rfft(sample_data))[:FFT_FEATURES]
                freqs = np.fft.rfftfreq(WINDOW, 1/SAMPLES_PER_SECOND)[:FFT_FEATURES]
                ax3.plot(freqs, fft_spectrum, lw=0.8)
                ax3.set_title(f"FFT Spectrum - Highest Prediction Window (Prediction: {predictions[max_pred_idx][0]:.4f})")
                ax3.set_xlabel("Frequency (Hz)")
                ax3.set_ylabel("Magnitude")
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save figure
                filename = f"unbalance_detection_{selected_label}_{timestamp_str}_window{window_idx}_row{start_idx}.png"
                output_path = os.path.join(output_dir, filename)
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"      Figure saved: {output_path}")

                # Log detection event to JSONL file for MCP server
                detections_log = os.path.join(output_dir, "detections.jsonl")
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
                with open(detections_log, 'a') as f:
                    f.write(json.dumps(detection_event) + '\n')

            # Track performance metrics
            # Ground truth: 0E = negative (no unbalance), 1E-4E = positive (unbalance)
            ground_truth_positive = selected_label != '0E'  # True if 1E-4E
            predicted_positive = detection_ratio > 0.5       # True if detected

            if ground_truth_positive and predicted_positive:
                # True Positive: Correctly detected unbalance in 1E-4E
                performance_stats[selected_label]['TP'] += 1
                result = 'TP'
            elif ground_truth_positive and not predicted_positive:
                # False Negative: Missed unbalance in 1E-4E
                performance_stats[selected_label]['FN'] += 1
                result = 'FN'
            elif not ground_truth_positive and predicted_positive:
                # False Positive: Incorrectly detected unbalance in 0E
                performance_stats[selected_label]['FP'] += 1
                result = 'FP'
            else:
                # True Negative: Correctly identified no unbalance in 0E
                performance_stats[selected_label]['TN'] += 1
                result = 'TN'

            # Log running metrics and update report file every N windows
            if (window_idx + 1) % log_interval == 0:
                # Log to console
                elapsed_time = time.time() - processing_start
                print(f"\n{'='*80}")
                print(f"Running Performance Metrics (after {window_idx + 1} windows, {elapsed_time:.1f}s elapsed)")
                print(f"{'='*80}")
                print(f"{'Dataset':<10} {'Processed':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Accuracy':<8}")
                print(f"{'-'*80}")

                for label in ['0E', '1E', '2E', '3E', '4E']:
                    if label in dataset_selection_counts and dataset_selection_counts[label] > 0:
                        stats = performance_stats[label]
                        total = stats['TP'] + stats['FP'] + stats['TN'] + stats['FN']
                        accuracy = (stats['TP'] + stats['TN']) / total if total > 0 else 0
                        print(f"{label:<10} {total:<10} {stats['TP']:<6} {stats['FP']:<6} {stats['TN']:<6} {stats['FN']:<6} {accuracy:<8.3f}")

                # Overall running stats
                overall_tp = sum(stats['TP'] for stats in performance_stats.values())
                overall_fp = sum(stats['FP'] for stats in performance_stats.values())
                overall_tn = sum(stats['TN'] for stats in performance_stats.values())
                overall_fn = sum(stats['FN'] for stats in performance_stats.values())
                overall_total = overall_tp + overall_fp + overall_tn + overall_fn
                overall_accuracy = (overall_tp + overall_tn) / overall_total if overall_total > 0 else 0

                print(f"{'-'*80}")
                print(f"{'Overall':<10} {overall_total:<10} {overall_tp:<6} {overall_fp:<6} {overall_tn:<6} {overall_fn:<6} {overall_accuracy:<8.3f}")
                print(f"{'='*80}\n")

                # Write performance report to file
                with open(report_filename, 'w') as report_file:
                    report_file.write(f"{'='*80}\n")
                    report_file.write(f"Performance Metrics\n")
                    report_file.write(f"{'='*80}\n")
                    report_file.write(f"{'Dataset':<10} {'Total':<8} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}\n")
                    report_file.write(f"{'-'*80}\n")

                    for label in ['0E', '1E', '2E', '3E', '4E']:
                        if label in dataset_selection_counts and dataset_selection_counts[label] > 0:
                            stats = performance_stats[label]
                            total = stats['TP'] + stats['FP'] + stats['TN'] + stats['FN']
                            accuracy = (stats['TP'] + stats['TN']) / total if total > 0 else 0
                            precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
                            recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0

                            report_file.write(f"{label:<10} {total:<8} {stats['TP']:<8} {stats['FP']:<8} {stats['TN']:<8} {stats['FN']:<8} "
                                            f"{accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}\n")

                    report_file.write(f"{'-'*80}\n")
                    report_file.write(f"{'Overall':<10} {overall_total:<8} {overall_tp:<8} {overall_fp:<8} {overall_tn:<8} {overall_fn:<8} "
                                    f"{overall_accuracy:<10.3f} ")

                    # Calculate overall precision and recall
                    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
                    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
                    report_file.write(f"{overall_precision:<10.3f} {overall_recall:<10.3f}\n")

            # Simulate real-time processing speed if requested
            if speed > 0:
                window_process_time = time.time() - window_process_start
                target_time = time_window_seconds / speed
                sleep_time = target_time - window_process_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            window_idx += 1

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Processing interrupted by user (Ctrl+C)")
        print(f"Processed {window_idx} windows before interruption")

    # Print processing summary
    total_processing_time = time.time() - processing_start
    print(f"\n{'='*80}")
    print(f"Weighted Random Sampling Complete")
    print(f"Total windows processed: {window_idx} ({time_window_seconds}s each)")
    print(f"Total time analyzed: {window_idx * time_window_seconds / 60:.1f} minutes")
    print(f"Total unbalance detections: {total_detections}")
    print(f"\nDataset Selection Statistics:")
    for label in ['0E', '1E', '2E', '3E', '4E']:
        if label in dataset_selection_counts and dataset_selection_counts[label] > 0:
            percentage = (dataset_selection_counts[label] / window_idx) * 100 if window_idx > 0 else 0
            print(f"  {label}: {dataset_selection_counts[label]} times ({percentage:.1f}%)")

    # Print rollover statistics if any occurred
    total_rollovers = sum(dataset_rollover_counts.values())
    if total_rollovers > 0:
        print(f"\nDataset Rollover Statistics:")
        for label in ['0E', '1E', '2E', '3E', '4E']:
            if dataset_rollover_counts[label] > 0:
                print(f"  {label}: {dataset_rollover_counts[label]} rollover(s)")

    # Print performance metrics
    print(f"\n{'='*80}")
    print(f"Performance Metrics")
    print(f"{'='*80}")
    print(f"{'Dataset':<10} {'Total':<8} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    print(f"{'-'*80}")

    overall_tp, overall_fp, overall_tn, overall_fn = 0, 0, 0, 0

    for label in ['0E', '1E', '2E', '3E', '4E']:
        if label in dataset_selection_counts and dataset_selection_counts[label] > 0:
            stats = performance_stats[label]
            total = stats['TP'] + stats['FP'] + stats['TN'] + stats['FN']

            # Calculate metrics
            accuracy = (stats['TP'] + stats['TN']) / total if total > 0 else 0
            precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
            recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0

            print(f"{label:<10} {total:<8} {stats['TP']:<8} {stats['FP']:<8} {stats['TN']:<8} {stats['FN']:<8} "
                  f"{accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}")

            # Accumulate overall stats
            overall_tp += stats['TP']
            overall_fp += stats['FP']
            overall_tn += stats['TN']
            overall_fn += stats['FN']

    # Overall metrics
    overall_total = overall_tp + overall_fp + overall_tn + overall_fn
    if overall_total > 0:
        overall_accuracy = (overall_tp + overall_tn) / overall_total
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0

        print(f"{'-'*80}")
        print(f"{'Overall':<10} {overall_total:<8} {overall_tp:<8} {overall_fp:<8} {overall_tn:<8} {overall_fn:<8} "
              f"{overall_accuracy:<10.3f} {overall_precision:<10.3f} {overall_recall:<10.3f}")

    print(f"\nTotal processing time: {total_processing_time:.2f} seconds")
    if speed > 0:
        print(f"Average speed: {speed:.2f}x real-time")
    print(f"Detection figures saved to: {output_dir}")
    print(f"Performance report saved to: {report_filename}")
    print(f"{'='*80}")

    # Write final performance report to file
    with open(report_filename, 'w') as report_file:
        report_file.write(f"{'='*80}\n")
        report_file.write(f"Performance Metrics\n")
        report_file.write(f"{'='*80}\n")
        report_file.write(f"{'Dataset':<10} {'Total':<8} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}\n")
        report_file.write(f"{'-'*80}\n")

        for label in ['0E', '1E', '2E', '3E', '4E']:
            if label in dataset_selection_counts and dataset_selection_counts[label] > 0:
                stats = performance_stats[label]
                total = stats['TP'] + stats['FP'] + stats['TN'] + stats['FN']
                accuracy = (stats['TP'] + stats['TN']) / total if total > 0 else 0
                precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
                recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0

                report_file.write(f"{label:<10} {total:<8} {stats['TP']:<8} {stats['FP']:<8} {stats['TN']:<8} {stats['FN']:<8} "
                                f"{accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}\n")

        overall_total = overall_tp + overall_fp + overall_tn + overall_fn
        if overall_total > 0:
            overall_accuracy = (overall_tp + overall_tn) / overall_total
            overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
            overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0

            report_file.write(f"{'-'*80}\n")
            report_file.write(f"{'Overall':<10} {overall_total:<8} {overall_tp:<8} {overall_fp:<8} {overall_tn:<8} {overall_fn:<8} "
                            f"{overall_accuracy:<10.3f} {overall_precision:<10.3f} {overall_recall:<10.3f}\n")


def process_from_redis(model, data, scaler, output_dir, redis_config,
                      stream_name='windows', consumer_group='detectors',
                      consumer_name='fft', log_interval=10):
    """
    Process windows from Redis stream for synchronized detection with FFT preprocessing.

    Args:
        model: Trained FFT FCN model
        data: Dictionary of evaluation datasets (0E-4E)
        scaler: Fitted RobustScaler
        output_dir: Directory to save detection figures
        redis_config: RedisConfig instance
        stream_name: Redis stream name
        consumer_group: Consumer group name
        consumer_name: Consumer name for this approach
        log_interval: Log metrics to console every N windows
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

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Performance tracking
    performance_stats = {
        '0E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        '1E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        '2E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        '3E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        '4E': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
    }

    # Create performance report file
    start_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    report_filename = os.path.join(output_dir, f"performance_report_{start_timestamp}.txt")
    print(f"Performance report: {report_filename}")
    print(f"Updating report every {log_interval} windows\n")

    total_detections = 0
    window_idx = 0
    processing_start = time.time()

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

            # Get window data
            if dataset_label not in data:
                print(f"⚠️  Warning: Dataset {dataset_label} not loaded, skipping")
                consumer.acknowledge(window_msg['message_id'])
                continue

            sensor_data = data[dataset_label][SENSOR].values
            window_data = sensor_data[start_idx:end_idx]

            # Split into 1-second windows for prediction
            n_second_windows = int(np.floor(len(window_data) / WINDOW))
            if n_second_windows == 0:
                print(f"⚠️  Warning: Window too small ({len(window_data)} samples), skipping")
                consumer.acknowledge(window_msg['message_id'])
                continue

            window_data_windowed = window_data[:n_second_windows * WINDOW].reshape((n_second_windows, WINDOW))

            # Apply FFT transformation
            window_data_fft = np.abs(np.fft.rfft(window_data_windowed, axis=1))[:, :FFT_FEATURES]
            window_data_fft[:, 0] = 0  # Zero out DC component

            # Apply scaling
            window_data_fft_scaled = scaler.transform(window_data_fft)

            # Predict
            predictions = model.predict(window_data_fft_scaled, verbose=0)
            predictions_binary = (predictions > UNBALANCE_THRESHOLD).astype(int).flatten()

            # Check if unbalance detected (majority voting)
            unbalance_detections = np.sum(predictions_binary)
            detection_ratio = unbalance_detections / len(predictions_binary)

            # Get current timestamp
            current_time = datetime.utcnow()
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")

            # Determine dataset name
            if dataset_label == '0E':
                selected_name = 'No Unbalance'
            else:
                level = ['1E', '2E', '3E', '4E'].index(dataset_label) + 1
                selected_name = f'Unbalance Level {level}'

            # If significant unbalance detected (>50% of windows), save figure
            if detection_ratio > 0.5:
                total_detections += 1

                print(f"\n  ⚠️  UNBALANCE DETECTED at window {window_idx}")
                print(f"      Source: {dataset_label} ({selected_name})")
                print(f"      Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"      Row index: {start_idx:,} - {end_idx:,}")
                print(f"      Detection ratio: {detection_ratio*100:.1f}%")
                print(f"      Mean prediction: {np.mean(predictions):.4f}")

                # Generate and save figure
                fig = plt.figure(figsize=(15, 10))

                # Plot 1: Full time window timeseries
                ax1 = plt.subplot(3, 1, 1)
                time_window_seconds = len(window_data) / SAMPLES_PER_SECOND
                time_axis = np.arange(len(window_data)) / SAMPLES_PER_SECOND
                ax1.plot(time_axis, window_data, lw=0.5)
                ax1.set_title(f"UNBALANCE DETECTED - Dataset {dataset_label} ({selected_name})\n"
                             f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                             f"Rows: {start_idx:,}-{end_idx:,} | Window: {time_window_seconds:.0f}s",
                             fontsize=14, fontweight='bold')
                ax1.set_xlabel("Time (seconds)")
                ax1.set_ylabel("Vibration Amplitude")
                ax1.grid(True, alpha=0.3)

                # Plot 2: Predictions over time
                ax2 = plt.subplot(3, 1, 2)
                pred_times = np.arange(len(predictions))
                ax2.plot(pred_times, predictions, marker='o', markersize=3, linewidth=1)
                ax2.axhline(y=UNBALANCE_THRESHOLD, color='r', linestyle='--',
                           label=f'Threshold ({UNBALANCE_THRESHOLD})')
                ax2.fill_between(pred_times, 0, 1, where=(predictions.flatten() > UNBALANCE_THRESHOLD),
                                alpha=0.3, color='red', label='Unbalance Detected')
                ax2.set_title(f"Predictions per Second (Detection Ratio: {detection_ratio*100:.1f}%)")
                ax2.set_xlabel("Second")
                ax2.set_ylabel("Prediction Score")
                ax2.set_ylim([-0.05, 1.05])
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Plot 3: FFT spectrum of highest prediction window
                ax3 = plt.subplot(3, 1, 3)
                max_pred_idx = np.argmax(predictions)
                sample_data = window_data_windowed[max_pred_idx]
                fft_spectrum = np.abs(np.fft.rfft(sample_data))[:FFT_FEATURES]
                freqs = np.fft.rfftfreq(WINDOW, 1/SAMPLES_PER_SECOND)[:FFT_FEATURES]
                ax3.plot(freqs, fft_spectrum, lw=0.8)
                ax3.set_title(f"FFT Spectrum - Highest Prediction Window (Prediction: {predictions[max_pred_idx][0]:.4f})")
                ax3.set_xlabel("Frequency (Hz)")
                ax3.set_ylabel("Magnitude")
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save figure
                filename = f"unbalance_detection_{dataset_label}_{timestamp_str}_window{window_idx}_row{start_idx}.png"
                output_path = os.path.join(output_dir, filename)
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"      Figure saved: {output_path}")

                # Log detection event to JSONL file
                detections_log = os.path.join(output_dir, "detections.jsonl")
                detection_event = {
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'window_idx': window_idx,
                    'dataset': dataset_label,
                    'dataset_name': selected_name,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'detection_ratio': detection_ratio if isinstance(detection_ratio, (int, float)) else detection_ratio.item(),
                    'mean_prediction': np.mean(predictions).item(),
                    'max_prediction': np.max(predictions).item(),
                    'figure_file': filename
                }
                with open(detections_log, 'a') as f:
                    f.write(json.dumps(detection_event) + '\n')

            # Track performance metrics
            ground_truth_positive = dataset_label != '0E'
            predicted_positive = detection_ratio > 0.5

            if ground_truth_positive and predicted_positive:
                performance_stats[dataset_label]['TP'] += 1
            elif ground_truth_positive and not predicted_positive:
                performance_stats[dataset_label]['FN'] += 1
            elif not ground_truth_positive and predicted_positive:
                performance_stats[dataset_label]['FP'] += 1
            else:
                performance_stats[dataset_label]['TN'] += 1

            # Acknowledge message
            consumer.acknowledge(window_msg['message_id'])

            # Log running metrics every N windows
            if (window_idx + 1) % log_interval == 0:
                elapsed_time = time.time() - processing_start
                print(f"\n{'='*80}")
                print(f"Running Performance Metrics (after {window_idx + 1} windows, {elapsed_time:.1f}s elapsed)")
                print(f"{'='*80}")
                print(f"{'Dataset':<10} {'Processed':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Accuracy':<8}")
                print(f"{'-'*80}")

                overall_tp, overall_fp, overall_tn, overall_fn = 0, 0, 0, 0

                for label in ['0E', '1E', '2E', '3E', '4E']:
                    stats = performance_stats[label]
                    total = stats['TP'] + stats['FP'] + stats['TN'] + stats['FN']
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
                with open(report_filename, 'w') as report_file:
                    report_file.write(f"{'='*80}\n")
                    report_file.write(f"Performance Metrics - FFT FCN Approach (Redis Consumer Mode)\n")
                    report_file.write(f"{'='*80}\n")
                    report_file.write(f"{'Dataset':<10} {'Total':<8} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}\n")
                    report_file.write(f"{'-'*80}\n")

                    for label in ['0E', '1E', '2E', '3E', '4E']:
                        stats = performance_stats[label]
                        total = stats['TP'] + stats['FP'] + stats['TN'] + stats['FN']
                        if total > 0:
                            accuracy = (stats['TP'] + stats['TN']) / total
                            precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
                            recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0

                            report_file.write(f"{label:<10} {total:<8} {stats['TP']:<8} {stats['FP']:<8} {stats['TN']:<8} {stats['FN']:<8} "
                                            f"{accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}\n")

                    # Write overall metrics
                    overall_total = overall_tp + overall_fp + overall_tn + overall_fn
                    if overall_total > 0:
                        overall_accuracy = (overall_tp + overall_tn) / overall_total
                        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
                        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0

                        report_file.write(f"{'-'*80}\n")
                        report_file.write(f"{'Overall':<10} {overall_total:<8} {overall_tp:<8} {overall_fp:<8} {overall_tn:<8} {overall_fn:<8} "
                                        f"{overall_accuracy:<10.3f} {overall_precision:<10.3f} {overall_recall:<10.3f}\n")

            window_idx += 1

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Processing interrupted by user (Ctrl+C)")
        print(f"Processed {window_idx} windows before interruption")

    # Final summary
    total_processing_time = time.time() - processing_start
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

    overall_tp, overall_fp, overall_tn, overall_fn = 0, 0, 0, 0

    for label in ['0E', '1E', '2E', '3E', '4E']:
        stats = performance_stats[label]
        total = stats['TP'] + stats['FP'] + stats['TN'] + stats['FN']
        if total > 0:
            accuracy = (stats['TP'] + stats['TN']) / total
            precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
            recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0

            print(f"{label:<10} {total:<8} {stats['TP']:<8} {stats['FP']:<8} {stats['TN']:<8} {stats['FN']:<8} "
                  f"{accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}")

            overall_tp += stats['TP']
            overall_fp += stats['FP']
            overall_tn += stats['TN']
            overall_fn += stats['FN']

    # Overall metrics
    overall_total = overall_tp + overall_fp + overall_tn + overall_fn
    if overall_total > 0:
        overall_accuracy = (overall_tp + overall_tn) / overall_total
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0

        print(f"{'-'*80}")
        print(f"{'Overall':<10} {overall_total:<8} {overall_tp:<8} {overall_fp:<8} {overall_tn:<8} {overall_fn:<8} "
              f"{overall_accuracy:<10.3f} {overall_precision:<10.3f} {overall_recall:<10.3f}")

    print(f"\nDetection figures saved to: {output_dir}")
    print(f"Performance report saved to: {report_filename}")
    print(f"{'='*80}")

    consumer.close()


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='FFT FCN-based Unbalance Detection with Real-time Anomaly Scanning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets at maximum speed
  python approach_2_fft.py --dataset all --speed 0

  # Process dataset 4E at real-time speed
  python approach_2_fft.py --dataset 4E --speed 1.0

  # Enable MCP server for real-time monitoring
  python approach_2_fft.py --dataset all --enable-mcp

  # Process exactly 100 windows then stop
  python approach_2_fft.py --dataset all --max-windows 100
        """)

    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'All', 'ALL', '0E', '1E', '2E', '3E', '4E'],
                       help='Dataset to evaluate: all, 0E (no unbalance), 1E, 2E, 3E, or 4E (max unbalance). Default: all')

    parser.add_argument('--speed', type=float, default=0,
                       help='Processing speed multiplier. 0=maximum speed (no delays), 1.0=real-time, 2.0=2x speed, etc. Default: 0 (max speed)')

    parser.add_argument('--time-window', type=int, default=60,
                       help='Time window size in seconds for processing. Default: 60 (1 minute)')

    parser.add_argument('--max-windows', type=int, default=None,
                       help='Maximum number of windows to process. Default: None (infinite with rollover, Ctrl+C to stop)')

    parser.add_argument('--normal-weight', type=float, default=0.9,
                       help='Weight for normal (0E) dataset in weighted random sampling (0.0-1.0). Only applies when --dataset=all. Default: 0.9 (90%% normal, 10%% unbalanced)')

    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log performance metrics to console every N windows. Default: 10')

    parser.add_argument('--enable-mcp', action='store_true',
                       help='Enable MCP server for real-time monitoring queries. Runs in background thread with HTTP transport.')

    parser.add_argument('--mcp-port', type=int, default=8000,
                       help='Port for MCP server HTTP transport. Default: 8000')

    parser.add_argument('--model-path', type=str,
                       default='../models/fft_fcn_4_layers.h5',
                       help='Path to trained model file. Default: ../models/fft_fcn_4_layers.h5')

    parser.add_argument('--data-url', type=str,
                       default='../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip',
                       help='Path or URL to dataset ZIP file')

    parser.add_argument('--output-dir', type=str, default='../figures/detections',
                       help='Directory to save detection figures. Default: ../figures/detections')

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
    parser.add_argument('--consumer-name', type=str, default='fft',
                       help='Consumer name for this approach (default: fft)')

    args = parser.parse_args()

    print("=" * 80)
    print("Machine Learning Based Unbalance Detection - Approach 2: FFT FCN")
    print("Real-time Anomaly Detection with 4-Layer FFT FCN")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    if args.dataset.upper() == 'ALL':
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

    # Start MCP server in background thread if enabled
    mcp_thread = None
    if args.enable_mcp:
        try:
            from mcp_server import run_mcp_server, set_detections_dir
            # Set the output directory for MCP server
            set_detections_dir(args.output_dir)
            mcp_thread = threading.Thread(
                target=lambda: run_mcp_server(transport="sse", port=args.mcp_port),
                daemon=True
            )
            mcp_thread.start()
            print("✓ MCP server started in background thread")
            print(f"  HTTP/SSE endpoint: http://localhost:{args.mcp_port}/sse")
            print("  Available for real-time monitoring queries")
            print()
        except ImportError as e:
            print(f"⚠️  Warning: Could not start MCP server: {e}")
            print("  Install fastmcp to enable: pip install fastmcp")
            print()
        except Exception as e:
            print(f"⚠️  Warning: MCP server failed to start: {e}")
            print()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("\nPlease ensure the reference model exists, or specify a different model with --model-path")
        return

    # Load only the datasets we need (much faster!)
    data = load_data(args.data_url, datasets_to_load=args.dataset)

    # Skip warm-up phase
    data = skip_warmup(data)

    # Prepare datasets for validation with FFT transformation
    X_val_fft, y_val, scaler = prepare_datasets_fft(data)

    # Load the pre-trained model
    print("=" * 80)
    print("STEP 1: Load Pre-trained Model")
    print("=" * 80)
    print(f"Loading reference model from: {args.model_path}")
    model = load_model(args.model_path)

    # Quick validation (if we have validation data)
    if len(X_val_fft) > 0:
        print("\n" + "=" * 80)
        print("STEP 2: Model Validation on Combined Evaluation Dataset")
        print("=" * 80)
        val_result = model.evaluate(X_val_fft, y_val, verbose=1)
        print(f"\nValidation Accuracy: {val_result[1]*100:.2f}%")
        print(f"Validation Loss: {val_result[0]:.4f}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process timeseries data
    print("\n" + "=" * 80)
    print("STEP 3: Real-time Unbalance Detection")
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
            data=data,
            scaler=scaler,
            output_dir=args.output_dir,
            redis_config=redis_config,
            stream_name=args.redis_stream,
            consumer_group=args.consumer_group,
            consumer_name=args.consumer_name,
            log_interval=args.log_interval
        )
    else:
        # Standalone mode - weighted random sampling
        print("Mode: Standalone (Weighted Random Sampling)")
        print()

        process_with_weighted_sampling(
            model=model,
            data=data,
            scaler=scaler,
            output_dir=args.output_dir,
            speed=args.speed,
            time_window_seconds=args.time_window,
            max_windows=args.max_windows,
            normal_weight=args.normal_weight,
            log_interval=args.log_interval
        )


if __name__ == "__main__":
    main()
