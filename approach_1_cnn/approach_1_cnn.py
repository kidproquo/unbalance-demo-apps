"""
Machine Learning Based Unbalance Detection of a Rotating Shaft Using Vibration Data
Notebook 2 of 5 - Approach 1: Convolutional Neural Network on Raw Sensor Data

Oliver Mey, Willi Neudeck, André Schneider and Olaf Enge-Rosenblatt
Fraunhofer IIS/EAS, Fraunhofer Institute for Integrated Circuits,
Division Engineering of Adaptive Systems, Dresden, Germany

This Python script is converted from the Jupyter Notebook that is part of a paper
submission to the 25th IEEE International Conference on Emerging Technologies and
Factory Automation, ETFA 2020.

Modified to:
1. Train only the most accurate 3-layer CNN model
2. Process evaluation data as timeseries with 1-minute windows
3. Detect unbalance anomalies in real-time
4. Generate and save figures for detected unbalances
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sc
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import os
import argparse
import time
import json
import threading
from sklearn.model_selection import train_test_split
from scipy.stats import mode

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dense, Dropout
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from datetime import datetime, timedelta

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


# CNN-specific configuration
LABELS = {'no_unbalance': 0, 'unbalance': 1}
SECONDS_PER_ANALYSIS = 1.0
WINDOW = int(SAMPLES_PER_SECOND * SECONDS_PER_ANALYSIS)
N_CONV_LAYERS = 3  # Use the most accurate 3-layer CNN model
UNBALANCE_THRESHOLD = 0.5  # Prediction threshold for unbalance detection


def get_features(data, label):
    """
    Extract features from data by windowing.

    Args:
        data: Pandas Series with sensor data
        label: Label for the data ('no_unbalance' or 'unbalance')

    Returns:
        Tuple of (X, y) - features and labels
    """
    n = int(np.floor(len(data) / WINDOW))
    data = data[:int(n) * WINDOW]
    X = data.values.reshape((n, WINDOW))
    y = np.ones(n) * LABELS[label]
    return X, y


def prepare_datasets(data):
    """
    Prepare validation datasets from loaded evaluation data.

    Args:
        data: Dictionary of dataframes with evaluation datasets (0E-4E)

    Returns:
        Tuple of (X_val, y_val) - combined validation features and labels
    """
    print("\nPreparing validation datasets...")

    # Only prepare the datasets that were loaded
    X_vals = []
    y_vals = []

    for label in ['0E', '1E', '2E', '3E', '4E']:
        if label in data:
            label_type = "no_unbalance" if label == '0E' else "unbalance"
            X, y = get_features(data[label][SENSOR], label_type)
            X_vals.append(X)
            y_vals.append(y)
            print(f"  {label}: {X.shape[0]} samples")

    # Concatenate all loaded datasets for overall validation
    if len(X_vals) > 0:
        X_val = np.concatenate(X_vals)
        y_val = np.concatenate(y_vals)
        print(f"\n  Combined validation shape: {X_val.shape}, labels: {y_val.shape}")
    else:
        # If no data loaded, return empty arrays
        X_val = np.array([])
        y_val = np.array([])

    return X_val, y_val


def create_cnn_model(n_conv_layers, input_shape):
    """
    Create a CNN model with specified number of convolutional layers.

    Args:
        n_conv_layers: Number of convolutional layers
        input_shape: Shape of input data

    Returns:
        Keras model
    """
    n_dense_units = 128
    dropout_rate = 0.0
    use_batch_normalization = True
    filter_size = 9

    X_in = Input(shape=input_shape)
    x = X_in

    for j in range(n_conv_layers):
        x = Conv1D(filters=(j+1)*10,
                   kernel_size=filter_size,
                   strides=1,
                   activation='linear',
                   kernel_initializer='he_uniform')(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = MaxPooling1D(pool_size=5, strides=2)(x)

    x = Flatten()(x)
    x = Dense(units=n_dense_units, activation='linear')(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x)
    X_out = Dense(units=1, activation='sigmoid')(x)

    classifier = Model(X_in, X_out)
    return classifier


def train_model(X, y, model_path, n_conv_layers=3):
    """
    Train the 3-layer CNN model (most accurate configuration).

    Args:
        X: Training features
        y: Training labels
        model_path: Path to save model
        n_conv_layers: Number of convolutional layers (default: 3)

    Returns:
        Path to the trained model
    """
    print(f"\n{'='*80}")
    print(f"Training {n_conv_layers}-layer CNN model (most accurate configuration)")
    print(f"{'='*80}")

    # Calculate class weights
    weight_for_0 = len(y) / (2 * len(y[y == 0]))
    weight_for_1 = len(y) / (2 * len(y[y == 1]))
    class_weight = {0: weight_for_0, 1: weight_for_1}

    # Training parameters
    train_test_ratio = 0.9
    learning_rate = 0.0001
    n_epochs = 100

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-train_test_ratio, random_state=0)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Class weights: no_unbalance={weight_for_0:.3f}, unbalance={weight_for_1:.3f}")

    # Create model
    classifier = create_cnn_model(n_conv_layers, (X_train.shape[1], 1))
    classifier.summary()

    # Compile
    classifier.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Setup checkpoint
    best_model_filepath = f"{model_path}/cnn_{n_conv_layers}_layers.h5"
    checkpoint = ModelCheckpoint(
        best_model_filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    # Train
    print(f"\nTraining for {n_epochs} epochs...")
    classifier.fit(
        X_train, y_train,
        epochs=n_epochs,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
        class_weight=class_weight
    )

    print(f"\n{'='*80}")
    print(f"Model saved to: {best_model_filepath}")
    print(f"{'='*80}")

    return best_model_filepath


def evaluate_models(X_val_separated, y_val_separated, X_val, y_val, model_path):
    """
    Evaluate trained models on validation datasets.

    Args:
        X_val_separated: List of validation datasets for each unbalance level
        y_val_separated: List of labels for each unbalance level
        X_val: Combined validation dataset
        y_val: Combined validation labels
        model_path: Path to load models from

    Returns:
        Tuple of (accuracies, accuracies_all) - per-dataset and overall accuracies
    """
    print("\nEvaluating models...")

    # Reshape validation data
    X_val_reshaped = [np.reshape(x, (x.shape[0], x.shape[1], 1)) for x in X_val_separated]
    X_val_combined = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    accuracies = []
    accuracies_all = []

    for layer_n in range(1, 5):
        print(f"\n  Evaluating {layer_n}-layer model...")
        filepath = f"{model_path}/cnn_{layer_n}_layers.h5"
        model_i = load_model(filepath)

        val_acc_1 = model_i.evaluate(X_val_reshaped[0], y_val_separated[0], verbose=0)[1]
        val_acc_2 = model_i.evaluate(X_val_reshaped[1], y_val_separated[1], verbose=0)[1]
        val_acc_3 = model_i.evaluate(X_val_reshaped[2], y_val_separated[2], verbose=0)[1]
        val_acc_4 = model_i.evaluate(X_val_reshaped[3], y_val_separated[3], verbose=0)[1]
        val_acc_5 = model_i.evaluate(X_val_reshaped[4], y_val_separated[4], verbose=0)[1]
        val_acc_all = model_i.evaluate(X_val_combined, y_val, verbose=0)[1]

        accuracies_layer_i = [val_acc_1, val_acc_2, val_acc_3, val_acc_4, val_acc_5]
        accuracies.append(accuracies_layer_i)
        accuracies_all.append(val_acc_all)

        print(f"    Accuracies: {[f'{a:.4f}' for a in accuracies_layer_i]}")
        print(f"    Overall accuracy: {val_acc_all:.4f}")

    return np.array(accuracies), np.array(accuracies_all)


def v2rpm(v):
    """Convert voltage to RPM."""
    return 212 * v + 209


def evaluate_rotation_speed_dependency(X_val, y_val, model_path):
    """
    Evaluate models based on rotation speed.

    Args:
        X_val: Validation features
        y_val: Validation labels
        model_path: Path to load models from

    Returns:
        Tuple of (rpm_borders, errors_per_rpm_range) for all models
    """
    print("\nEvaluating rotation speed dependency...")

    # Reconstruct voltage/RPM profile
    fade_in = np.arange(0.0, 4.0, 4.0 / (3 * 4096))
    measurement_cycle = np.repeat(np.arange(4.0, 8.2, 0.1), 4096 * 20.0)
    measurement = np.concatenate([fade_in, np.tile(measurement_cycle, 3)])
    measurement1 = measurement[SKIP:]
    measurement1 = measurement1[:int(len(measurement1) / 4096) * 4096].reshape(-1, 4096)
    voltages_measurement = mode(measurement1, axis=1)[0]

    # Calculate RPMs for validation data
    voltages_used = voltages_measurement[:len(X_val)]
    rpms_used = v2rpm(voltages_used)

    # Reshape validation data
    X_val_reshaped = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    # Evaluate in RPM ranges
    rpm_borders = np.arange(1050, 1975, 25)
    errors_per_rpm_range = []

    for layer_n in range(1, 5):
        filepath = f"{model_path}/cnn_{layer_n}_layers.h5"
        model = load_model(filepath)

        errors_range = []
        for i in range(len(rpm_borders) - 1):
            eval_inds = np.where((rpms_used > rpm_borders[i]) &
                                (rpms_used < rpm_borders[i+1]))[0]
            if len(eval_inds) > 0:
                predictions = model.predict(X_val_reshaped[eval_inds], verbose=0)
                accuracy = 1 - np.mean(np.abs(
                    np.int32(predictions > 0.5).reshape(-1) - y_val[eval_inds]))
                errors_range.append(accuracy)
            else:
                errors_range.append(0)

        errors_per_rpm_range.append(errors_range)
        print(f"  {layer_n}-layer model: mean RPM-based accuracy = {np.mean(errors_range):.4f}")

    return rpm_borders, errors_per_rpm_range


def process_with_weighted_sampling(model, data, output_dir, speed, time_window_seconds, max_windows=None, normal_weight=0.9, log_interval=10):
    """
    Process data with weighted random sampling.
    Configurable weight on 0E (no unbalance), remaining weight on 1E-4E (random unbalanced).

    Args:
        model: Trained CNN model
        data: Dictionary of evaluation datasets
        output_dir: Directory to save detection figures
        speed: Processing speed multiplier
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
            # Use configured normal_weight for selecting 0E vs 1E-4E
            rand_val = np.random.random()

            if rand_val < normal_weight and '0E' in data:
                # Select 0E (no unbalance)
                selected_label = '0E'
                selected_name = 'No Unbalance'
            else:
                # Select randomly from 1E-4E (unbalanced datasets)
                unbalanced_labels = [label for label in ['1E', '2E', '3E', '4E'] if label in data]
                if not unbalanced_labels:
                    # If no unbalanced data available, use 0E
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
            window_data_windowed = window_data[:n_second_windows * WINDOW].reshape((n_second_windows, WINDOW, 1))

            # Predict on all 1-second windows in this time window
            predictions = model.predict(window_data_windowed, verbose=0)
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

                # Plot 3: Sample window with highest prediction
                ax3 = plt.subplot(3, 1, 3)
                max_pred_idx = np.argmax(predictions)
                sample_start = max_pred_idx * WINDOW
                sample_end = sample_start + WINDOW
                sample_data = window_data[sample_start:sample_end]
                sample_time = np.arange(len(sample_data)) / SAMPLES_PER_SECOND
                ax3.plot(sample_time, sample_data, lw=0.8)
                ax3.set_title(f"Highest Prediction Window (Prediction: {predictions[max_pred_idx][0]:.4f})")
                ax3.set_xlabel("Time (seconds)")
                ax3.set_ylabel("Vibration Amplitude")
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


def process_timeseries_data(model, data, output_dir='../../figures/detections',
                           datasets='all', speed=1.0, time_window_seconds=60, max_windows=None, normal_weight=0.9, log_interval=10):
    """
    Process evaluation data as timeseries with configurable time windows.
    Detect unbalance anomalies and save figures for detections.

    Args:
        model: Trained CNN model
        data: Dictionary of evaluation datasets (0E-4E)
        output_dir: Directory to save detection figures
        datasets: Which dataset(s) to process - 'all', '0E', '1E', '2E', '3E', or '4E'
        speed: Processing speed multiplier (1.0 = real-time, 0 = max speed)
        time_window_seconds: Time window size in seconds (default: 60)
        max_windows: Maximum number of windows to process (None = infinite with rollover)
        normal_weight: Weight for 0E dataset in weighted random sampling (default: 0.9)
        log_interval: Log metrics to console every N windows (default: 10)
    """
    print(f"\n{'='*80}")
    print("Processing Evaluation Data as Timeseries")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # Determine which datasets to process
    all_labels = ['0E', '1E', '2E', '3E', '4E']
    all_names = ['No Unbalance', 'Unbalance Level 1', 'Unbalance Level 2',
                 'Unbalance Level 3', 'Unbalance Level 4']

    if datasets.upper() == 'ALL':
        # When processing ALL datasets, use weighted random sampling
        print("Mode: Weighted Random Sampling")
        print(f"  {normal_weight*100:.0f}% weight on 0E (no unbalance)")
        print(f"  {(1-normal_weight)*100:.0f}% weight on 1E-4E (unbalanced, randomly selected)")
        print(f"  Time window: {time_window_seconds} seconds")
        print()

        process_with_weighted_sampling(model, data, output_dir, speed, time_window_seconds, max_windows, normal_weight, log_interval)
        return
    else:
        # Process single dataset sequentially
        dataset_upper = datasets.upper()
        if dataset_upper not in all_labels:
            print(f"Error: Invalid dataset '{datasets}'. Must be one of: {', '.join(all_labels)} or 'all'")
            return
        idx = all_labels.index(dataset_upper)
        dataset_labels = [all_labels[idx]]
        dataset_names = [all_names[idx]]

    # Display processing configuration
    print(f"Dataset(s): {', '.join(dataset_labels)}")
    print(f"Time window: {time_window_seconds} seconds")
    if speed == 0:
        print(f"Speed: Maximum (no delays)")
    elif speed == 1.0:
        print(f"Speed: Real-time ({time_window_seconds}s window = {time_window_seconds}s processing)")
    else:
        print(f"Speed: {speed}x real-time")
    print()

    # Calculate time window in samples
    time_window_samples = int(time_window_seconds * SAMPLES_PER_SECOND)

    total_detections = 0
    processing_start = time.time()

    for dataset_label, dataset_name in zip(dataset_labels, dataset_names):
        print(f"\n--- Processing Dataset {dataset_label}: {dataset_name} ---")

        sensor_data = data[dataset_label][SENSOR].values
        total_samples = len(sensor_data)

        print(f"Total samples: {total_samples:,}")
        print(f"Duration: {total_samples / SAMPLES_PER_SECOND / 60:.2f} minutes")
        print(f"Processing in {time_window_seconds}s windows ({time_window_samples:,} samples each)...")

        # Process in time window chunks
        n_windows = int(np.floor(total_samples / time_window_samples))

        for window_idx in range(n_windows):
            window_process_start = time.time()

            # Extract time window of data
            start_idx = window_idx * time_window_samples
            end_idx = start_idx + time_window_samples
            window_data = sensor_data[start_idx:end_idx]

            # Split into 1-second windows for prediction
            n_second_windows = int(np.floor(len(window_data) / WINDOW))
            window_data_windowed = window_data[:n_second_windows * WINDOW].reshape((n_second_windows, WINDOW, 1))

            # Predict on all 1-second windows in this time window
            predictions = model.predict(window_data_windowed, verbose=0)
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

                print(f"\n  ⚠️  UNBALANCE DETECTED at window {window_idx + 1}")
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
                ax1.set_title(f"UNBALANCE DETECTED - Dataset {dataset_label} ({dataset_name})\n"
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

                # Plot 3: Sample window with highest prediction
                ax3 = plt.subplot(3, 1, 3)
                max_pred_idx = np.argmax(predictions)
                sample_start = max_pred_idx * WINDOW
                sample_end = sample_start + WINDOW
                sample_data = window_data[sample_start:sample_end]
                sample_time = np.arange(len(sample_data)) / SAMPLES_PER_SECOND
                ax3.plot(sample_time, sample_data, lw=0.8)
                ax3.set_title(f"Highest Prediction Window (Prediction: {predictions[max_pred_idx][0]:.4f})")
                ax3.set_xlabel("Time (seconds)")
                ax3.set_ylabel("Vibration Amplitude")
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save figure
                filename = f"unbalance_detection_{dataset_label}_{timestamp_str}_window{window_idx}_row{start_idx}.png"
                output_path = os.path.join(output_dir, filename)
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"      Figure saved: {output_path}")

            # Simulate real-time processing speed if requested
            if speed > 0:
                window_process_time = time.time() - window_process_start
                target_time = time_window_seconds / speed
                sleep_time = target_time - window_process_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print(f"\nDataset {dataset_label} complete: {total_samples:,} samples processed")

    # Print processing summary
    total_processing_time = time.time() - processing_start
    print(f"\n{'='*80}")
    print(f"Timeseries Processing Complete")
    print(f"Total unbalance detections: {total_detections}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    if speed > 0:
        print(f"Average speed: {speed:.2f}x real-time")
    print(f"Detection figures saved to: {output_dir}")
    print(f"{'='*80}")


def process_from_redis(model, output_dir, redis_config, stream_name,
                       consumer_group, consumer_name, log_interval=10, data=None):
    """
    Process windows from Redis stream for synchronized detection across approaches.

    Args:
        model: Pre-trained CNN model
        output_dir: Directory to save detection figures
        redis_config: RedisConfig object
        stream_name: Redis stream name to consume from
        consumer_group: Consumer group name
        consumer_name: Unique consumer name for this process
        log_interval: How often to log performance metrics
        data: Optional dictionary of {label: DataFrame} for fallback (if sensor data not in Redis)
    """
    import json
    from datetime import datetime, timezone

    print(f"\n{'='*80}")
    print(f"Redis Consumer Mode - Synchronized Processing")
    print(f"{'='*80}")
    print(f"  Redis: {redis_config.host}:{redis_config.port}")
    print(f"  Stream: {stream_name}")
    print(f"  Consumer Group: {consumer_group}")
    print(f"  Consumer Name: {consumer_name}")
    print(f"  Output Directory: {output_dir}")
    print()

    # Clear and recreate output directory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"  Cleared existing output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Create Redis consumer
    consumer = WindowConsumer(
        redis_config,
        stream_name=stream_name,
        consumer_group=consumer_group,
        consumer_name=consumer_name
    )

    # Wait for stream to be created by coordinator
    if not consumer.wait_for_stream(timeout_s=30):
        print("ERROR: Redis stream not found. Make sure data_coordinator.py is running!")
        return

    # Track performance metrics per dataset
    all_labels = ['0E', '1E', '2E', '3E', '4E']
    dataset_stats = {label: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'total': 0}
                     for label in all_labels}

    # Create timestamped report filename
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    report_filename = os.path.join(output_dir, f"performance_report_{timestamp}.txt")
    detections_log = os.path.join(output_dir, "detections.jsonl")

    total_detections = 0
    windows_processed = 0
    start_time = time.time()

    print(f"Waiting for windows from coordinator...")
    print(f"Performance report: {report_filename}")
    print(f"Updating report every {log_interval} windows\n")

    try:
        while True:
            # Read next window from Redis (blocks for up to 5 seconds)
            window_msg = consumer.read_window(block_ms=5000)

            if window_msg is None:
                # Timeout - no new messages
                continue

            # Extract window information
            dataset_label = window_msg['dataset']
            window_idx = window_msg['window_idx']
            start_idx = window_msg['start_idx']
            end_idx = window_msg['end_idx']
            message_id = window_msg['message_id']

            # Get window data from Redis message (if available) or from loaded data
            if 'sensor_data' in window_msg:
                # Sensor data from Redis: columns are [Measured_RPM, Vibration_1, Vibration_2, Vibration_3]
                # CNN uses Vibration_1 (column index 1)
                window_data = window_msg['sensor_data'][:, 1]
            elif data is not None and dataset_label in data:
                # Fallback to loaded data
                window_data = data[dataset_label][SENSOR].iloc[start_idx:end_idx].values
            else:
                print(f"⚠️  No sensor data available for window {window_idx}, skipping")
                consumer.acknowledge(message_id)
                continue

            # Reshape for CNN: (samples, 1) - CNN expects 2D input
            window_data_reshaped = window_data.reshape(-1, 1)

            # Predict
            prediction = model.predict(np.array([window_data_reshaped]), verbose=0)[0][0]

            # Determine if unbalance detected
            detected = prediction > 0.5

            # Track metrics
            ground_truth_positive = dataset_label != '0E'  # True if 1E-4E

            if ground_truth_positive and detected:
                dataset_stats[dataset_label]['TP'] += 1
            elif not ground_truth_positive and not detected:
                dataset_stats[dataset_label]['TN'] += 1
            elif not ground_truth_positive and detected:
                dataset_stats[dataset_label]['FP'] += 1
                total_detections += 1
            else:  # ground_truth_positive and not detected
                dataset_stats[dataset_label]['FN'] += 1

            dataset_stats[dataset_label]['total'] += 1

            # Generate figure if unbalance detected
            if detected:
                total_detections += 1
                current_time = datetime.now(timezone.utc)

                print(f"\n  ⚠️  UNBALANCE DETECTED")
                print(f"      Source: {dataset_label}")
                print(f"      Window: {window_idx}")
                print(f"      Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"      Prediction: {prediction:.4f}")

                # Create detection figure
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                fig.suptitle(f'Unbalance Detection - {dataset_label}\\n'
                             f'Window {window_idx} | Rows {start_idx:,}-{end_idx:,} | '
                             f'{current_time.strftime("%Y-%m-%d %H:%M:%S")} UTC',
                             fontsize=14)

                # Panel 1: Raw signal
                time_axis = np.arange(len(window_data)) / SAMPLES_PER_SECOND
                axes[0].plot(time_axis, window_data, linewidth=0.5)
                axes[0].set_ylabel('Amplitude')
                axes[0].set_title('Raw Vibration Signal')
                axes[0].grid(True, alpha=0.3)

                # Panel 2: CNN prediction
                axes[1].bar(['Prediction'], [prediction], color='red' if detected else 'green')
                axes[1].axhline(y=0.5, color='orange', linestyle='--', label='Threshold')
                axes[1].set_ylabel('Probability')
                axes[1].set_title(f'CNN Prediction (Score: {prediction:.4f})')
                axes[1].set_ylim([0, 1])
                axes[1].legend()
                axes[1].grid(True, alpha=0.3, axis='y')

                plt.tight_layout()

                # Save figure
                timestamp_str = current_time.strftime('%Y%m%d_%H%M%S')
                filename = f"unbalance_detection_{dataset_label}_{timestamp_str}_row{start_idx}.png"
                output_path = os.path.join(output_dir, filename)
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"      Figure saved: {output_path}")

                # Log detection event
                detection_event = {
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'window_idx': window_idx,
                    'dataset': dataset_label,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'prediction': prediction if isinstance(prediction, (int, float)) else prediction.item(),
                    'figure_file': filename,
                    'approach': 'cnn'
                }
                with open(detections_log, 'a') as f:
                    f.write(json.dumps(detection_event) + '\n')

            # Acknowledge message
            consumer.acknowledge(message_id)
            windows_processed += 1

            # Log performance metrics periodically
            if windows_processed % log_interval == 0:
                elapsed_time = time.time() - start_time

                print(f"\n{'='*80}")
                print(f"Performance Metrics (after {windows_processed} windows, {elapsed_time:.1f}s elapsed)")
                print(f"{'='*80}")
                print(f"{'Dataset':<10} {'Processed':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Accuracy':<8}")
                print(f"{'-'*80}")

                for label in sorted(dataset_stats.keys()):
                    stats = dataset_stats[label]
                    total = stats['total']
                    if total > 0:
                        accuracy = (stats['TP'] + stats['TN']) / total
                        print(f"{label:<10} {total:<10} {stats['TP']:<6} {stats['FP']:<6} {stats['TN']:<6} {stats['FN']:<6} {accuracy:<8.3f}")

                # Overall metrics
                overall_tp = sum(s['TP'] for s in dataset_stats.values())
                overall_fp = sum(s['FP'] for s in dataset_stats.values())
                overall_tn = sum(s['TN'] for s in dataset_stats.values())
                overall_fn = sum(s['FN'] for s in dataset_stats.values())
                overall_total = sum(s['total'] for s in dataset_stats.values())
                overall_accuracy = (overall_tp + overall_tn) / overall_total if overall_total > 0 else 0

                print(f"{'-'*80}")
                print(f"{'Overall':<10} {overall_total:<10} {overall_tp:<6} {overall_fp:<6} {overall_tn:<6} {overall_fn:<6} {overall_accuracy:<8.3f}")
                print(f"{'='*80}\n")

                # Write to report file
                with open(report_filename, 'w') as f:
                    f.write(f"Performance Report - Approach 1: CNN (Redis Mode)\\n")
                    f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\\n")
                    f.write(f"Windows Processed: {windows_processed}\\n")
                    f.write(f"Elapsed Time: {elapsed_time:.1f} seconds\\n")
                    f.write(f"\\n{'='*80}\\n")
                    f.write(f"{'Dataset':<10} {'Total':<8} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}\\n")
                    f.write(f"{'-'*80}\\n")

                    for label in sorted(dataset_stats.keys()):
                        stats = dataset_stats[label]
                        total = stats['total']
                        if total > 0:
                            accuracy = (stats['TP'] + stats['TN']) / total
                            precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
                            recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0
                            f.write(f"{label:<10} {total:<8} {stats['TP']:<8} {stats['FP']:<8} {stats['TN']:<8} {stats['FN']:<8} "
                                    f"{accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f}\\n")

                    if overall_total > 0:
                        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
                        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
                        f.write(f"{'-'*80}\\n")
                        f.write(f"{'Overall':<10} {overall_total:<8} {overall_tp:<8} {overall_fp:<8} {overall_tn:<8} {overall_fn:<8} "
                                f"{overall_accuracy:<10.3f} {overall_precision:<10.3f} {overall_recall:<10.3f}\\n")

    except KeyboardInterrupt:
        print(f"\\n\\n⚠️  Processing interrupted by user (Ctrl+C)")
        print(f"Processed {windows_processed} windows before interruption")

    finally:
        consumer.close()

        # Final summary
        total_time = time.time() - start_time
        print(f"\\n{'='*80}")
        print(f"Redis Consumer Mode Complete")
        print(f"Total windows processed: {windows_processed}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total unbalance detections: {total_detections}")
        print(f"Detection figures saved to: {output_dir}")
        print(f"Performance report saved to: {report_filename}")
        print(f"{'='*80}")


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='CNN-based Unbalance Detection with Real-time Anomaly Scanning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets at maximum speed
  python approach_1_cnn.py --dataset all --speed 0

  # Process dataset 4E at real-time speed
  python approach_1_cnn.py --dataset 4E --speed 1.0

  # Process dataset 1E at 2x real-time speed
  python approach_1_cnn.py --dataset 1E --speed 2.0

  # Use custom trained model
  python approach_1_cnn.py --dataset 2E --model-path /path/to/custom_model.h5

  # Process exactly 100 windows then stop
  python approach_1_cnn.py --dataset all --max-windows 100

  # Infinite processing (runs forever with rollover, Ctrl+C to stop)
  python approach_1_cnn.py --dataset all --speed 1.0
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
                       default='../models/cnn_3_layers.h5',
                       help='Path to trained model file. Default: ../models/cnn_3_layers.h5')

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
    parser.add_argument('--consumer-name', type=str, default='cnn',
                       help='Consumer name for this approach (default: cnn)')

    args = parser.parse_args()

    print("=" * 80)
    print("Machine Learning Based Unbalance Detection - Approach 1: CNN")
    print("Real-time Anomaly Detection with 3-Layer CNN")
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
                target=lambda: run_mcp_server(transport="streamable-http", port=args.mcp_port, host="0.0.0.0"),
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

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("\nPlease ensure the reference model exists, or specify a different model with --model-path")
        return

    # In Redis mode, skip data loading - data comes from coordinator via Redis
    if args.redis_mode:
        print("=" * 80)
        print("Redis Mode - Skipping Data Loading")
        print("=" * 80)
        print("Data will be received from coordinator via Redis stream")
        data = None
        X_val, y_val = np.array([]), np.array([])
    else:
        # Load only the datasets we need (much faster!)
        data = load_data(args.data_url, datasets_to_load=args.dataset)

        # Skip warm-up phase
        data = skip_warmup(data)

        # Prepare datasets for validation
        X_val, y_val = prepare_datasets(data)

    # Load the pre-trained model
    print("=" * 80)
    print("STEP 1: Load Pre-trained Model")
    print("=" * 80)
    print(f"Loading reference model from: {args.model_path}")
    model = load_model(args.model_path)

    # Quick validation (if we have validation data)
    if len(X_val) > 0:
        print("\n" + "=" * 80)
        print("STEP 2: Validate Model Performance")
        print("=" * 80)
        X_val_reshaped = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
        val_loss, val_acc = model.evaluate(X_val_reshaped, y_val, verbose=0)
        print(f"Validation accuracy: {val_acc*100:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")
    else:
        val_acc = 0.0
        print("\n" + "=" * 80)
        print("STEP 2: Skipping Validation (no validation data loaded)")
        print("=" * 80)

    # Process evaluation data as timeseries
    print("\n" + "=" * 80)
    print("STEP 3: Real-time Anomaly Detection")
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
            data=data  # Optional fallback if sensor data not in Redis
        )
    else:
        # Standalone mode - weighted random sampling
        print("Mode: Standalone (Weighted Random Sampling)")
        print()

        process_timeseries_data(model, data, output_dir=args.output_dir,
                               datasets=args.dataset, speed=args.speed,
                               time_window_seconds=args.time_window,
                               max_windows=args.max_windows,
                               normal_weight=args.normal_weight,
                               log_interval=args.log_interval)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nModel: {N_CONV_LAYERS}-layer CNN")
    if len(X_val) > 0:
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Detection Threshold: {UNBALANCE_THRESHOLD}")
    print(f"Window Size: {args.time_window} seconds ({args.time_window * SAMPLES_PER_SECOND:,} samples)")
    print(f"\nCheck the {args.output_dir}/ directory for detected anomalies.")


if __name__ == "__main__":
    main()
