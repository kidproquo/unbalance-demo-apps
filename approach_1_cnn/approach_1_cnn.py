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
import zipfile
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import os
import argparse
import time
from sklearn.model_selection import train_test_split
from scipy.stats import mode

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dense, Dropout
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from datetime import datetime, timedelta


# Configuration
LABELS = {'no_unbalance': 0, 'unbalance': 1}
SENSOR = 'Vibration_1'
SAMPLES_PER_SECOND = 4096
SECONDS_PER_ANALYSIS = 1.0
WINDOW = int(SAMPLES_PER_SECOND * SECONDS_PER_ANALYSIS)
SKIP = 50000  # Skip warm-up phase
N_CONV_LAYERS = 3  # Use the most accurate 3-layer CNN model
MINUTE_WINDOW = 60 * SAMPLES_PER_SECOND  # 1 minute of data
UNBALANCE_THRESHOLD = 0.5  # Prediction threshold for unbalance detection


def load_data(url, datasets_to_load='all'):
    """
    Load measurement data from ZIP file.

    Args:
        url: Path to the ZIP file containing the dataset
        datasets_to_load: Which datasets to load - 'all', 'eval_only', or specific like '4E'

    Returns:
        Dictionary containing requested datasets
    """
    print("Loading measurement data...")
    data = {}

    # Determine which datasets to load
    if datasets_to_load.upper() == 'ALL':
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


def skip_warmup(data, skip=SKIP):
    """
    Skip warm-up phase of each measurement.

    Args:
        data: Dictionary of dataframes
        skip: Number of samples to skip (default: 50000)

    Returns:
        Dictionary of dataframes with warm-up phase removed
    """
    print(f"\nSkipping first {skip} samples (warm-up phase)...")
    processed_data = {}

    for label, df in data.items():
        processed_data[label] = df.iloc[skip:, :]
        print(f"  {label}: {len(processed_data[label])} rows remaining")

    return processed_data


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


def process_with_weighted_sampling(model, data, output_dir, speed):
    """
    Process data with weighted random sampling.
    75% weight on 0E (no unbalance), 25% on 1E-4E (random unbalanced).

    Args:
        model: Trained CNN model
        data: Dictionary of evaluation datasets
        output_dir: Directory to save detection figures
        speed: Processing speed multiplier
    """
    total_detections = 0
    start_time = datetime(2020, 1, 1, 0, 0, 0)
    processing_start = time.time()

    # Calculate how many minutes we can process from each dataset
    dataset_minutes = {}
    for label in ['0E', '1E', '2E', '3E', '4E']:
        if label in data:
            total_samples = len(data[label][SENSOR].values)
            n_minutes = int(np.floor(total_samples / MINUTE_WINDOW))
            dataset_minutes[label] = n_minutes
            print(f"{label}: {n_minutes} minutes available")

    if not dataset_minutes:
        print("No data available to process")
        return

    # Determine total number of minutes to process (use the maximum available)
    max_minutes = max(dataset_minutes.values())
    print(f"\nProcessing {max_minutes} minutes with weighted random sampling...")
    print()

    # Track stats
    dataset_selection_counts = {label: 0 for label in ['0E', '1E', '2E', '3E', '4E']}

    # Keep track of current position in each dataset
    dataset_positions = {label: 0 for label in dataset_minutes.keys()}

    for minute_idx in range(max_minutes):
        minute_process_start = time.time()

        # Weighted random selection
        # 75% chance of selecting 0E, 25% chance of selecting from 1E-4E
        rand_val = np.random.random()

        if rand_val < 0.75 and '0E' in data and dataset_positions['0E'] < dataset_minutes['0E']:
            # Select 0E (no unbalance)
            selected_label = '0E'
            selected_name = 'No Unbalance'
        else:
            # Select randomly from 1E-4E (unbalanced datasets)
            unbalanced_labels = [label for label in ['1E', '2E', '3E', '4E']
                                if label in data and dataset_positions[label] < dataset_minutes[label]]
            if not unbalanced_labels:
                # If no unbalanced data available, try 0E
                if '0E' in data and dataset_positions['0E'] < dataset_minutes['0E']:
                    selected_label = '0E'
                    selected_name = 'No Unbalance'
                else:
                    # No data left to process
                    break
            else:
                selected_label = np.random.choice(unbalanced_labels)
                level = ['1E', '2E', '3E', '4E'].index(selected_label) + 1
                selected_name = f'Unbalance Level {level}'

        # Track selection
        dataset_selection_counts[selected_label] += 1

        # Get the current minute from the selected dataset
        current_pos = dataset_positions[selected_label]
        sensor_data = data[selected_label][SENSOR].values

        start_idx = current_pos * MINUTE_WINDOW
        end_idx = start_idx + MINUTE_WINDOW
        minute_data = sensor_data[start_idx:end_idx]

        # Move to next minute in this dataset
        dataset_positions[selected_label] += 1

        # Split into 1-second windows (60 windows per minute)
        n_windows = int(np.floor(len(minute_data) / WINDOW))
        minute_data_windowed = minute_data[:n_windows * WINDOW].reshape((n_windows, WINDOW, 1))

        # Predict on all windows in this minute
        predictions = model.predict(minute_data_windowed, verbose=0)
        predictions_binary = (predictions > UNBALANCE_THRESHOLD).astype(int).flatten()

        # Check if unbalance detected (majority voting)
        unbalance_detections = np.sum(predictions_binary)
        detection_ratio = unbalance_detections / len(predictions_binary)

        # Calculate timestamp
        current_time = start_time + timedelta(seconds=minute_idx * 60)
        timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")

        # If significant unbalance detected (>50% of windows), save figure
        if detection_ratio > 0.5:
            total_detections += 1

            print(f"\n  ⚠️  UNBALANCE DETECTED at minute {minute_idx + 1}")
            print(f"      Source: {selected_label} ({selected_name})")
            print(f"      Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      Row index: {start_idx:,} - {end_idx:,}")
            print(f"      Detection ratio: {detection_ratio*100:.1f}%")
            print(f"      Mean prediction: {np.mean(predictions):.4f}")

            # Generate and save figure
            fig = plt.figure(figsize=(15, 10))

            # Plot 1: Full minute timeseries
            ax1 = plt.subplot(3, 1, 1)
            time_axis = np.arange(len(minute_data)) / SAMPLES_PER_SECOND
            ax1.plot(time_axis, minute_data, lw=0.5)
            ax1.set_title(f"UNBALANCE DETECTED - Dataset {selected_label} ({selected_name})\n"
                         f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                         f"Rows: {start_idx:,}-{end_idx:,}", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Vibration Amplitude")
            ax1.grid(True, alpha=0.3)

            # Plot 2: Predictions over time
            ax2 = plt.subplot(3, 1, 2)
            window_times = np.arange(len(predictions))
            ax2.plot(window_times, predictions, marker='o', markersize=3, linewidth=1)
            ax2.axhline(y=UNBALANCE_THRESHOLD, color='r', linestyle='--',
                       label=f'Threshold ({UNBALANCE_THRESHOLD})')
            ax2.fill_between(window_times, 0, 1, where=(predictions.flatten() > UNBALANCE_THRESHOLD),
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
            sample_data = minute_data[sample_start:sample_end]
            sample_time = np.arange(len(sample_data)) / SAMPLES_PER_SECOND
            ax3.plot(sample_time, sample_data, lw=0.8)
            ax3.set_title(f"Highest Prediction Window (Prediction: {predictions[max_pred_idx][0]:.4f})")
            ax3.set_xlabel("Time (seconds)")
            ax3.set_ylabel("Vibration Amplitude")
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            filename = f"unbalance_detection_{selected_label}_{timestamp_str}_minute{minute_idx}_row{start_idx}.png"
            output_path = os.path.join(output_dir, filename)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"      Figure saved: {output_path}")

        # Simulate real-time processing speed if requested
        if speed > 0:
            minute_process_time = time.time() - minute_process_start
            target_time = 60.0 / speed  # 60 seconds for 1x speed
            sleep_time = target_time - minute_process_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Print processing summary
    total_processing_time = time.time() - processing_start
    print(f"\n{'='*80}")
    print(f"Weighted Random Sampling Complete")
    print(f"Total minutes processed: {minute_idx + 1}")
    print(f"Total unbalance detections: {total_detections}")
    print(f"\nDataset Selection Statistics:")
    for label in ['0E', '1E', '2E', '3E', '4E']:
        if label in dataset_selection_counts and dataset_selection_counts[label] > 0:
            percentage = (dataset_selection_counts[label] / (minute_idx + 1)) * 100
            print(f"  {label}: {dataset_selection_counts[label]} times ({percentage:.1f}%)")
    print(f"\nTotal processing time: {total_processing_time:.2f} seconds")
    if speed > 0:
        print(f"Average speed: {speed:.2f}x real-time")
    print(f"Detection figures saved to: {output_dir}")
    print(f"{'='*80}")


def process_timeseries_data(model, data, output_dir='../../figures/detections',
                           datasets='all', speed=1.0):
    """
    Process evaluation data as timeseries with 1-minute windows.
    Detect unbalance anomalies and save figures for detections.

    Args:
        model: Trained CNN model
        data: Dictionary of evaluation datasets (0E-4E)
        output_dir: Directory to save detection figures
        datasets: Which dataset(s) to process - 'all', '0E', '1E', '2E', '3E', or '4E'
        speed: Processing speed multiplier (1.0 = real-time, 0 = max speed)
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
        print("  75% weight on 0E (no unbalance)")
        print("  25% weight on 1E-4E (unbalanced, randomly selected)")
        print()

        process_with_weighted_sampling(model, data, output_dir, speed)
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
    if speed == 0:
        print(f"Speed: Maximum (no delays)")
    elif speed == 1.0:
        print(f"Speed: Real-time (1 minute data = 60 seconds processing)")
    else:
        print(f"Speed: {speed}x real-time")
    print()

    total_detections = 0
    start_time = datetime(2020, 1, 1, 0, 0, 0)  # Reference start time
    processing_start = time.time()

    for dataset_label, dataset_name in zip(dataset_labels, dataset_names):
        print(f"\n--- Processing Dataset {dataset_label}: {dataset_name} ---")

        sensor_data = data[dataset_label][SENSOR].values
        total_samples = len(sensor_data)

        print(f"Total samples: {total_samples:,}")
        print(f"Duration: {total_samples / SAMPLES_PER_SECOND / 60:.2f} minutes")
        print(f"Processing in 1-minute windows ({MINUTE_WINDOW:,} samples each)...")

        # Process in 1-minute chunks
        n_minutes = int(np.floor(total_samples / MINUTE_WINDOW))

        for minute_idx in range(n_minutes):
            minute_process_start = time.time()

            # Extract 1 minute of data
            start_idx = minute_idx * MINUTE_WINDOW
            end_idx = start_idx + MINUTE_WINDOW
            minute_data = sensor_data[start_idx:end_idx]

            # Split into 1-second windows (60 windows per minute)
            n_windows = int(np.floor(len(minute_data) / WINDOW))
            minute_data_windowed = minute_data[:n_windows * WINDOW].reshape((n_windows, WINDOW, 1))

            # Predict on all windows in this minute
            predictions = model.predict(minute_data_windowed, verbose=0)
            predictions_binary = (predictions > UNBALANCE_THRESHOLD).astype(int).flatten()

            # Check if unbalance detected (majority voting)
            unbalance_detections = np.sum(predictions_binary)
            detection_ratio = unbalance_detections / len(predictions_binary)

            # Calculate timestamp
            current_time = start_time + timedelta(seconds=minute_idx * 60)
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")

            # If significant unbalance detected (>50% of windows), save figure
            if detection_ratio > 0.5:
                total_detections += 1

                print(f"\n  ⚠️  UNBALANCE DETECTED at minute {minute_idx + 1}")
                print(f"      Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"      Row index: {start_idx:,} - {end_idx:,}")
                print(f"      Detection ratio: {detection_ratio*100:.1f}%")
                print(f"      Mean prediction: {np.mean(predictions):.4f}")

                # Generate and save figure
                fig = plt.figure(figsize=(15, 10))

                # Plot 1: Full minute timeseries
                ax1 = plt.subplot(3, 1, 1)
                time_axis = np.arange(len(minute_data)) / SAMPLES_PER_SECOND
                ax1.plot(time_axis, minute_data, lw=0.5)
                ax1.set_title(f"UNBALANCE DETECTED - Dataset {dataset_label} ({dataset_name})\n"
                             f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                             f"Rows: {start_idx:,}-{end_idx:,}", fontsize=14, fontweight='bold')
                ax1.set_xlabel("Time (seconds)")
                ax1.set_ylabel("Vibration Amplitude")
                ax1.grid(True, alpha=0.3)

                # Plot 2: Predictions over time
                ax2 = plt.subplot(3, 1, 2)
                window_times = np.arange(len(predictions))
                ax2.plot(window_times, predictions, marker='o', markersize=3, linewidth=1)
                ax2.axhline(y=UNBALANCE_THRESHOLD, color='r', linestyle='--',
                           label=f'Threshold ({UNBALANCE_THRESHOLD})')
                ax2.fill_between(window_times, 0, 1, where=(predictions.flatten() > UNBALANCE_THRESHOLD),
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
                sample_data = minute_data[sample_start:sample_end]
                sample_time = np.arange(len(sample_data)) / SAMPLES_PER_SECOND
                ax3.plot(sample_time, sample_data, lw=0.8)
                ax3.set_title(f"Highest Prediction Window (Prediction: {predictions[max_pred_idx][0]:.4f})")
                ax3.set_xlabel("Time (seconds)")
                ax3.set_ylabel("Vibration Amplitude")
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save figure
                filename = f"unbalance_detection_{dataset_label}_{timestamp_str}_row{start_idx}.png"
                output_path = os.path.join(output_dir, filename)
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"      Figure saved: {output_path}")

            # Simulate real-time processing speed if requested
            if speed > 0:
                minute_process_time = time.time() - minute_process_start
                target_time = 60.0 / speed  # 60 seconds for 1x speed
                sleep_time = target_time - minute_process_time
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
        """)

    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'All', 'ALL', '0E', '1E', '2E', '3E', '4E'],
                       help='Dataset to evaluate: all, 0E (no unbalance), 1E, 2E, 3E, or 4E (max unbalance). Default: all')

    parser.add_argument('--speed', type=float, default=0,
                       help='Processing speed multiplier. 0=maximum speed (no delays), 1.0=real-time (1 min data = 60s processing), 2.0=2x speed, etc. Default: 0 (max speed)')

    parser.add_argument('--model-path', type=str,
                       default='../../models/reference/cnn_3_layers.h5',
                       help='Path to trained model file. Default: ../../models/reference/cnn_3_layers.h5')

    parser.add_argument('--data-url', type=str,
                       default='../../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip',
                       help='Path or URL to dataset ZIP file')

    parser.add_argument('--output-dir', type=str, default='../../figures/detections',
                       help='Directory to save detection figures. Default: ../../figures/detections')

    args = parser.parse_args()

    print("=" * 80)
    print("Machine Learning Based Unbalance Detection - Approach 1: CNN")
    print("Real-time Anomaly Detection with 3-Layer CNN")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Speed: {'Maximum' if args.speed == 0 else f'{args.speed}x real-time'}")
    print(f"  Model Path: {args.model_path}")
    print(f"  Output Directory: {args.output_dir}")
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
    process_timeseries_data(model, data, output_dir=args.output_dir,
                           datasets=args.dataset, speed=args.speed)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nModel: {N_CONV_LAYERS}-layer CNN")
    if len(X_val) > 0:
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Detection Threshold: {UNBALANCE_THRESHOLD}")
    print(f"Window Size: 1 minute ({MINUTE_WINDOW:,} samples)")
    print(f"\nCheck the {args.output_dir}/ directory for detected anomalies.")


if __name__ == "__main__":
    main()
