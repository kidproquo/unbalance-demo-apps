"""
Machine Learning Based Unbalance Detection of a Rotating Shaft Using Vibration Data
Notebook 2 of 5 - Approach 1: Convolutional Neural Network on Raw Sensor Data

Oliver Mey, Willi Neudeck, André Schneider and Olaf Enge-Rosenblatt
Fraunhofer IIS/EAS, Fraunhofer Institute for Integrated Circuits,
Division Engineering of Adaptive Systems, Dresden, Germany

This Python script is converted from the Jupyter Notebook that is part of a paper
submission to the 25th IEEE International Conference on Emerging Technologies and
Factory Automation, ETFA 2020.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sc
import zipfile
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import os
from sklearn.model_selection import train_test_split
from scipy.stats import mode

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dense, Dropout
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2


# Configuration
LABELS = {'no_unbalance': 0, 'unbalance': 1}
SENSOR = 'Vibration_1'
SAMPLES_PER_SECOND = 4096
SECONDS_PER_ANALYSIS = 1.0
WINDOW = int(SAMPLES_PER_SECOND * SECONDS_PER_ANALYSIS)
SKIP = 50000  # Skip warm-up phase


def load_data(url):
    """
    Load measurement data from ZIP file.

    Args:
        url: Path to the ZIP file containing the dataset

    Returns:
        Dictionary containing all datasets (0D-4D, 0E-4E)
    """
    print("Loading measurement data...")
    data = {}

    with zipfile.ZipFile(url, 'r') as f:
        for label in ['0D', '0E', '1D', '1E', '2D', '2E', '3D', '3E', '4D', '4E']:
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
    Prepare training and validation datasets.

    Args:
        data: Dictionary of dataframes with all datasets

    Returns:
        Tuple of (X, y, X_val, y_val, X_val_separated, y_val_separated)
    """
    print("\nPreparing datasets...")

    # Training data (D datasets)
    X0, y0 = get_features(data['0D'][SENSOR], "no_unbalance")
    X1, y1 = get_features(data['1D'][SENSOR], "unbalance")
    X2, y2 = get_features(data['2D'][SENSOR], "unbalance")
    X3, y3 = get_features(data['3D'][SENSOR], "unbalance")
    X4, y4 = get_features(data['4D'][SENSOR], "unbalance")
    X = np.concatenate([X0, X1, X2, X3, X4])
    y = np.concatenate([y0, y1, y2, y3, y4])

    # Validation data (E datasets)
    X0_val, y0_val = get_features(data['0E'][SENSOR], "no_unbalance")
    X1_val, y1_val = get_features(data['1E'][SENSOR], "unbalance")
    X2_val, y2_val = get_features(data['2E'][SENSOR], "unbalance")
    X3_val, y3_val = get_features(data['3E'][SENSOR], "unbalance")
    X4_val, y4_val = get_features(data['4E'][SENSOR], "unbalance")
    X_val = np.concatenate([X0_val, X1_val, X2_val, X3_val, X4_val])
    y_val = np.concatenate([y0_val, y1_val, y2_val, y3_val, y4_val])

    # Separated validation data for per-dataset evaluation
    X_val_separated = [
        X_val[:len(y0_val), :],
        X_val[len(y0_val):len(y0_val)+len(y1_val), :],
        X_val[len(y0_val)+len(y1_val):len(y0_val)+len(y1_val)+len(y2_val), :],
        X_val[len(y0_val)+len(y1_val)+len(y2_val):len(y0_val)+len(y1_val)+len(y2_val)+len(y3_val), :],
        X_val[len(y0_val)+len(y1_val)+len(y2_val)+len(y3_val):, :]
    ]
    y_val_separated = [y0_val, y1_val, y2_val, y3_val, y4_val]

    # Also keep development sets for pairwise training
    X_dev = [X0, X1, X2, X3, X4]
    y_dev = [y0, y1, y2, y3, y4]

    print(f"  Training data shape: {X.shape}, labels: {y.shape}")
    print(f"  Validation data shape: {X_val.shape}, labels: {y_val.shape}")

    return X, y, X_val, y_val, X_val_separated, y_val_separated, X_dev, y_dev


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


def train_models(X, y, model_path, use_reference_models=True):
    """
    Train CNN models with different numbers of layers.

    Args:
        X: Training features
        y: Training labels
        model_path: Path to save/load models
        use_reference_models: If True, use pre-trained models instead of training
    """
    if use_reference_models:
        print("\nUsing reference models (skipping training)")
        return

    print("\nTraining models with different layer configurations...")

    # Calculate class weights
    weight_for_0 = len(y) / (2 * len(y[y == 0]))
    weight_for_1 = len(y) / (2 * len(y[y == 1]))
    class_weight = {0: weight_for_0, 1: weight_for_1}

    # Training parameters
    train_test_ratio = 0.9
    learning_rate = 0.0001
    n_epochs = 100

    for n_conv_layers in range(1, 5):
        print(f"\n--- Training model with {n_conv_layers} convolutional layer(s) ---")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-train_test_ratio, random_state=0)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

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
        classifier.fit(
            X_train, y_train,
            epochs=n_epochs,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint],
            class_weight=class_weight
        )


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


def plot_results(accuracies, accuracies_all, rpm_borders, errors_per_rpm_range,
                output_dir='../../figures'):
    """
    Plot evaluation results.

    Args:
        accuracies: Per-dataset accuracies
        accuracies_all: Overall accuracies
        rpm_borders: RPM range boundaries
        errors_per_rpm_range: Accuracies per RPM range
        output_dir: Directory to save figures
    """
    print("\nPlotting results...")
    os.makedirs(output_dir, exist_ok=True)

    unbalances = np.array([0, 4.59e-5, 6.07e-5, 7.55e-5, 1.521e-4])

    # Plot 1: Unbalance classification
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(111, title="Unbalance Classification Trained with all Unbalances")
    for i in range(4):
        ax1.plot(1e6 * unbalances, accuracies[i, :],
                label=f"{i+1} conv. layer(s), mean: {100.0*accuracies_all[i]:.1f}%",
                marker="+", ls="--")
    plt.ylabel("Accuracy on Evaluation Dataset")
    plt.xlabel("Unbalance Factor [mm g]")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.ylim([0.45, 1.05])
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "Fig.5a_cnn_unbalance.png")
    fig.savefig(output_path, dpi=200)
    print(f"  Saved to {output_path}")
    plt.close()

    # Plot 2: Rotation speed dependency
    fig = plt.figure(figsize=(12, 8))
    ax2 = plt.subplot(111, title="Rotation Speed Dependent Evaluation")
    for i in range(4):
        ax2.plot(np.array(rpm_borders[:-1]) + 25, errors_per_rpm_range[i],
                marker="+", ls="--", label=f"{i+1} conv. layer(s)")
    plt.ylabel("Accuracy on Evaluation Dataset")
    plt.xlabel("Rotation Speed [rpm]")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.ylim([0.45, 1.05])
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "Fig.5b_cnn_rpm.png")
    fig.savefig(output_path, dpi=200)
    print(f"  Saved to {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Machine Learning Based Unbalance Detection - Approach 1: CNN")
    print("=" * 80)

    # Configuration
    # Option a) local file contains a small subset of the entire dataset
    url = '../../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'

    # Option b) the entire dataset can be directly downloaded via public Fraunhofer Fortadis dataspace
    # url = 'https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/1/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'

    # Option c) selected pre-trained models can be found in the directory model/reference
    use_reference_models = True
    model_path = '../../models/reference'

    # Option d) all models will be trained again
    # use_reference_models = False
    # model_path = '../../models'

    # Load data
    data = load_data(url)

    # Skip warm-up phase
    data = skip_warmup(data)

    # Prepare datasets
    X, y, X_val, y_val, X_val_separated, y_val_separated, X_dev, y_dev = prepare_datasets(data)

    # Train models (or use reference models)
    train_models(X, y, model_path, use_reference_models)

    # Evaluate models
    accuracies, accuracies_all = evaluate_models(
        X_val_separated, y_val_separated, X_val, y_val, model_path)

    print("\n" + "=" * 80)
    print("Overall Model Accuracies:")
    for i, acc in enumerate(accuracies_all):
        print(f"  {i+1} conv. layer(s): {100.0*acc:.2f}%")
    print("=" * 80)

    # Evaluate rotation speed dependency
    rpm_borders, errors_per_rpm_range = evaluate_rotation_speed_dependency(
        X_val, y_val, model_path)

    # Plot results
    plot_results(accuracies, accuracies_all, rpm_borders, errors_per_rpm_range)

    print("\n" + "=" * 80)
    print("Approach 1 CNN analysis complete!")
    print("=" * 80)
    print("\nNote: This script covers the main evaluation.")
    print("For pairwise training experiments, see the original notebook.")


if __name__ == "__main__":
    main()
