#!/usr/bin/env python3
"""
Train Minimal Features Random Forest Classifier for Unbalance Detection

This script trains the RFC model using 7 simple statistical features:
- 1x RPM mean
- 3x Vibration sensor standard deviations
- 3x Vibration sensor kurtosis values
"""

import numpy as np
import pandas as pd
import zipfile
import scipy.stats
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
from joblib import dump
from pathlib import Path

# Configuration
DATA_ZIP = '../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'
MODEL_OUTPUT = 'models/minimal_rfc.joblib'
SKIP = 50000  # Skip warm-up phase
WINDOW = 4096  # Samples per window (1 second)

def reshape_features(data, column, f):
    """Extract features from a single column using windowing."""
    features = data[column].values
    features = features[:(len(features)//WINDOW)*WINDOW]
    features = np.reshape(features, (-1, WINDOW), order='C')
    return f(features, axis=1)

def calc_features(zip_file, *files, columns=range(1, 4)):
    """
    Calculate minimal set of features from CSV files.

    Args:
        zip_file: Open zipfile object
        files: CSV filenames to process
        columns: Which sensors to use (1, 2, 3 for Vibration_1, 2, 3)

    Returns:
        Array of shape (n_windows, n_features) where n_features = 1 + 2*len(columns)
    """
    features = []
    for file in files:
        with zip_file.open(file, 'r') as f:
            data = pd.read_csv(f)
            data = data.iloc[SKIP:, :]

            # Feature extraction:
            # 1. Mean of RPM
            # 2-4. Standard deviation of each vibration sensor
            # 5-7. Kurtosis of each vibration sensor
            data_features = np.stack(
                [reshape_features(data, 'Measured_RPM', np.mean)]
                + [reshape_features(data, f'Vibration_{i}', np.std)
                   for i in columns]
                + [reshape_features(data, f'Vibration_{i}', scipy.stats.kurtosis)
                   for i in columns],
                axis=1)
            features.append(data_features)

    return np.concatenate(features, axis=0)

def get_features(url, group='D', columns=range(1, 4)):
    """
    Load and extract features from all datasets in a group.

    Args:
        url: Path to data ZIP file
        group: 'D' for training, 'E' for evaluation
        columns: Which sensors to use

    Returns:
        X: Feature array
        y: Label array (0=no unbalance, 1=unbalance)
    """
    with zipfile.ZipFile(url, 'r') as f:
        good = calc_features(f, f'0{group}.csv', columns=columns)
        bad1 = calc_features(f, f'1{group}.csv', columns=columns)
        bad2 = calc_features(f, f'2{group}.csv', columns=columns)
        bad3 = calc_features(f, f'3{group}.csv', columns=columns)
        bad4 = calc_features(f, f'4{group}.csv', columns=columns)

    X = np.concatenate([good, bad1, bad2, bad3, bad4], axis=0)
    length_good = good.shape[0]
    length_bad = bad1.shape[0] + bad2.shape[0] + bad3.shape[0] + bad4.shape[0]
    y = np.concatenate([np.zeros(length_good), np.ones(length_bad)], axis=0)

    return X, y

def get_features_single(url, file, columns=range(1, 4), label=1):
    """Load features from a single dataset file."""
    with zipfile.ZipFile(url, 'r') as f:
        X = calc_features(f, file, columns=columns)
    y = label * np.ones(X.shape[0])
    return X, y

def main():
    print("=" * 80)
    print("Training Minimal Features Random Forest Classifier")
    print("=" * 80)

    # Check if data file exists
    data_path = Path(DATA_ZIP)
    if not data_path.exists():
        print(f"Error: Data file not found at {DATA_ZIP}")
        print("Please ensure the data ZIP file is in the correct location.")
        return

    # Extract training features
    print(f"\nExtracting minimal features from training data...")
    print(f"  - Using all 3 vibration sensors")
    print(f"  - Window size: {WINDOW} samples (1 second)")
    print(f"  - Features per window: 7 (1 RPM mean + 3 sensor std + 3 sensor kurtosis)")

    X_train, y_train = get_features(str(data_path), group='D')

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"  - Total windows: {X_train.shape[0]:,}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Normal (0): {np.sum(y_train == 0):,} windows")
    print(f"  - Unbalanced (1): {np.sum(y_train == 1):,} windows")

    # Train model with GridSearch
    print(f"\nTraining Random Forest Classifier with GridSearch...")
    print(f"  - Searching parameters:")
    print(f"    - n_estimators: [30, 100, 300]")
    print(f"    - max_depth: [2, 5, 10, 20]")
    print(f"    - min_samples_leaf: [1, 10, 100]")
    print(f"  - Scoring: balanced_accuracy")
    print(f"\nThis may take several minutes...")

    model = sklearn.model_selection.GridSearchCV(
        sklearn.ensemble.RandomForestClassifier(),
        {'n_estimators': [30, 100, 300],
         'max_depth': [2, 5, 10, 20],
         'min_samples_leaf': [1, 10, 100]},
        scoring='balanced_accuracy',
        verbose=1,
        n_jobs=-1)

    model.fit(X_train, y_train)

    print(f"\nTraining complete!")
    print(f"  - Best cross-validation score: {model.best_score_:.4f}")
    print(f"  - Best parameters: {model.best_params_}")

    # Evaluate on evaluation datasets
    print(f"\nEvaluating on evaluation datasets (0E-4E)...")
    accuracies = []
    for eval_file, label_name in [('0E.csv', '0E (No Unbalance)'),
                                    ('1E.csv', '1E (Level 1)'),
                                    ('2E.csv', '2E (Level 2)'),
                                    ('3E.csv', '3E (Level 3)'),
                                    ('4E.csv', '4E (Level 4)')]:
        if eval_file == '0E.csv':
            X, y = get_features_single(str(data_path), eval_file, label=0)
        else:
            X, y = get_features_single(str(data_path), eval_file, label=1)

        score = sklearn.metrics.accuracy_score(y, model.predict(X))
        accuracies.append(score)
        print(f"  - {label_name}: {score:.4f} ({score*100:.1f}%)")

    mean_accuracy = np.mean(accuracies)
    print(f"\nMean accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.1f}%)")

    # Save model
    output_path = Path(MODEL_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, str(output_path))
    print(f"\nModel saved to: {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024:.1f} KB")

    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
