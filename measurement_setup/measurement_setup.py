"""
Machine Learning Based Unbalance Detection of a Rotating Shaft Using Vibration Data
Notebook 1 of 4 - Measurement Setup

Oliver Mey, Willi Neudeck, André Schneider and Olaf Enge-Rosenblatt
Fraunhofer IIS/EAS, Fraunhofer Institute for Integrated Circuits,
Division Engineering of Adaptive Systems, Dresden, Germany

This Python script is converted from the Jupyter Notebook that is part of a paper
submission to the 25th IEEE International Conference on Emerging Technologies and
Factory Automation, ETFA 2020.
"""

import pandas as pd
import numpy as np
import scipy as sc
import zipfile
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import os


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


def skip_warmup(data, skip=50000):
    """
    Skip warm-up phase of each measurement.

    The first about 10 seconds are noisy due to the warm-up phase of the
    measuring device, so the first 50000 samples are skipped.

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


def inspect_dataset(data, label):
    """
    Inspect a specific dataset.

    Args:
        data: Dictionary of dataframes
        label: Label of the dataset to inspect (e.g., '0E')
    """
    print(f"\n=== Dataset {label} ===")
    print(data[label])
    print(f"\n=== Dataset {label} Statistics ===")
    print(data[label].describe())


def preprocess_fft(data, label, window=4096):
    """
    Preprocess data using FFT.

    Args:
        data: Dictionary of dataframes
        label: Label of the dataset to process
        window: Window size for FFT (default: 4096)

    Returns:
        Tuple of (time_domain_data, fft_data)
    """
    print(f"\nPreprocessing {label} with FFT (window={window})...")

    n = int(np.floor(len(data[label]['Vibration_1']) / window))
    X = data[label]['Vibration_1'][:int(n) * window].values.reshape((n, window))

    X_fft = np.abs(np.fft.rfft(X, axis=1))[:, :int(window / 2)]
    X_fft[:, 0] = 0  # Remove DC component

    print(f"  Created {n} windows of size {window}")
    print(f"  FFT output shape: {X_fft.shape}")

    return X, X_fft


def plot_timeseries(data, output_dir='../../figures'):
    """
    Plot measurement datasets 0E and 4E timeseries.

    Args:
        data: Dictionary of dataframes
        output_dir: Directory to save figures
    """
    print("\nPlotting timeseries comparison...")

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2,
                           title="Dataset 0E - No Unbalance")
    ax1.plot(np.arange(len(data['0E']['Vibration_1'])) / 4096,
             data['0E']['Vibration_1'], lw=0.5)
    ax1.set_xlabel("Sample Index (1s Window)")
    ax1.set_ylabel("Vibration Sensor 1 Amplitude")
    ax1.text(-300, 0.125, "(a)", fontsize=12)

    ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2,
                           title="Dataset 4E - Largest Unbalance", sharey=ax1)
    ax2.plot(np.arange(len(data['4E']['Vibration_1'])) / 4096,
             data['4E']['Vibration_1'], lw=0.5)
    ax2.set_xlabel("Sample Index (1s Window)")
    ax2.text(-300, 0.125, "(b)", fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "Fig.4_ab.png")
    fig.savefig(output_path, dpi=200)
    print(f"  Saved to {output_path}")
    plt.close()


def plot_samples(data, output_dir='../../figures'):
    """
    Plot sample windows from datasets 0E and 4E at ~1000 RPM.

    Args:
        data: Dictionary of dataframes
        output_dir: Directory to save figures
    """
    print("\nPlotting sample windows...")

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4,
                           title="Sample from Dataset 0E @ ~1000 RPM - No Unbalance")
    ax1.plot(np.arange(4096),
             data['0E']['Vibration_1'].loc[2950000:2950000 + 4095], lw=0.5)
    ax1.text(0, 0.1, "(c)", fontsize=12)

    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=4,
                           title="Sample from Dataset 4E @ ~1000 RPM - Largest Unbalance",
                           sharex=ax1, sharey=ax1)
    ax2.plot(np.arange(4096),
             data['4E']['Vibration_1'].loc[2950000:2950000 + 4095], lw=0.5)
    ax2.text(0, 0.1, "(d)", fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "Fig.4_cd.png")
    fig.savefig(output_path, dpi=200)
    print(f"  Saved to {output_path}")
    plt.close()


def plot_fft(X_0E_fft, X_4E_fft, output_dir='../../figures'):
    """
    Plot FFT of datasets 0E and 4E.

    Args:
        X_0E_fft: FFT data for dataset 0E
        X_4E_fft: FFT data for dataset 4E
        output_dir: Directory to save figures
    """
    print("\nPlotting FFT comparison...")

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4,
                           title="FFT of Dataset 0E (Vibration1), Second Circle - No Unbalance")
    ax1.pcolorfast(np.arange(1, 2050), np.linspace(1057., 1926.2, len(X_0E_fft[831:1670, :]) + 1),
                   X_0E_fft[831:1670, :], cmap="Blues",
                   norm=LogNorm(vmin=0.005, vmax=5.0))
    ax1.set_ylabel("Rotation Speed [rpm]")

    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=4,
                           title="FFT of Dataset 4E (Vibration1), Second Circle - Largest Unbalance")
    ax2.pcolorfast(np.arange(1, 2050), np.linspace(1057., 1926.2, len(X_4E_fft[831:1670, :]) + 1),
                   X_4E_fft[831:1670, :], cmap="Blues",
                   norm=LogNorm(vmin=0.005, vmax=5.0))
    ax2.set_ylabel("Rotation Speed [rpm]")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "Fig.4_ef.png")
    fig.savefig(output_path, dpi=200)
    print(f"  Saved to {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Machine Learning Based Unbalance Detection - Measurement Setup")
    print("=" * 80)

    # Option a) local file contains a small subset of the entire dataset
    url = '../../data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'

    # Option b) the entire dataset can be directly downloaded via public Fraunhofer Fortadis dataspace
    # url = 'https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/1/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'

    # Load data
    data = load_data(url)

    # Skip warm-up phase
    data = skip_warmup(data, skip=50000)

    # Inspect datasets
    inspect_dataset(data, '0E')
    inspect_dataset(data, '4E')

    # Plot timeseries
    plot_timeseries(data)

    # Plot sample windows
    plot_samples(data)

    # Preprocess with FFT
    X_0E, X_0E_fft = preprocess_fft(data, '0E', window=4096)
    X_4E, X_4E_fft = preprocess_fft(data, '4E', window=4096)

    # Plot FFT
    plot_fft(X_0E_fft, X_4E_fft)

    print("\n" + "=" * 80)
    print("Measurement setup analysis complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("- Approach 1: Convolutional Neural Network on Raw Sensor Data")
    print("- Approach 2: Fully-Connected Neural Network on FFT-transformed Data")
    print("- Approach 3: Random Forest on Automatically Extracted Timeseries Features")
    print("- Approach 4: Hidden Markov Model")


if __name__ == "__main__":
    main()
