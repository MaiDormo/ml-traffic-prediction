import csv
import datetime
import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.torch import DeepAREstimator
from scipy.fft import fft, fftfreq

# CONFIGURATION
INPUT_CSV_FILE = "exported"
USE_PROPHET_INPUT = True  # Set to True to use Prophet's preprocessed data
DT = 2.0
AUGMENTER_MULTIPLIER = 5
RANDOM_NOISE_N_PACKETS = 0.5


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    freq: str
    detected_period: float
    bins_per_cycle: float
    prediction_length: int
    context_length: int
    total_samples: int


def load_prophet_dataset() -> Tuple[pd.DataFrame, float, float]:
    print("\n" + "=" * 60)
    print("USING PROPHET PREPROCESSED DATA")
    print("=" * 60)

    prophet_input_file = INPUT_CSV_FILE + "_prophet_input.csv"
    if not os.path.exists(prophet_input_file):
        raise FileNotFoundError(
            f"{prophet_input_file} not found. Run: python3 preprocess.py first"
        )

    print(f"\nLoading: {prophet_input_file}")
    df = pd.read_csv(prophet_input_file)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.rename(columns={'ds': 'timestamp', 'y': 'target'})

    print(f"✓ Loaded {len(df)} samples")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Packet range: {df['target'].min():.2f} to {df['target'].max():.2f}")

    time_diff = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    if time_diff == DT:
        print("\n✓ Timestamps are uniform (2s intervals)")
        print("  Data is directly compatible with GluonTS")
    else:
        print("\n⚠ Reindexing with uniform 2s spacing (GluonTS requirement)")
        new_index = pd.date_range(
            start=pd.Timestamp('2025-11-17 19:00:00'),
            periods=len(df),
            freq=f'{int(DT)}s'
        )
        df['timestamp'] = new_index
        print(f"  New time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    L_augmented = len(df)
    estimated_cycles = 25
    bins_per_cycle = L_augmented / estimated_cycles
    detected_period = bins_per_cycle * DT

    print(f"\nEstimated from Prophet data:")
    print(f"  Period: ~{detected_period:.1f}s")
    print(f"  Bins per cycle: {bins_per_cycle:.1f}")
    print(f"  Total cycles: {estimated_cycles:.1f}")

    return df, detected_period, bins_per_cycle


def aggregate_packets(timestamps: List[datetime.datetime], dt: float) -> List[int]:
    if not timestamps:
        return []

    bins: List[int] = []
    current_bin = 0
    bin_start = timestamps[0]

    for timestamp in timestamps:
        if (timestamp - bin_start).total_seconds() > dt:
            bins.append(current_bin)
            current_bin = 1
            bin_start = timestamp
        else:
            current_bin += 1

    bins.append(current_bin)
    return bins


def detect_period(packet_counts: List[float], dt: float) -> float:
    N = len(packet_counts)
    if not N:
        return dt

    data_normalized = np.array(packet_counts) - np.mean(packet_counts)
    yf = fft(data_normalized)
    xf = fftfreq(N, dt)

    positive_freq_idx = xf > 0
    frequencies = xf[positive_freq_idx]
    power = np.abs(yf[positive_freq_idx])

    if len(power) == 0:
        return (N * dt) / 2

    dominant_freq = frequencies[np.argmax(power)]
    detected_period = 1.0 / dominant_freq if dominant_freq > 0 else (N * dt) / 2

    top_3_idx = np.argsort(power)[-3:][::-1]
    print(f"\nTop 3 detected frequencies:")
    for i, idx in enumerate(top_3_idx, 1):
        freq = frequencies[idx]
        period = 1.0 / freq if freq > 0 else float('inf')
        print(f"  {i}. Frequency: {freq:.4f} Hz, Period: {period:.1f}s, Power: {power[idx]:.1f}")

    return detected_period


def load_raw_dataset() -> Tuple[pd.DataFrame, float, float]:
    raw_input_file_path = os.path.join('./', INPUT_CSV_FILE + ".csv")
    now = datetime.datetime.now()
    timestamps: List[datetime.datetime] = []

    print(f"Reading input file: {raw_input_file_path}")
    with open(raw_input_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            try:
                timestamps.append(now + datetime.timedelta(seconds=float(row[1])))
            except (ValueError, IndexError):
                continue
    print(f"Loaded {len(timestamps)} packets")

    if timestamps:
        max_time = max(timestamps)
        timestamps = [ts for ts in timestamps if (max_time - ts).total_seconds() >= 1.0]
        print(f"After removing last second: {len(timestamps)} packets")

    print(f"\nAggregating into {DT}s bins...")
    packet_counts = aggregate_packets(timestamps, DT)
    L = len(packet_counts)
    print(f"Created {L} time bins")

    print("\n" + "=" * 60)
    print("AUTO-DETECTING PERIOD")
    print("=" * 60)

    detected_period = detect_period(packet_counts, DT)
    if detected_period <= 0:
        detected_period = (L * DT) / 2
        print(f"⚠ Using fallback: {detected_period:.1f}s")

    bins_per_cycle = detected_period / DT
    cycles = L / bins_per_cycle

    print(f"✓ Period: {detected_period:.1f}s")
    print(f"  Bins per cycle: {bins_per_cycle:.1f}")
    print(f"  Cycles in data: {cycles:.2f}")
    print("=" * 60)

    print(f"\nAugmenting data (repeating {AUGMENTER_MULTIPLIER}x)...")
    augmented_counts: List[float] = []
    for _ in range(AUGMENTER_MULTIPLIER):
        augmented_counts.extend(packet_counts)
    print(f"Total data points after augmentation: {len(augmented_counts)}")

    print(f"Adding minimal noise (±{RANDOM_NOISE_N_PACKETS} packets)...")
    for idx in range(len(augmented_counts)):
        noise = (np.random.random() - 0.5) * 2 * RANDOM_NOISE_N_PACKETS
        augmented_counts[idx] = max(0, augmented_counts[idx] + noise)

    start_time = timestamps[0]
    time_index = pd.date_range(
        start=start_time,
        periods=len(augmented_counts),
        freq=pd.Timedelta(seconds=int(DT))
    )

    df = pd.DataFrame({'timestamp': time_index, 'target': augmented_counts})
    print(f"\nDataset statistics:")
    print(f"  Shape: {df.shape}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Target range: {df['target'].min():.1f} - {df['target'].max():.1f}")
    print(f"  Mean: {df['target'].mean():.1f}, Std: {df['target'].std():.1f}")

    return df, detected_period, bins_per_cycle


def ensure_uniform_timestamps(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    print(f"\nEnsuring uniform timestamps for GluonTS...")
    time_diff = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    start_time = df['timestamp'].iloc[0]
    n_samples = len(df)

    if time_diff >= 3600:
        freq_str = 'h'
        print("  Detected frequency: hourly")
        index = pd.date_range(start=start_time, periods=n_samples, freq=freq_str)
    elif time_diff >= 60:
        freq_str = f'{int(time_diff // 60)}min'
        print(f"  Detected frequency: {freq_str}")
        index = pd.date_range(start=start_time, periods=n_samples, freq=freq_str)
    else:
        freq_str = f'{int(time_diff)}s'
        print(f"  Detected frequency: {freq_str}")
        index = pd.date_range(start=start_time, periods=n_samples, freq=pd.Timedelta(seconds=int(time_diff)))

    df = df.copy()
    df['timestamp'] = index
    print(f"  ✓ Recreated {n_samples} uniform timestamps")
    return df, freq_str


def compute_adaptive_parameters(total_samples: int, bins_per_cycle: float) -> Tuple[int, int]:
    prediction_length = min(100, total_samples // 4)
    context_length = max(int(bins_per_cycle * 3), prediction_length * 2)
    context_length = min(context_length, total_samples - prediction_length - 10)

    if context_length + prediction_length >= total_samples:
        raise ValueError(
            "Not enough data for training. Increase capture duration or augmentation."
        )

    print(f"\nAdaptive parameters (matching Prophet):")
    print(f"  Prediction length: {prediction_length} bins ({prediction_length / bins_per_cycle:.1f} cycles)")
    print(f"  Context length: {context_length} bins ({context_length / bins_per_cycle:.1f} cycles)")
    print(f"  Total required: {context_length + prediction_length} bins")
    print(f"  Available: {total_samples} bins")

    return prediction_length, context_length


def build_gluonts_datasets(df: pd.DataFrame, freq: str, prediction_length: int) -> Tuple[PandasDataset, PandasDataset, pd.DataFrame, pd.DataFrame]:
    test_size = prediction_length
    train_df = df.iloc[:-test_size].copy()
    test_df = df.copy()

    train_dataset = PandasDataset(
        dataframes=train_df.set_index('timestamp'),
        target='target',
        freq=freq
    )
    test_dataset = PandasDataset(
        dataframes=test_df.set_index('timestamp'),
        target='target',
        freq=freq
    )

    print("✓ Datasets created")
    return train_dataset, test_dataset, train_df, test_df



def train_predictor(train_dataset: PandasDataset, freq: str, prediction_length: int, context_length: int):
    print("\n" + "=" * 60)
    print("TRAINING GLUONTS DEEPAR MODEL")
    print("=" * 60)

    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        num_layers=2,
        hidden_size=40,
        dropout_rate=0.1,
        trainer_kwargs={
            'max_epochs': 10,
            'accelerator': 'cpu',
            'devices': 1,
            'enable_progress_bar': True,
            'enable_model_summary': True
        }
    )

    print("\nTraining model...")
    predictor = estimator.train(train_dataset)
    print("✓ Model trained")
    return predictor


def generate_predictions(predictor, test_dataset):
    print("\nGenerating predictions...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_dataset,
        predictor=predictor,
        num_samples=100
    )
    forecasts = list(forecast_it)
    timeseries = list(ts_it)
    print("✓ Predictions generated")
    return forecasts, timeseries


def evaluate_metrics(timeseries, forecasts) -> Tuple[dict, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    evaluator = Evaluator()
    agg_metrics, item_metrics = evaluator(iter(timeseries), iter(forecasts))

    print("\nAggregate Metrics:")
    for key in ['RMSE', 'abs_error', 'MASE', 'MAPE', 'sMAPE', 'mean_wQuantileLoss']:
        if key in agg_metrics:
            print(f"  {key}: {agg_metrics[key]:.4f}")

    metrics_file = INPUT_CSV_FILE + "_gluonts_metrics.csv"
    pd.DataFrame([agg_metrics]).to_csv(metrics_file, index=False)
    print(f"\n✓ Saved metrics: {metrics_file}")
    return agg_metrics, item_metrics


def _to_datetime_index(ts):
    return ts.index.to_timestamp() if hasattr(ts.index, 'to_timestamp') else ts.index


def _build_forecast_index(ts_index, prediction_length, freq):
    test_start_idx = len(ts_index) - prediction_length
    forecast_index = pd.date_range(
        start=ts_index[test_start_idx],
        periods=prediction_length,
        freq=freq
    )
    return forecast_index, test_start_idx


def plot_full_forecast(ts, forecast, freq, detected_period, prediction_length):
    ts_index = _to_datetime_index(ts)
    forecast_index, test_start_idx = _build_forecast_index(ts_index, prediction_length, freq)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(ts_index, ts.values, 'k.', label='Observations', markersize=3, alpha=0.5)
    ax.plot(forecast_index, forecast.mean, color='#0072B2', linewidth=2, label='Forecast')
    ax.fill_between(
        forecast_index,
        forecast.quantile(0.025),
        forecast.quantile(0.975),
        alpha=0.2,
        color='#0072B2',
        label='95% Uncertainty Interval'
    )
    ax.fill_between(
        forecast_index,
        forecast.quantile(0.1),
        forecast.quantile(0.9),
        alpha=0.3,
        color='#0072B2'
    )

    ax.set_xlabel('Timestamp', fontsize=11)
    ax.set_ylabel('Packets per Bin', fontsize=11)
    ax.set_title(f'Network Traffic Forecast - GluonTS DeepAR (Period: {detected_period:.1f}s)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('gluonts_forecast.png', dpi=200)
    print("✓ Saved: gluonts_forecast.png")
    plt.close()

    return forecast_index, test_start_idx, ts_index


def plot_detail_forecast(ts, ts_index, forecast, forecast_index, context_length, prediction_length, detected_period):
    history_length = min(len(ts), context_length + prediction_length)
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(ts_index[-history_length:], ts.values[-history_length:],
            label='Historical', linewidth=2, color='#2C3E50')
    ax.plot(ts_index[-prediction_length:], ts.values[-prediction_length:],
            label='Actual', linewidth=2.5, color='#E74C3C', marker='o', markersize=4)
    ax.plot(forecast_index, forecast.mean,
            label='DeepAR Forecast', linewidth=2, color='#3498DB', linestyle='--')
    ax.fill_between(
        forecast_index,
        forecast.quantile(0.1),
        forecast.quantile(0.9),
        alpha=0.3,
        color='#AED6F1',
        label='80% Confidence'
    )

    ax.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
    ax.set_ylabel('Packets per Bin', fontsize=12, fontweight='bold')
    ax.set_title(f'GluonTS DeepAR Forecast Detail (Period: {detected_period:.1f}s)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('gluonts_forecast_detail.png', dpi=200)
    print("✓ Saved: gluonts_forecast_detail.png")
    plt.close()


def save_forecast_csv(forecast_index, ts_values, test_start_idx, forecast):
    forecast_df = pd.DataFrame({
        'timestamp': forecast_index,
        'actual': ts_values[test_start_idx:].ravel(),
        'predicted': np.array(forecast.mean).ravel(),
        'lower_80': np.array(forecast.quantile(0.1)).ravel(),
        'upper_80': np.array(forecast.quantile(0.9)).ravel(),
        'lower_95': np.array(forecast.quantile(0.025)).ravel(),
        'upper_95': np.array(forecast.quantile(0.975)).ravel()
    })

    forecast_file = INPUT_CSV_FILE + "_gluonts_forecast.csv"
    forecast_df.to_csv(forecast_file, index=False)
    print(f"✓ Saved forecast: {forecast_file}")


def summarize_run(bundle: DatasetBundle, metrics: dict):
    print("\n" + "=" * 60)
    print("GLUONTS ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Period: {bundle.detected_period:.1f}s")
    print(f"Bins per cycle: {bundle.bins_per_cycle:.1f}")
    print(f"Data source: {'Prophet preprocessed input' if USE_PROPHET_INPUT else 'Raw capture'}")
    print(f"Total samples: {bundle.total_samples}")
    print(f"Prediction length: {bundle.prediction_length}")
    print(f"Context length: {bundle.context_length}")
    if 'RMSE' in metrics:
        print(f"RMSE: {metrics['RMSE']:.4f}")
    if 'abs_error' in metrics:
        print(f"Mean Absolute Error: {metrics['abs_error']:.4f}")
    print("=" * 60)


def main() -> None:
    if USE_PROPHET_INPUT:
        df, detected_period, bins_per_cycle = load_prophet_dataset()
    else:
        df, detected_period, bins_per_cycle = load_raw_dataset()

    df, freq = ensure_uniform_timestamps(df)
    prediction_length, context_length = compute_adaptive_parameters(len(df), bins_per_cycle)

    train_dataset, test_dataset, train_df, test_df = build_gluonts_datasets(df, freq, prediction_length)
    predictor = train_predictor(train_dataset, freq, prediction_length, context_length)
    forecasts, timeseries = generate_predictions(predictor, test_dataset)
    metrics, _ = evaluate_metrics(timeseries, forecasts)

    ts = timeseries[0]
    forecast = forecasts[0]
    forecast_index, test_start_idx, ts_index = plot_full_forecast(
        ts, forecast, freq, detected_period, prediction_length
    )
    plot_detail_forecast(ts, ts_index, forecast, forecast_index, context_length, prediction_length, detected_period)
    save_forecast_csv(forecast_index, ts.values, test_start_idx, forecast)

    bundle = DatasetBundle(
        df=df,
        train_df=train_df,
        test_df=test_df,
        freq=freq,
        detected_period=detected_period,
        bins_per_cycle=bins_per_cycle,
        prediction_length=prediction_length,
        context_length=context_length,
        total_samples=len(df)
    )
    summarize_run(bundle, metrics)


if __name__ == "__main__":
    main()
