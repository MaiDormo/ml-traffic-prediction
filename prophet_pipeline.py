import csv
import datetime
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.metrics import mean_absolute_error, mean_squared_error

# PARAMETERS - NO HARDCODED PERIOD
SQUASHER_ENABLED = True
AUGMENTER_ENABLED = True
RANDOM_NOISE_INJECTER = True
USE_REAL_TIMESTAMPS = True  # Use real 2s intervals (compatible with GluonTS)

INPUT_CSV_FILE = "exported"

# Basic parameters only
DT = 2.0  # 2-second bins for aggregation
AUGMENTER_MULTIPLIER = 4  # Repeat pattern for more data
RANDOM_NOISE_N_PACKETS = 2  # Minimal noise


@dataclass
class TimeSeriesContext:
    period_seconds: float
    detection_method: str
    bins_per_cycle: float
    cycles: float
    custom_seasonality_period: Optional[float] = None


raw_input_file_path = os.path.join('./', INPUT_CSV_FILE + ".csv")
REFERENCE_TIME = datetime.datetime.now()


def to_ts_f(x: float) -> datetime.datetime:
    """Convert relative time (seconds) to absolute timestamp."""
    return REFERENCE_TIME + datetime.timedelta(seconds=float(x))


def read_packet_timestamps(file_path: str) -> List[datetime.datetime]:
    """Load packet timestamps from CSV file."""
    timestamps: List[datetime.datetime] = []
    print(f"Reading input file: {file_path}")
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                print(f'Column names: {", ".join(row)}')
                continue
            try:
                timestamps.append(to_ts_f(row[1]))
            except (ValueError, IndexError):
                continue
    print(f"Loaded {len(timestamps)} packets")
    return timestamps


def trim_last_second(timestamps: List[datetime.datetime]) -> List[datetime.datetime]:
    """Drop timestamps that fall within the last second to avoid partial bins."""
    if not timestamps:
        return []
    max_time = max(timestamps)
    trimmed = [ts for ts in timestamps if (max_time - ts).total_seconds() >= 1.0]
    print(f"After removing last second: {len(trimmed)} packets")
    return trimmed


def squash_timestamps_weighted(ts: List[datetime.datetime], W: List[int], dt: float) -> List[float]:
    """Aggregate packets into time bins."""
    L = len(ts)
    assert L == len(W)

    weighted: List[float] = []
    pts = datetime.datetime.fromtimestamp(0)
    tmp = 0.0

    for idx in range(L):
        cts = ts[idx]
        if (cts - pts).total_seconds() > dt:
            weighted.append(tmp)
            tmp = W[idx]
            pts = cts
        else:
            tmp += W[idx]

    weighted.append(tmp)
    return weighted


def detect_period_fft(data: List[float], dt: float) -> Optional[float]:
    """Detect dominant period using FFT."""
    N = len(data)
    if not N:
        return None

    data_normalized = np.array(data) - np.mean(data)
    yf = fft(data_normalized)
    xf = fftfreq(N, dt)

    positive_freq_idx = xf > 0
    frequencies = xf[positive_freq_idx]
    power = np.abs(yf[positive_freq_idx])

    if len(power) == 0:
        print("⚠ Warning: No periodic pattern detected")
        return None

    dominant_idx = np.argmax(power)
    dominant_freq = frequencies[dominant_idx]

    if dominant_freq <= 0:
        return None

    detected_period = 1.0 / dominant_freq

    top_3_idx = np.argsort(power)[-3:][::-1]
    print(f"\nTop 3 detected frequencies:")
    for i, idx in enumerate(top_3_idx, 1):
        freq = frequencies[idx]
        period = 1.0 / freq if freq > 0 else float('inf')
        print(f"  {i}. Frequency: {freq:.4f} Hz, Period: {period:.1f}s, Power: {power[idx]:.1f}")

    return detected_period


def detect_period_autocorr(data: List[float], dt: float) -> Optional[float]:
    """Detect period using autocorrelation (alternative method)."""
    if not data:
        return None

    data_normalized = np.array(data) - np.mean(data)
    autocorr = np.correlate(data_normalized, data_normalized, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    autocorr = autocorr / autocorr[0]
    min_lag = max(1, len(autocorr) // 10)
    peaks, _ = signal.find_peaks(autocorr[min_lag:])

    if len(peaks) == 0:
        print("⚠ Warning: No periodic pattern detected via autocorrelation")
        return None

    first_peak_lag = peaks[0] + min_lag
    detected_period = first_peak_lag * dt
    print(f"\nAutocorrelation detected period: {detected_period:.1f}s")
    return detected_period


def detect_period_metadata(npackets: List[float], dt: float) -> TimeSeriesContext:
    """Determine dominant period information for the aggregated series."""
    print("\n" + "=" * 60)
    print("DETECTING PERIOD AUTOMATICALLY")
    print("=" * 60)

    detected_period_fft = detect_period_fft(npackets, dt)
    detected_period_autocorr = detect_period_autocorr(npackets, dt)

    if detected_period_fft is not None and detected_period_fft > 0:
        period_seconds = detected_period_fft
        detection_method = "FFT"
    elif detected_period_autocorr is not None:
        period_seconds = detected_period_autocorr
        detection_method = "Autocorrelation"
    else:
        period_seconds = (len(npackets) * dt) / 2 if npackets else dt
        detection_method = "Fallback (estimated)"
        print("⚠ Using fallback estimate")

    bins_per_cycle = period_seconds / dt if dt else 1
    cycles = len(npackets) / bins_per_cycle if bins_per_cycle else 0
    print(f"\n✓ Detected period: {period_seconds:.1f}s (method: {detection_method})")
    print(f"  Bins per cycle: {bins_per_cycle:.1f}")

    return TimeSeriesContext(
        period_seconds=period_seconds,
        detection_method=detection_method,
        bins_per_cycle=bins_per_cycle,
        cycles=cycles,
    )


def build_time_axis(start_ts: datetime.datetime, length: int, period_seconds: float) -> Tuple[List[datetime.datetime], Optional[float]]:
    """Create evenly spaced timestamps aligned with the detected period."""
    timestamps: List[datetime.datetime] = []
    if USE_REAL_TIMESTAMPS:
        print(f"  Using REAL timestamps: {DT}s intervals")
        print("  ✓ Output compatible with both Prophet and GluonTS")
        for idx in range(length):
            timestamps.append(start_ts + datetime.timedelta(seconds=(idx * DT)))
        custom_seasonality_period = period_seconds / 86400
    else:
        time_multiplier = (24 * 3600) / period_seconds if period_seconds else 1
        print(f"  Time multiplier: {time_multiplier:.0f}x")
        print(f"  Mapping: {period_seconds:.1f}s (real) → 24 hours (Prophet)")
        for idx in range(length):
            scaled = idx * DT * time_multiplier
            timestamps.append(start_ts + datetime.timedelta(seconds=scaled))
        custom_seasonality_period = None
    return timestamps, custom_seasonality_period


def prepare_time_series(raw_timestamps: List[datetime.datetime]) -> Tuple[List[datetime.datetime], List[float], TimeSeriesContext]:
    """Aggregate packets, detect the dominant period, and align timestamps."""
    if not raw_timestamps:
        raise ValueError("No timestamps available for preprocessing")
    if not SQUASHER_ENABLED:
        raise ValueError("SQUASHER must be enabled for preprocessing")

    npackets = squash_timestamps_weighted(raw_timestamps, [1] * len(raw_timestamps), DT)
    ctx = detect_period_metadata(npackets, DT)
    timeline, custom_period = build_time_axis(raw_timestamps[0], len(npackets), ctx.period_seconds)
    ctx.custom_seasonality_period = custom_period
    print(f"  Expected cycles in data: {ctx.cycles:.2f}")
    print("\nData statistics:")
    if timeline:
        print(f"  Time range: {timeline[0]} to {timeline[-1]}")
    if npackets:
        print(f"  Packet range: {min(npackets):.1f} - {max(npackets):.1f}")
    print("=" * 60)
    return timeline, npackets, ctx


def augment_series(timestamps: List[datetime.datetime], values: List[float], ctx: TimeSeriesContext) -> Tuple[List[datetime.datetime], List[float]]:
    """Repeat the detected pattern to create additional training data."""
    if not AUGMENTER_ENABLED or not timestamps:
        return timestamps, values

    print(f"\nAugmenting data (repeating {AUGMENTER_MULTIPLIER}x)...")
    L = len(timestamps)
    time_delta = timestamps[-1] - timestamps[0]

    augmented_timestamps: List[datetime.datetime] = []
    augmented_values: List[float] = []

    for i in range(AUGMENTER_MULTIPLIER):
        offset = time_delta * i
        for idx in range(L):
            augmented_timestamps.append(timestamps[idx] + offset)
            augmented_values.append(values[idx])

    ctx.cycles *= AUGMENTER_MULTIPLIER
    print(f"Total data points: {len(augmented_timestamps)}")
    print(f"Total cycles: {ctx.cycles:.1f}")
    duration_days = (augmented_timestamps[-1] - augmented_timestamps[0]).total_seconds() / 86400
    print(f"Prophet dataset duration: {duration_days:.1f} days")
    return augmented_timestamps, augmented_values


def add_minimal_noise(values: List[float]) -> List[float]:
    """Inject a tiny amount of random noise to stabilize training."""
    if not RANDOM_NOISE_INJECTER:
        return values

    print(f"\nAdding minimal noise (±{RANDOM_NOISE_N_PACKETS} packets)...")
    noisy_values = values.copy()
    for idx in range(len(noisy_values)):
        noise = (np.random.random() - 0.5) * 2 * RANDOM_NOISE_N_PACKETS
        noisy_values[idx] = max(0, noisy_values[idx] + noise)
    return noisy_values


def plot_ts4prophet(x_values: List[datetime.datetime], y_values: List[float], ctx: TimeSeriesContext) -> None:
    """Plot the processed time series for sanity checks."""
    if not x_values:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.plot(x_values, y_values, linewidth=1.5, alpha=0.8, color='#2E86AB')
    ax1.set_xlabel('Timestamp', fontsize=11)
    ax1.set_ylabel('Number of Packets', fontsize=11)
    ax1.set_title('Network Traffic Distribution (Full Augmented Dataset)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    bins_per_cycle = max(1, int(ctx.bins_per_cycle))
    first_cycle_values = y_values[:bins_per_cycle]
    ax2.plot(range(len(first_cycle_values)), first_cycle_values,
             linewidth=2, alpha=0.8, color='#E74C3C', marker='o', markersize=3)
    ax2.set_xlabel('Bin Index', fontsize=11)
    ax2.set_ylabel('Number of Packets', fontsize=11)
    ax2.set_title(f'Single Cycle Shape ({ctx.period_seconds:.1f}s period → 24h in Prophet)',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('traffic_plot.png', dpi=150)
    print("\n✓ Plot saved: traffic_plot.png")
    plt.close()


def save_prophet_input(file_path: str, timestamps: List[datetime.datetime], values: List[float]) -> None:
    """Persist processed data for Prophet training."""
    with open(file_path, "w+") as output_file:
        output_file.write("ds,y\r\n")
        for ts, val in zip(timestamps, values):
            output_file.write(f"{datetime.datetime.strftime(ts, '%Y-%m-%d %H:%M:%S')},{val}\r\n")
    print(f"✓ Saved {len(timestamps)} records")


def build_prophet_model(ctx: TimeSeriesContext) -> Prophet:
    """Configure Prophet with the detected seasonality."""
    model = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=15.0,
        seasonality_mode='multiplicative',
        daily_seasonality=False if USE_REAL_TIMESTAMPS else True,
        weekly_seasonality=False,
        yearly_seasonality=False,
        interval_width=0.95
    )

    if USE_REAL_TIMESTAMPS and ctx.custom_seasonality_period:
        model.add_seasonality(
            name='traffic_pattern',
            period=ctx.custom_seasonality_period,
            fourier_order=10
        )
        print(f"\nAdded custom seasonality: {ctx.period_seconds:.1f}s period ({ctx.custom_seasonality_period:.4f} days)")
    else:
        print(f"\nUsing daily seasonality (mapped from {ctx.period_seconds:.1f}s period)")
    return model


def main() -> None:
    raw_timestamps = read_packet_timestamps(raw_input_file_path)
    processed_timestamps = trim_last_second(raw_timestamps)

    if not processed_timestamps:
        print("No usable timestamps after trimming. Exiting.")
        return

    timeline, npackets, ctx = prepare_time_series(processed_timestamps)
    augmented_timestamps, augmented_values = augment_series(timeline, npackets, ctx)

    if not augmented_timestamps:
        print("No data available after aggregation. Exiting.")
        return

    augmented_values = add_minimal_noise(augmented_values)
    plot_ts4prophet(augmented_timestamps, augmented_values, ctx)

    output_data_file = os.path.join('./', INPUT_CSV_FILE + "_prophet_input.csv")
    print(f"\nSaving Prophet input: {output_data_file}")
    save_prophet_input(output_data_file, augmented_timestamps, augmented_values)

    print("\n" + "=" * 60)
    print("TRAINING PROPHET MODEL")
    print("=" * 60)

    df = pd.read_csv(output_data_file)
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Packet count range: {df['y'].min():.2f} to {df['y'].max():.2f}")

    if len(df) < 10:
        print("\n⚠ WARNING: Very few data points!")
        return

    test_length = min(100, len(df) // 4)
    train_df = df.iloc[:-test_length].copy()
    test_df = df.iloc[-test_length:].copy()

    print(f"\nTrain/test split:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Testing: {test_length} samples")

    model = build_prophet_model(ctx)
    print(f"\nFitting model on training data (auto-detected {ctx.period_seconds:.1f}s period)...")
    model.fit(train_df)
    print("✓ Model trained")

    print(f"\nGenerating predictions for {test_length} test samples...")
    if USE_REAL_TIMESTAMPS:
        future = model.make_future_dataframe(periods=test_length, freq=f'{int(DT)}s')
    else:
        future = model.make_future_dataframe(periods=test_length, freq='h')

    fcst = model.predict(future)
    print("✓ Predictions generated")

    test_predictions = fcst.tail(test_length)['yhat'].values
    test_actuals = test_df['y'].values

    mse = mean_squared_error(test_actuals, test_predictions)
    mae = mean_absolute_error(test_actuals, test_predictions)
    rmse = np.sqrt(mse)

    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    print("\nGenerating plots...")
    fig1 = model.plot(fcst)
    plt.title(f'Network Traffic Forecast (Auto-detected {ctx.period_seconds:.1f}s period → 24h)',
              fontsize=13, fontweight='bold')
    plt.ylabel('Packets per Bin', fontsize=11)
    plt.xlabel('Date (Prophet Time)', fontsize=11)
    plt.tight_layout()
    plt.savefig('prophet_forecast.png', dpi=200)
    print("✓ Saved: prophet_forecast.png")
    plt.close()

    fig2 = model.plot_components(fcst)
    plt.suptitle(f'Components: Daily pattern = {ctx.period_seconds:.1f}s sine wave (auto-detected)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('prophet_components.png', dpi=200)
    print("✓ Saved: prophet_components.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    bins_in_2_cycles = max(1, int(2 * ctx.bins_per_cycle))
    plot_start = max(0, len(df) - bins_in_2_cycles)

    df['ds'] = pd.to_datetime(df['ds'])
    fcst['ds'] = pd.to_datetime(fcst['ds'])

    ax.plot(df['ds'].iloc[plot_start:], df['y'].iloc[plot_start:],
            label='Actual', linewidth=2.5, color='#2C3E50', marker='o', markersize=4)
    ax.plot(fcst['ds'].iloc[plot_start:], fcst['yhat'].iloc[plot_start:],
            label='Prophet Fit', linewidth=2, color='#E74C3C', linestyle='--')
    ax.fill_between(fcst['ds'].iloc[plot_start:],
                    fcst['yhat_lower'].iloc[plot_start:],
                    fcst['yhat_upper'].iloc[plot_start:],
                    alpha=0.3, color='#F5B7B1', label='95% Confidence')

    ax.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
    ax.set_ylabel('Packets per Bin', fontsize=12, fontweight='bold')
    ax.set_title(f'Last 2 Cycles: Prophet vs Actual (Period: {ctx.period_seconds:.1f}s)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prophet_fit_quality.png', dpi=200)
    print("✓ Saved: prophet_fit_quality.png")
    plt.close()

    print("\nSaving comparison data...")
    test_predictions_df = fcst.tail(test_length)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    test_predictions_df['actual'] = test_df['y'].values
    test_predictions_df.insert(0, 'index', range(len(test_predictions_df)))

    insample_file = INPUT_CSV_FILE + "_prophet_insample.csv"
    test_predictions_df.to_csv(insample_file, index=False)
    print(f"✓ Saved test set comparison: {insample_file} ({len(test_predictions_df)} samples)")
    if USE_REAL_TIMESTAMPS:
        print(f"  ✓ Using real {DT}s timestamps - compatible with GluonTS!")
    else:
        print("  Note: 'ds' column contains fake hourly timestamps for Prophet's seasonality")
        print("  Use 'index' column for alignment with GluonTS")

    future_periods = min(50, test_length // 2)
    future_extended = model.make_future_dataframe(
        periods=len(train_df) + test_length + future_periods,
        freq='h'
    )
    fcst_extended = model.predict(future_extended)

    future_only = fcst_extended.tail(future_periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_file = INPUT_CSV_FILE + "_prophet_forecast.csv"
    future_only.to_csv(forecast_file, index=False)
    print(f"✓ Saved future forecast: {forecast_file} ({len(future_only)} samples)")

    try:
        plot_plotly(model, fcst)
        plot_components_plotly(model, fcst)
        print("✓ Interactive plots generated")
    except Exception as exc:  # pragma: no cover - best effort only
        print(f"⚠ Interactive plots skipped: {exc}")

    print("\n" + "=" * 60)
    print("PROPHET ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Auto-detected period: {ctx.period_seconds:.1f}s → Mapped to 24h (daily)")
    print(f"Detection method: {ctx.detection_method}")
    print(f"Total cycles: {ctx.cycles:.1f}")
    print(f"Training samples: {len(df)}")
    print(f"Future predictions: {future_periods}")
    print("=" * 60)


if __name__ == "__main__":
    main()

