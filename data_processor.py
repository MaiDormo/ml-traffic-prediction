import csv
import datetime
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy import signal
from typing import List, Tuple
from dataclasses import dataclass
from config import Config

# Try to import scapy, handle error if missing
try:
    from scapy.utils import PcapReader
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

@dataclass
class DatasetMeta:
    period_seconds: float
    bins_per_cycle: float
    total_cycles: float
    detection_method: str

class DataProcessor:
    def __init__(self, config: Config):
        self.cfg = config

    def load_and_process(self) -> Tuple[pd.DataFrame, pd.DataFrame, DatasetMeta]:
        """
        Main pipeline entry point.
        Returns: (Augmented DataFrame, Original DataFrame, Metadata)
        """
        # 1. Load (Smart detection of CSV vs PCAP)
        timestamps = self._read_data()
        
        # 2. Squash (Binning)
        timeline_orig, counts_orig = self._squash_timestamps(timestamps)
        
        # 3. Create Original DataFrame
        df_orig = pd.DataFrame({'timestamp': timeline_orig, 'target': counts_orig})
        df_orig['timestamp'] = pd.to_datetime(df_orig['timestamp'])
        
        # 4. Detect Period
        meta = self._detect_period(counts_orig)
        
        # 5. Augment
        timeline = list(timeline_orig)
        counts = list(counts_orig)
        
        if self.cfg.AUGMENT_MULTIPLIER > 1:
            timeline, counts, meta = self._augment(timeline, counts, meta)
            
        # 6. Noise
        if self.cfg.ADD_NOISE:
            counts = self._add_noise(counts)

        # 7. Final Augmented DataFrame
        df_aug = pd.DataFrame({'timestamp': timeline, 'target': counts})
        df_aug['timestamp'] = pd.to_datetime(df_aug['timestamp'])
        
        # Save intermediates
        self._save_intermediates(df_orig, df_aug)
        
        return df_aug, df_orig, meta

    def _read_data(self) -> List[datetime.datetime]:
        file_path = self.cfg.INPUT_FILE
        print(f"Reading input: {file_path}...")
        
        if file_path.suffix == '.csv':
            return self._read_csv(file_path)
        elif file_path.suffix == '.pcap' or file_path.suffix == '.cap':
            return self._read_pcap(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _read_csv(self, path) -> List[datetime.datetime]:
        ts_list = []
        try:
            with open(path) as f:
                reader = csv.reader(f)
                next(reader) # Skip header
                base_time = datetime.datetime.now()
                for row in reader:
                    try:
                        # Legacy: CSV had relative time, convert to absolute
                        ts_list.append(base_time + datetime.timedelta(seconds=float(row[1])))
                    except (IndexError, ValueError):
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {path}")
        return self._trim_last_second(ts_list)

    def _read_pcap(self, path) -> List[datetime.datetime]:
        if not SCAPY_AVAILABLE:
            raise ImportError("Please run: pip install scapy")
            
        ts_list = []
        try:
            # Use PcapReader for memory efficient streaming
            with PcapReader(str(path)) as pcap_reader:
                for pkt in pcap_reader:
                    # Extract timestamp from packet meta
                    ts = float(pkt.time)
                    ts_list.append(datetime.datetime.fromtimestamp(ts))
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {path}")
        
        print(f"  Loaded {len(ts_list)} packets from PCAP")
        return self._trim_last_second(ts_list)

    def _trim_last_second(self, ts: List[datetime.datetime]) -> List[datetime.datetime]:
        if not ts: return []
        # Sort to be safe (PCAP usually sorted, but good practice)
        ts.sort()
        cutoff = ts[-1] - datetime.timedelta(seconds=1.0)
        return [t for t in ts if t < cutoff]

    def _squash_timestamps(self, ts: List[datetime.datetime]) -> Tuple[List[datetime.datetime], List[float]]:
        if not ts: return [], []
        
        weighted = []
        current_bin_val = 0.0
        
        start_time = ts[0]
        # Round start_time down to nearest second for cleaner plots
        start_time = start_time.replace(microsecond=0)
        
        pts = start_time
        
        # Logic: Iterate through packets. If a packet falls into the next 'DT' bin,
        # close the current bin and start a new one.
        for t in ts:
            while (t - pts).total_seconds() >= self.cfg.DT:
                weighted.append(current_bin_val)
                current_bin_val = 0.0
                pts += datetime.timedelta(seconds=self.cfg.DT)
            
            current_bin_val += 1.0 # Count 1 packet
            
        weighted.append(current_bin_val)
        
        # Create timeline axis
        timeline = [start_time + datetime.timedelta(seconds=i*self.cfg.DT) for i in range(len(weighted))]
        return timeline, weighted

    def _detect_period(self, data: List[float]) -> DatasetMeta:
        arr = np.array(data)
        n = len(arr)
        if n == 0: return DatasetMeta(self.cfg.DT, 1, 0, "Empty")

        centered = arr - np.mean(arr)
        yf = fft(centered)
        xf = fftfreq(n, self.cfg.DT)
        mask = xf > 0
        
        period = -1.0
        method = "Fallback"
        
        if np.any(mask):
            dominant_freq = xf[mask][np.argmax(np.abs(yf[mask]))]
            if dominant_freq > 0:
                period = 1.0 / dominant_freq
                method = "FFT"

        if period <= 0:
            corr = signal.correlate(centered, centered, mode='full')[n//2:]
            peaks, _ = signal.find_peaks(corr)
            if len(peaks) > 0:
                period = peaks[0] * self.cfg.DT
                method = "Autocorrelation"
        
        if period <= 0:
            period = n * self.cfg.DT / 2 

        bins_per_cycle = period / self.cfg.DT
        cycles = n / bins_per_cycle
        print(f"Detected Period: {period:.2f}s ({method}) | Cycles: {cycles:.1f}")
        return DatasetMeta(period, bins_per_cycle, cycles, method)

    def _augment(self, timeline: List[datetime.datetime], values: List[float], meta: DatasetMeta) -> Tuple[List[datetime.datetime], List[float], DatasetMeta]:
        if not timeline: return timeline, values, meta

        print(f"Augmenting dataset {self.cfg.AUGMENT_MULTIPLIER}x...")
        aug_time = []
        aug_vals = []
        
        # Cycle duration = total length of data
        cycle_duration = datetime.timedelta(seconds=len(timeline) * self.cfg.DT)
        
        for i in range(self.cfg.AUGMENT_MULTIPLIER):
            offset = cycle_duration * i
            aug_time.extend([t + offset for t in timeline])
            aug_vals.extend(values)
            
        meta.total_cycles *= self.cfg.AUGMENT_MULTIPLIER
        return aug_time, aug_vals, meta

    def _add_noise(self, values: List[float]) -> List[float]:
        print(f"Injecting noise (level {self.cfg.NOISE_LEVEL})...")
        noise = (np.random.random(len(values)) - 0.5) * 2 * self.cfg.NOISE_LEVEL
        return [max(0, v + n) for v, n in zip(values, noise)]

    def _save_intermediates(self, df_orig: pd.DataFrame, df_aug: pd.DataFrame):
        df_orig.to_csv(self.cfg.OUTPUT_DIR / "processed_original.csv", index=False)
        df_aug.to_csv(self.cfg.OUTPUT_DIR / "processed_augmented.csv", index=False)