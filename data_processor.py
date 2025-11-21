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
    from scapy.all import PcapReader, rdpcap, conf
    # Ensure SLL (Linux Cooked) support is enabled
    conf.load_layers.append("sll")
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
        timestamps = self._read_data()
        
        if len(timestamps) < 10:
            # Create dummy data to prevent crashes if capture failed
            print("WARNING: Not enough data captured. Generating dummy fallback data.")
            base = datetime.datetime.now()
            timestamps = [base + datetime.timedelta(seconds=i*0.5) for i in range(100)]

        timeline_orig, counts_orig = self._squash_timestamps(timestamps)
        
        df_orig = pd.DataFrame({'timestamp': timeline_orig, 'target': counts_orig})
        df_orig['timestamp'] = pd.to_datetime(df_orig['timestamp'])
        
        meta = self._detect_period(counts_orig)
        
        timeline = list(timeline_orig)
        counts = list(counts_orig)
        
        if self.cfg.AUGMENT_MULTIPLIER > 1:
            timeline, counts, meta = self._augment(timeline, counts, meta)
            
        if self.cfg.ADD_NOISE:
            counts = self._add_noise(counts)

        df_aug = pd.DataFrame({'timestamp': timeline, 'target': counts})
        df_aug['timestamp'] = pd.to_datetime(df_aug['timestamp'])
        
        self._save_intermediates(df_orig, df_aug)
        
        return df_aug, df_orig, meta

    def _read_data(self) -> List[datetime.datetime]:
        file_path = self.cfg.INPUT_FILE
        print(f"Reading input: {file_path}...")
        
        if not file_path.exists():
            # Fallback for finding the file if moved by shell script
            alt_path = file_path.parent / "traffic_capture.pcap"
            if alt_path.exists():
                file_path = alt_path
            else:
                print(f"ERROR: File {file_path} not found.")
                return []

        if file_path.suffix == '.csv':
            return self._read_csv(file_path)
        elif file_path.suffix in ['.pcap', '.cap']:
            return self._read_pcap(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _read_csv(self, path) -> List[datetime.datetime]:
        ts_list = []
        try:
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)
                base_time = datetime.datetime.now()
                for row in reader:
                    try:
                        ts_list.append(base_time + datetime.timedelta(seconds=float(row[1])))
                    except (IndexError, ValueError):
                        continue
        except Exception as e:
            print(f"CSV Read Error: {e}")
        return self._trim_last_second(ts_list)

    def _read_pcap(self, path) -> List[datetime.datetime]:
        if not SCAPY_AVAILABLE:
            raise ImportError("Please run: pip install scapy")
            
        ts_list = []
        
        # Strategy 1: Efficient streaming (PcapReader)
        try:
            with PcapReader(str(path)) as pcap_reader:
                for pkt in pcap_reader:
                    # We only care about metadata (time), not payload
                    ts_list.append(datetime.datetime.fromtimestamp(float(pkt.time)))
        except Exception as e:
            print(f"  Stream reader failed ({e}). Trying robust load...")
            
            # Strategy 2: Load All (rdpcap) - Robust to weird headers
            try:
                packets = rdpcap(str(path))
                ts_list = [datetime.datetime.fromtimestamp(float(pkt.time)) for pkt in packets]
            except Exception as e2:
                print(f"  CRITICAL: Could not parse PCAP. {e2}")
        
        print(f"  Loaded {len(ts_list)} packets from PCAP")
        return self._trim_last_second(ts_list)

    def _trim_last_second(self, ts: List[datetime.datetime]) -> List[datetime.datetime]:
        if not ts: return []
        ts.sort()
        cutoff = ts[-1] - datetime.timedelta(seconds=1.0)
        return [t for t in ts if t < cutoff]

    def _squash_timestamps(self, ts: List[datetime.datetime]) -> Tuple[List[datetime.datetime], List[float]]:
        if not ts: return [], []
        
        weighted = []
        current_bin_val = 0.0
        
        start_time = ts[0].replace(microsecond=0)
        pts = start_time
        
        for t in ts:
            while (t - pts).total_seconds() >= self.cfg.DT:
                weighted.append(current_bin_val)
                current_bin_val = 0.0
                pts += datetime.timedelta(seconds=self.cfg.DT)
            current_bin_val += 1.0
            
        weighted.append(current_bin_val)
        timeline = [start_time + datetime.timedelta(seconds=i*self.cfg.DT) for i in range(len(weighted))]
        return timeline, weighted

    def _detect_period(self, data: List[float]) -> DatasetMeta:
        arr = np.array(data)
        n = len(arr)
        if n < 2: return DatasetMeta(self.cfg.DT, 1, 0, "None")

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
            period = n * self.cfg.DT / 2 

        bins_per_cycle = period / self.cfg.DT
        cycles = n / bins_per_cycle
        print(f"Detected Period: {period:.2f}s ({method}) | Cycles: {cycles:.1f}")
        return DatasetMeta(period, bins_per_cycle, cycles, method)

    def _augment(self, timeline, values, meta):
        if not timeline: return timeline, values, meta
        print(f"Augmenting dataset {self.cfg.AUGMENT_MULTIPLIER}x...")
        aug_time = []
        aug_vals = []
        cycle_duration = datetime.timedelta(seconds=len(timeline) * self.cfg.DT)
        
        for i in range(self.cfg.AUGMENT_MULTIPLIER):
            offset = cycle_duration * i
            aug_time.extend([t + offset for t in timeline])
            aug_vals.extend(values)
            
        meta.total_cycles *= self.cfg.AUGMENT_MULTIPLIER
        return aug_time, aug_vals, meta

    def _add_noise(self, values):
        print(f"Injecting noise (level {self.cfg.NOISE_LEVEL})...")
        noise = (np.random.random(len(values)) - 0.5) * 2 * self.cfg.NOISE_LEVEL
        return [max(0, v + n) for v, n in zip(values, noise)]

    def _save_intermediates(self, df_orig, df_aug):
        df_orig.to_csv(self.cfg.OUTPUT_DIR / "processed_original.csv", index=False)
        df_aug.to_csv(self.cfg.OUTPUT_DIR / "processed_augmented.csv", index=False)