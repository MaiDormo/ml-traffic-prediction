from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Central configuration for the ML traffic prediction pipeline."""
    
    # --- File Paths ---
    INPUT_FILE: Path = Path("./traffic_capture.pcap")  # Raw packet capture
    OUTPUT_DIR: Path = Path("output")                   # All generated artifacts
    
    # --- Preprocessing ---
    DT: float = 2.0              # Bin size in seconds (aggregation window)
    AUGMENT_MULTIPLIER: int = 3  # Replicate dataset N times for more training data
    ADD_NOISE: bool = True       # Enable multiplicative noise injection
    NOISE_LEVEL: int = 2         # Noise std as percentage of signal (2%)
    
    # --- Train/Test Split ---
    TEST_SPLIT_RATIO: float = 0.20  # Fraction of data reserved for testing
    MAX_TEST_SAMPLES: int = 100     # Hard cap on test set size (1 cycle)
    
    # --- DeepAR Hyperparameters ---
    CONTEXT_LENGTH_MULTIPLIER: int = 2  # RNN sees N full cycles as history
    EPOCHS: int = 30                     # Training iterations (increase for complex patterns)
    
    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(exist_ok=True)  # Ensure output directory exists
        if not self.INPUT_FILE.exists():
            print(f"Warning: {self.INPUT_FILE} not found.")

    @property
    def freq_str(self) -> str:
        """GluonTS frequency string (e.g., '2s' for 2-second bins)."""
        return f"{int(self.DT)}s"