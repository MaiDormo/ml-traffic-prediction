from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Update this to point to your actual capture file location
    # If using the script, it might be in /tmp/traffic_capture.pcap
    INPUT_FILE: Path = Path("./traffic_capture.pcap") 
    OUTPUT_DIR: Path = Path("output")
    
    # Preprocessing
    DT: float = 2.0  # Bin size in seconds (Aggregation window)
    AUGMENT_MULTIPLIER: int = 3
    ADD_NOISE: bool = True
    NOISE_LEVEL: int = 2
    
    # Model Settings
    TEST_SPLIT_RATIO: float = 0.20
    MAX_TEST_SAMPLES: int = 100
    
    # GluonTS Specific
    # INCREASED: Give model more history (2 full cycles) to see the pattern repeats
    CONTEXT_LENGTH_MULTIPLIER: int = 2  
    # INCREASED: DeepAR needs more epochs to converge on complex patterns
    EPOCHS: int = 30                    
    
    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        if not self.INPUT_FILE.exists():
            # Fallback or warning
            print(f"Warning: {self.INPUT_FILE} not found.")

    @property
    def freq_str(self) -> str:
        return f"{int(self.DT)}s"