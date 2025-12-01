import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import warnings

# -----------------------------------------------------------------------------
# GLUONTS & PYTORCH COMPATIBILITY PATCH
# -----------------------------------------------------------------------------
# Fixes a deprecation warning/error in PyTorch 2.9+ where list indexing 
# is no longer supported for multidimensional tensors. We monkey-patch 
# GluonTS to use tuple indexing instead.
# -----------------------------------------------------------------------------
try:
    import gluonts.torch.util

    def fixed_slice_along_dim(a, dim, slice_):
        idx = [slice(None)] * len(a.shape)
        idx[dim] = slice_
        return a[tuple(idx)]

    gluonts.torch.util.slice_along_dim = fixed_slice_along_dim
    gluonts.torch.util.take_slice = lambda a, idx: a[tuple(idx)] if isinstance(idx, list) else a[idx]

except ImportError:
    pass

# --- GLUONTS IMPORTS (Safe to load now) ---
from prophet import Prophet
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions
from config import Config
from data_processor import DatasetMeta

class BaseModel(ABC):
    def __init__(self, config: Config, meta: DatasetMeta):
        self.cfg = config
        self.meta = meta
        self.train_df = None
        self.test_df = None
        self.test_len = 0

    def prepare_data(self, df: pd.DataFrame):
        self.test_len = min(self.cfg.MAX_TEST_SAMPLES, int(len(df) * self.cfg.TEST_SPLIT_RATIO))
        
        self.train_df = df.iloc[:-self.test_len].copy()
        self.test_df = df.iloc[-self.test_len:].copy()     
        
        print(f"\nTrain/Test Split ({(1-self.cfg.TEST_SPLIT_RATIO)*100:.0f}/{self.cfg.TEST_SPLIT_RATIO*100:.0f}):")
        print(f"   Train: {len(self.train_df)} samples ({100*len(self.train_df)/len(df):.1f}%)")
        print(f"   Test:  {len(self.test_df)} samples ({100*len(self.test_df)/len(df):.1f}%)")

    @abstractmethod
    def train(self): pass

    @abstractmethod
    def predict(self) -> pd.DataFrame: 
        """Must return DataFrame with columns: [timestamp, mean, lower, upper, actual]"""
        pass

class ProphetAdapter(BaseModel):
    def train(self):
        
        # Prophet requires specific column names: 'ds' (date) and 'y' (value)
        df_prophet = self.train_df.rename(columns={'timestamp': 'ds', 'target': 'y'})
        
        self.model = Prophet(
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=15.0,
            daily_seasonality=False,
            weekly_seasonality=False
        )
        
        # # Inject custom seasonality derived from the data processor (FFT or Manual)
        period_days = self.meta.period_seconds / 86400.0
        self.model.add_seasonality(name='custom_cycle', period=period_days, fourier_order=10)
        self.model.fit(df_prophet)

    def predict(self) -> pd.DataFrame:
        # Create a dataframe holding dates for the test period
        future = self.model.make_future_dataframe(
            periods=self.test_len, 
            freq=self.cfg.freq_str
        )
        forecast = self.model.predict(future)
        
        # Extract only the test period predictions
        result = forecast.tail(self.test_len).copy()
        result = result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        result.columns = ['timestamp', 'mean', 'lower', 'upper']

        # Attach actual values for metric calculation
        result['actual'] = self.test_df['target'].values
        return result
    
class DeepARAdapter(BaseModel):
    def __init__(self, config: Config, meta: DatasetMeta):
        super().__init__(config, meta)
        self.predictor = None

    def train(self):
        # DeepAR Hyperparameters
        prediction_length = self.test_len
        # Context length determines how far back the RNN looks to establish state
        context_length = int(self.meta.bins_per_cycle * self.cfg.CONTEXT_LENGTH_MULTIPLIER)
        
        # Convert pandas DataFrame to GluonTS dataset format
        ds = PandasDataset(
            self.train_df.set_index('timestamp'), 
            target='target', 
            freq=self.cfg.freq_str
        )
        
        trainer_kwargs = {
            'max_epochs': self.cfg.EPOCHS,
        }
        
        estimator = DeepAREstimator(
            freq=self.cfg.freq_str,
            prediction_length=prediction_length,
            context_length=context_length,
            scaling=True, # Automatically scales data to handle varying traffic volumes           
            trainer_kwargs=trainer_kwargs
        )
        
        self.predictor = estimator.train(ds)

    def predict(self) -> pd.DataFrame:
        # ---------------------------------------------------------
        # Context Handling for DeepAR
        # ---------------------------------------------------------
        # Unlike Prophet, DeepAR is an RNN that requires historical context 
        # to initialize its hidden state before making a prediction.
        #
        # We pass the FULL dataset (Train + Test). GluonTS automatically:
        # 1. Uses the 'Train' portion as context to prime the RNN.
        # 2. Masks the 'Test' portion so the model cannot see the answers.
        # 3. Generates predictions for the 'Test' window.
        # ---------------------------------------------------------
        
        full_df = pd.concat([self.train_df, self.test_df])
        
        ds_test = PandasDataset(
            full_df.set_index('timestamp'), 
            target='target', 
            freq=self.cfg.freq_str
        )
        
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=ds_test, 
            predictor=self.predictor, 
            num_samples=100 # Generate 100 sample paths for probabilistic bounds
        )
        
        forecast = list(forecast_it)[0]
        
        # Reconstruct the timestamp index for the test period
        dates = pd.date_range(
            start=self.test_df.iloc[0]['timestamp'], 
            periods=self.test_len, 
            freq=self.cfg.freq_str
        )
        
        return pd.DataFrame({
            'timestamp': dates,
            'mean': forecast.mean,
            'lower': forecast.quantile(0.1),
            'upper': forecast.quantile(0.9),
            'actual': self.test_df['target'].values
        })