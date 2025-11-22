import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import warnings

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
        """Standard train/test split based on config."""
        self.test_len = min(self.cfg.MAX_TEST_SAMPLES, int(len(df) * self.cfg.TEST_SPLIT_RATIO))
        self.train_df = df.iloc[:-self.test_len].copy()
        self.test_df = df.copy() # Full dataset often needed for evaluation context

    @abstractmethod
    def train(self): pass

    @abstractmethod
    def predict(self) -> pd.DataFrame: 
        """Must return DataFrame with columns: [timestamp, mean, lower, upper, actual]"""
        pass

class ProphetAdapter(BaseModel):
    def train(self):
        # Rename for Prophet
        df_prophet = self.train_df.rename(columns={'timestamp': 'ds', 'target': 'y'})
        
        self.model = Prophet(
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=15.0,
            daily_seasonality=False,
            weekly_seasonality=False
        )
        
        # Add custom seasonality derived from FFT
        # Prophet expects period in days
        period_days = self.meta.period_seconds / 86400.0
        self.model.add_seasonality(name='custom_cycle', period=period_days, fourier_order=10)
        
        self.model.fit(df_prophet)

    def predict(self) -> pd.DataFrame:
        future = self.model.make_future_dataframe(
            periods=self.test_len, 
            freq=self.cfg.freq_str
        )
        forecast = self.model.predict(future)
        
        # Standardize output
        result = forecast.tail(self.test_len).copy()
        result = result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        result.columns = ['timestamp', 'mean', 'lower', 'upper']
        result['actual'] = self.test_df.iloc[-self.test_len:]['target'].values
        return result

class DeepARAdapter(BaseModel):
    def __init__(self, config: Config, meta: DatasetMeta):
        super().__init__(config, meta)
        self.predictor = None

    def train(self):
        # Adaptive parameters
        prediction_length = self.test_len
        context_length = int(self.meta.bins_per_cycle * self.cfg.CONTEXT_LENGTH_MULTIPLIER)
        
        ds = PandasDataset(
            self.train_df.set_index('timestamp'), 
            target='target', 
            freq=self.cfg.freq_str
        )
        
        # --- Trainer Configuration ---
        # Standard CPU execution
        trainer_kwargs = {
            'max_epochs': self.cfg.EPOCHS,
        }
        
        # --- Model Architecture ---
        estimator = DeepAREstimator(
            freq=self.cfg.freq_str,
            prediction_length=prediction_length,
            context_length=context_length,
            
            # Scaling handles traffic volume differences automatically
            scaling=True,
            
            trainer_kwargs=trainer_kwargs
        )
        
        self.predictor = estimator.train(ds)

    def predict(self) -> pd.DataFrame:
        ds_test = PandasDataset(
            self.test_df.set_index('timestamp'), 
            target='target', 
            freq=self.cfg.freq_str
        )
        
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=ds_test, 
            predictor=self.predictor, 
            num_samples=100
        )
        
        forecast = list(forecast_it)[0]
        
        # Extract data for standard format
        dates = pd.date_range(
            start=self.test_df.iloc[-self.test_len]['timestamp'], 
            periods=self.test_len, 
            freq=self.cfg.freq_str
        )
        
        return pd.DataFrame({
            'timestamp': dates,
            'mean': forecast.mean,
            'lower': forecast.quantile(0.1), # 80% interval
            'upper': forecast.quantile(0.9),
            'actual': self.test_df.iloc[-self.test_len:]['target'].values
        })