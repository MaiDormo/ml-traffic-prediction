import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict

def plot_dataset_comparison(df_orig: pd.DataFrame, df_aug: pd.DataFrame, output_path: Path):
    """Plots the original captured data overlaid on the augmented dataset."""
    plt.figure(figsize=(14, 6))
    
    # 1. Plot Augmented (Background)
    plt.plot(df_aug['timestamp'], df_aug['target'], 
             label='Augmented / Synthetic Data', 
             color='#7f8c8d', # Grey
             linestyle='--', 
             linewidth=1.5,
             alpha=0.6)
             
    # 2. Plot Original (Foreground)
    plt.plot(df_orig['timestamp'], df_orig['target'], 
             label='Original Capture (Ground Truth)', 
             color='#c0392b', # Dark Red
             linewidth=2.5,
             alpha=1.0)
             
    plt.title("Network Traffic: Original Capture vs. Augmented Training Data", fontsize=14, fontweight='bold')
    plt.xlabel("Timestamp")
    plt.ylabel("Packets per Bin")
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved data overview plot: {output_path}")

def plot_forecast(df: pd.DataFrame, title: str, output_path: Path):
    """Individual model plot with confidence intervals."""
    plt.figure(figsize=(14, 7))
    
    plt.plot(df['timestamp'], df['actual'], 
             label='Actual', color='black', marker='.', alpha=0.5)
    
    plt.plot(df['timestamp'], df['mean'], 
             label='Forecast', color='#0072B2', linewidth=2)
    
    plt.fill_between(
        df['timestamp'], 
        df['lower'], 
        df['upper'], 
        color='#0072B2', 
        alpha=0.2, 
        label='Confidence Interval'
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Timestamp")
    plt.ylabel("Packets")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved forecast plot: {output_path}")

def plot_model_comparison(results_dict: Dict[str, pd.DataFrame], output_path: Path):
    """
    Plots multiple model predictions against the actual data on one chart.
    """
    plt.figure(figsize=(14, 7))
    
    # Colors and Styles
    colors = {'Prophet': '#2980b9', 'DeepAR': '#e67e22'} # Blue, Orange
    styles = {'Prophet': '-', 'DeepAR': '--'}
    
    # 1. Plot Actual Data (Use the first model's dataframe as reference)
    first_key = list(results_dict.keys())[0]
    df_ref = results_dict[first_key]
    
    plt.plot(df_ref['timestamp'], df_ref['actual'], 
             label='Actual Data', 
             color='black', 
             linewidth=2.5, 
             alpha=0.8,
             zorder=1) # Keep actuals behind lines if needed, or zorder=10 to put on top

    # 2. Plot Each Model
    for name, df in results_dict.items():
        c = colors.get(name, 'green') # Fallback color
        s = styles.get(name, '-')
        
        plt.plot(df['timestamp'], df['mean'], 
                 label=f'{name} Prediction', 
                 color=c, 
                 linestyle=s, 
                 linewidth=2,
                 alpha=0.9,
                 zorder=5)

    plt.title("Model Comparison: Prophet vs. DeepAR (Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("Timestamp")
    plt.ylabel("Packets per Bin")
    plt.legend(loc='best', frameon=True, fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"✓ Saved comparison plot: {output_path}")

def save_metrics(df: pd.DataFrame, name: str, output_path: Path):
    mse = ((df['actual'] - df['mean']) ** 2).mean()
    mae = (df['actual'] - df['mean']).abs().mean()
    
    with open(output_path, 'w') as f:
        f.write(f"Model: {name}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write("-" * 20 + "\n")
    
    print(f"  {name} Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}")