import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict

def plot_dataset_comparison(df_orig: pd.DataFrame, df_aug: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(14, 6))
    plt.plot(df_aug['timestamp'], df_aug['target'], label='Augmented / Synthetic', color='#7f8c8d', linestyle='--', alpha=0.6)
    plt.plot(df_orig['timestamp'], df_orig['target'], label='Original Capture', color='#c0392b', linewidth=2.5, alpha=1.0)
    plt.title("Original vs Augmented Data", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved data overview: {output_path}")

def plot_forecast(df: pd.DataFrame, title: str, output_path: Path):
    plt.figure(figsize=(14, 7))
    plt.plot(df['timestamp'], df['actual'], label='Actual', color='black', marker='.', alpha=0.5)
    plt.plot(df['timestamp'], df['mean'], label='Forecast', color='#0072B2', linewidth=2)
    plt.fill_between(df['timestamp'], df['lower'], df['upper'], color='#0072B2', alpha=0.2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved forecast plot: {output_path}")

def plot_model_comparison(results_dict: Dict[str, pd.DataFrame], output_path: Path):
    # Subplot 1: Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    colors = {'Prophet': '#2980b9', 'DeepAR': '#e67e22'}
    
    first_key = list(results_dict.keys())[0]
    df_ref = results_dict[first_key]
    ax1.plot(df_ref['timestamp'], df_ref['actual'], label='Actual', color='black', linewidth=2, alpha=0.8)

    for name, df in results_dict.items():
        c = colors.get(name, 'green')
        ax1.plot(df['timestamp'], df['mean'], label=f'{name}', color=c, linewidth=2)
        
        # Subplot 2: Residuals
        residuals = df['actual'] - df['mean']
        ax2.plot(df['timestamp'], residuals, label=f'{name} Error', color=c, alpha=0.7)

    ax1.set_title("Model Forecast Comparison", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Packets per Bin")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Residuals (Actual - Forecast)", fontsize=12)
    ax2.set_ylabel("Error")
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3)
    
    plt.xlabel("Timestamp")
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