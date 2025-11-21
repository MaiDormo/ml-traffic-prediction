import pandas as pd
from config import Config
from data_processor import DataProcessor
from models import ProphetAdapter, DeepARAdapter
from visualizer import plot_forecast, save_metrics, plot_dataset_comparison, plot_model_comparison

def main():
    # 1. Setup
    cfg = Config()
    processor = DataProcessor(cfg)
    
    # 2. Process Data
    print("--- Data Processing ---")
    df_aug, df_orig, meta = processor.load_and_process()
    
    # 3. Plot Overview
    overview_path = cfg.OUTPUT_DIR / "dataset_overview.png"
    plot_dataset_comparison(df_orig, df_aug, overview_path)
    
    # 4. Define Models
    models = [
        ("Prophet", ProphetAdapter(cfg, meta)),
        ("DeepAR", DeepARAdapter(cfg, meta))
    ]
    
    results_store = {}
    
    # 5. Train & Evaluate Loop
    print("\n--- Model Execution ---")
    for name, model in models:
        print(f"\nRunning {name}...")
        model.prepare_data(df_aug)
        model.train()
        results = model.predict()
        results_store[name] = results
        
        results.to_csv(cfg.OUTPUT_DIR / f"forecast_{name.lower()}.csv", index=False)
        plot_forecast(results, f"{name} Forecast", cfg.OUTPUT_DIR / f"plot_{name.lower()}.png")
        save_metrics(results, name, cfg.OUTPUT_DIR / f"metrics_{name.lower()}.txt")

    # 6. Comparison
    print("\nGenerating comparison...")
    plot_model_comparison(results_store, cfg.OUTPUT_DIR / "model_comparison.png")

    # 7. Summary Table
    print("\n--- Final Results Summary ---")
    summary_data = []
    for name, df in results_store.items():
        mse = ((df['actual'] - df['mean']) ** 2).mean()
        mae = (df['actual'] - df['mean']).abs().mean()
        summary_data.append({'Model': name, 'MAE': mae, 'MSE': mse})
    
    print(pd.DataFrame(summary_data).to_string(index=False))
    print("\n✓ Pipeline Complete.")

if __name__ == "__main__":
    main()