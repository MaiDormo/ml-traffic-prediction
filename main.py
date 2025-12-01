import pandas as pd
from config import Config
from data_processor import DataProcessor
from models import ProphetAdapter, DeepARAdapter
from visualizer import plot_forecast, save_metrics, plot_dataset_comparison, plot_model_comparison

def main():
    # 1. Setup: Load config and initialize processor
    cfg = Config()
    processor = DataProcessor(cfg)
    
    # 2. Data Pipeline: PCAP → DataFrame with augmentation
    print("--- Data Processing ---")
    df_aug, df_orig, meta = processor.load_and_process()
    
    # 3. Visualization: Compare raw vs augmented data
    overview_path = cfg.OUTPUT_DIR / "dataset_overview.png"
    plot_dataset_comparison(df_orig, df_aug, overview_path)
    
    # 4. Model Registry: Define adapters for each forecasting method
    models = [
        ("Prophet", ProphetAdapter(cfg, meta)),   # Facebook's additive model
        ("DeepAR", DeepARAdapter(cfg, meta))      # Amazon's RNN-based model
    ]
    
    results_store = {}  # Collect predictions for comparison plot
    
    # 5. Training Loop: Prepare → Train → Predict → Save
    print("\n--- Model Execution ---")
    for name, model in models:
        print(f"\nRunning {name}...")
        
        model.prepare_data(df_aug)  # Split into train/test
        model.train()                # Fit model parameters
        results = model.predict()    # Generate forecast DataFrame
        results_store[name] = results
        
        # Persist outputs
        results.to_csv(cfg.OUTPUT_DIR / f"forecast_{name.lower()}.csv", index=False)
        plot_forecast(results, f"{name} Forecast", cfg.OUTPUT_DIR / f"plot_{name.lower()}.png")
        save_metrics(results, name, cfg.OUTPUT_DIR / f"metrics_{name.lower()}.txt")

    # 6. Side-by-Side Comparison Plot
    print("\nGenerating comparison...")
    plot_model_comparison(results_store, cfg.OUTPUT_DIR / "model_comparison.png")

    # 7. Summary Table: Print final metrics
    print("\n--- Final Results Summary ---")
    summary_data = []
    for name, df in results_store.items():
        mse = ((df['actual'] - df['mean']) ** 2).mean()  # Mean Squared Error
        mae = (df['actual'] - df['mean']).abs().mean()   # Mean Absolute Error
        summary_data.append({'Model': name, 'MAE': mae, 'MSE': mse})
    
    print(pd.DataFrame(summary_data).to_string(index=False))
    print("\n✓ Pipeline Complete.")

if __name__ == "__main__":
    main()