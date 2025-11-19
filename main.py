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
    # Unpack tuple: (Augmented, Original, Metadata)
    df_aug, df_orig, meta = processor.load_and_process()
    
    # 3. Plot Overview (Original vs Augmented)
    overview_path = cfg.OUTPUT_DIR / "dataset_overview.png"
    plot_dataset_comparison(df_orig, df_aug, overview_path)
    
    # 4. Define Models
    models = [
        ("Prophet", ProphetAdapter(cfg, meta)),
        ("DeepAR", DeepARAdapter(cfg, meta))
    ]
    
    # Store results for comparison plot
    results_store = {}
    
    # 5. Train & Evaluate Loop
    print("\n--- Model Execution ---")
    for name, model in models:
        print(f"\nRunning {name}...")
        
        # Prepare & Train (Use Augmented DF for training)
        model.prepare_data(df_aug)
        model.train()
        
        # Predict
        results = model.predict()
        results_store[name] = results
        
        # Save Results
        csv_path = cfg.OUTPUT_DIR / f"forecast_{name.lower()}.csv"
        results.to_csv(csv_path, index=False)
        
        # Visualize Individual & Metrics
        plot_path = cfg.OUTPUT_DIR / f"plot_{name.lower()}.png"
        metric_path = cfg.OUTPUT_DIR / f"metrics_{name.lower()}.txt"
        
        plot_forecast(results, f"{name} Forecast (Period: {meta.period_seconds:.1f}s)", plot_path)
        save_metrics(results, name, metric_path)

    # 6. Generate Comparison Plot
    print("\nGenerating comparison...")
    comparison_path = cfg.OUTPUT_DIR / "model_comparison.png"
    plot_model_comparison(results_store, comparison_path)

    print("\n✓ Pipeline Complete. Check 'output/' directory.")

if __name__ == "__main__":
    main()