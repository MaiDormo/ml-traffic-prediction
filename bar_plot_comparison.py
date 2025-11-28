import matplotlib.pyplot as plt

from pathlib import Path

# 1. Setup Paths
# Assuming files are in the 'output' directory based on your workspace
files = [
    Path("output/metrics_deepar.txt"),
    Path("output/metrics_prophet.txt")
]

# Data storage
results = {}

# 2. Parse Files
print("Reading metrics...")
for file_path in files:
    if file_path.exists():
        with open(file_path, "r") as f:
            lines = f.readlines()
            # Parse format: "Key: Value"
            name = lines[0].split(":")[1].strip()
            mse = float(lines[1].split(":")[1].strip())
            mae = float(lines[2].split(":")[1].strip())
            results[name] = {"MSE": mse, "MAE": mae}
    else:
        print(f"Warning: {file_path} not found. Using dummy data for visualization.")
        # Fallback for demonstration if files aren't generated yet
        if "deepar" in str(file_path): 
            results["DeepAR"] = {"MSE": 3123.41, "MAE": 37.42}
        if "prophet" in str(file_path): 
            results["Prophet"] = {"MSE": 1521778.66, "MAE": 1026.79}

# 3. Prepare Data for Plotting
models = list(results.keys())
mae_values = [results[m]["MAE"] for m in models]
mse_values = [results[m]["MSE"] for m in models]

# Define colors (DeepAR=Orange, Prophet=Blue to match your project style)
colors = ['#e67e22' if 'DeepAR' in m else '#2980b9' for m in models]

# 4. Create Plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot 1: MAE ---
bars1 = ax1.bar(models, mae_values, color=colors, width=0.5, edgecolor='black', alpha=0.8)
ax1.set_title("Mean Absolute Error (MAE)\n(Lower is Better)", fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel("Error Magnitude")

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + (max(mae_values)*0.01),
             f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# --- Subplot 2: MSE ---
bars2 = ax2.bar(models, mse_values, color=colors, width=0.5, edgecolor='black', alpha=0.8)
ax2.set_title("Mean Squared Error (MSE)\n(Lower is Better)", fontsize=14, fontweight='bold', pad=15)

# Add value labels on top of bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (max(mse_values)*0.01),
             f'{height:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5. Final Layout Adjustments
plt.suptitle("Model Performance Comparison", fontsize=18, y=1.02)
plt.tight_layout()

# Save or Show
output_file = "output/metrics_comparison_bar.png"
plt.savefig(output_file, bbox_inches='tight', dpi=150)
print(f"Plot saved to {output_file}")
plt.show()