# Network Traffic Prediction

This project implements an end-to-end machine learning pipeline for simulating, capturing, and forecasting network traffic. It integrates **Mininet** for network simulation with **Prophet** and **GluonTS (DeepAR)** for time-series forecasting, providing a comparative analysis of statistical versus deep learning approaches.

## 🚀 Features

- **Network Simulation**: Custom Mininet topologies (Star, Tree) with SDN controller integration (Ryu).
- **Interactive Demo**: Jupyter notebook for multi-day traffic simulation and forecasting without root privileges.
- **Traffic Generation**: Diverse traffic patterns including Periodic, Random Bursts, and Daily Cycles.
- **Data Pipeline**: Automated PCAP parsing, time-binning, data augmentation, and noise injection.
- **Model Comparison**: Side-by-side evaluation of Prophet and DeepAR on identical training splits.

## 📂 Project Structure

- `network_simulator.py`: Mininet topology and traffic generation script.
- `vm_traffic_pipeline.sh`: Shell script to orchestrate Mininet simulation and capture.
- `demo_multi_day.ipynb`: Interactive notebook for multi-day simulation and model benchmarking.
- `sdn_controller.py`: Ryu-based SDN controller for flow management.
- `data_processor.py`: Handles PCAP ingestion and time-series preprocessing.
- `models.py`: Adapter classes for Prophet and DeepAR models.
- `main.py`: Orchestrator for the data loading, training, and evaluation pipeline.
- `config.py`: Centralized configuration for simulation and model parameters.

## 🛠️ Usage

### 1. Traffic Generation & Capture

Use the provided shell script to spin up the Mininet environment, generate traffic, and capture the output.

```bash
# Syntax: ./vm_traffic_pipeline.sh <duration> <pattern> <output_file> <topology>
sudo ./vm_traffic_pipeline.sh 180 daily ./traffic_capture.pcap tree
```

### 2. Interactive Demo (No Mininet Required)

For a quick start without setting up the Mininet VM, use the Jupyter Notebook. It simulates 15 days of traffic to benchmark DeepAR vs Prophet.

1. Open `demo_multi_day.ipynb` in VS Code.
2. Run all cells to generate synthetic data and train models.
3. View the publication-ready plots inline.

### 3. Training & Evaluation (Full Pipeline)

Run the main pipeline to process the captured data and train the models.

```bash
python3 main.py
```

## 📊 Outputs & Artifacts

All generated files are stored in the `output` directory:

| Category | File | Description |
|----------|------|-------------|
| **Data** | `output/processed_original.csv` | Raw time-series data from capture. |
| | `output/processed_augmented.csv` | Augmented training data with noise. |
| **Visuals** | `output/dataset_overview.png` | Comparison of original vs. augmented data. |
| | `output/model_comparison.png` | Combined forecast plot (Main Pipeline). |
| | `output/demo_results.png` | Multi-day simulation forecast (Interactive Demo). |
| **Prophet** | `output/plot_prophet.png` | Forecast visualization for Prophet. |
| | `output/metrics_prophet.txt` | MSE and MAE metrics. |
| **DeepAR** | `output/plot_deepar.png` | Forecast visualization for DeepAR. |
| | `output/metrics_deepar.txt` | MSE and MAE metrics. |

## 📈 Model Comparison

The interactive demo generates a professional publication-ready plot comparing the actual traffic versus the predicted values.

![Model Comparison](output/demo_results.png)

## ⚙️ Configuration

Modify `config.py` to adjust:

- **Preprocessing**: Bin size (`DT`), augmentation factors, and noise levels.
- **Training**: Epochs, context length, and test split ratios.
- **Paths**: Input/Output directories.
