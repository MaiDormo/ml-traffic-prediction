# ML Traffic Prediction

End-to-end pipeline for capturing Mininet traffic, preprocessing it, and comparing Prophet and GluonTS (DeepAR) forecasts. The raw capture (`exported.csv` by default) remains at the repository root, while all generated artifacts are organized under `artifacts/`.

## Workflow

1. **Capture traffic**
   - Use the Mininet + Ryu setup (`network_topology.py`, `sdn_controller.py`, `traffic_patterns.py`, `vm_traffic_pipeline.sh`) to produce `exported.csv`.
2. **Prophet preprocessing + training**
   - `python3 preprocess.py`
   - Performs FFT/autocorrelation period detection, augmentation, and Prophet training with uniform 2 s timestamps to stay aligned with GluonTS.
   - Outputs CSVs and plots under `artifacts/prophet/`:
     - `exported_prophet_input.csv`, `exported_prophet_insample.csv`, `exported_prophet_forecast.csv`
     - `prophet_forecast.png`, `prophet_components.png`, `prophet_fit_quality.png`, `traffic_plot.png`
3. **GluonTS DeepAR training**
   - `python3 gluton_preprocessor.py`
   - Consumes Prophet’s preprocessed data by default, enforces uniform timestamps, computes adaptive context/prediction windows, trains DeepAR, and exports diagnostics under `artifacts/gluonts/`:
     - `exported_gluonts_forecast.csv`, `exported_gluonts_metrics.csv`
     - `gluonts_forecast.png`, `gluonts_forecast_detail.png`
4. **(Optional) Model comparison**
   - `python3 compare_models.py` (if present) can ingest the artifacts above for side-by-side evaluation.

## Artifact Management

- `artifacts/` is ignored by git; feel free to keep multiple timestamped runs inside (e.g., `artifacts/2025-11-17/prophet/...`).
- To keep a clean working tree, move or delete large CSV/PNG outputs after collecting the relevant metrics.
- If you need to share results, copy the curated files out of `artifacts/` before committing.

## Next Steps / Ideas

- Expand `traffic_patterns.py` with additional generators (e.g., RandomBurstTraffic already included).
- Extend README with concrete comparison metrics once `compare_models.py` is finalized.
- Consider wiring `vm_traffic_pipeline.sh` so it runs the full pipeline (capture → preprocess → GluonTS → comparison) in one go.
