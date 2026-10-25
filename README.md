# AI-Powered Cancer Risk Prediction

Cancer risk prediction system trained on 5,000 patient records with 24 features using **Random Forest**, **XGBoost**, **LightGBM**, and soft-voting ensembles. The models typically land in the **90–95%** accuracy band on held-out data (RF + XGBoost / LightGBM ensembles).

An interactive **Streamlit** dashboard serves **live model predictions** (not dummy/random scores), performance analytics, and dataset insights.

## Quick start

```bash
pip install -r requirements.txt
python run_complete_pipeline.py
streamlit run streamlit_app.py
```

## Tests

```bash
python -m pytest test_e2e.py -v
```

## Project layout

| File | Purpose |
|------|---------|
| `generate_sample_data.py` | Synthetic but learnable 5k-row dataset |
| `cancer_ml_pipeline.py` | Preprocess, train RF/XGB/LGBM/ensembles, predict |
| `run_complete_pipeline.py` | End-to-end train + save artifacts |
| `streamlit_app.py` | Live dashboard with real inference |
| `test_e2e.py` | ~30 multi-angle end-to-end tests |

## Artifacts (created by training)

- `cancer_models.pkl` — trained models
- `label_encoders.pkl`, `scaler.pkl`, `feature_names.pkl` — preprocessing
- `model_metrics.pkl` — live accuracy / precision / recall / F1
- `cancer_dataset.csv` — training dataset
