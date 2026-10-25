"""
End-to-end multi-angle tests (~30) for the cancer risk prediction system.
Run: python -m pytest test_e2e.py -v
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from cancer_ml_pipeline import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    CancerPredictionPipeline,
)
from generate_sample_data import generate_cancer_dataset, save_dataset

ARTIFACTS = [
    "cancer_models.pkl",
    "label_encoders.pkl",
    "scaler.pkl",
    "feature_names.pkl",
    "model_metrics.pkl",
    "cancer_dataset.csv",
]

REQUIRED_MODELS = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "RF + XGBoost Ensemble",
    "RF + LightGBM Ensemble",
]


def _sample_patient(**overrides):
    base = {
        "Country": "USA",
        "Age": 55,
        "Gender": "Male",
        "Smoking_Status": "Current",
        "Second_Hand_Smoke": "Yes",
        "Air_Pollution_Exposure": 8.5,
        "Occupation_Exposure": "High",
        "Rural_or_Urban": "Urban",
        "Socioeconomic_Status": "Medium",
        "Healthcare_Access": "Good",
        "Insurance_Coverage": "Partial",
        "Screening_Availability": "Yes",
        "Stage_at_Diagnosis": 2,
        "Cancer_Type": "Lung",
        "Mutation_Type": "No Mutation",
        "Treatment_Access": "Medium",
        "Clinical_Trial_Access": "No",
        "Language_Barrier": "No",
        "Mortality_Risk": 0.6,
        "5_Year_Survival_Probability": 0.7,
        "Delay_in_Diagnosis": 45,
        "Family_History": "Yes",
        "Indoor_Smoke_Exposure": "Yes",
        "Tobacco_Marketing_Exposure": "High",
    }
    base.update(overrides)
    return pd.DataFrame([base])


@pytest.fixture(scope="module")
def pipeline():
    for path in ARTIFACTS:
        if not Path(path).exists():
            pytest.skip(f"Missing artifact {path}; run run_complete_pipeline.py first")
    return CancerPredictionPipeline().load_artifacts()


@pytest.fixture(scope="module")
def dataset():
    path = Path("cancer_dataset.csv")
    if not path.exists():
        pytest.skip("cancer_dataset.csv missing")
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


# ---- Data generation / dataset (1-6) ----

def test_01_generated_dataset_shape_and_columns():
    df = generate_cancer_dataset(200, random_state=7)
    assert df.shape[0] == 200
    assert "Final_Prediction" in df.columns
    for col in CATEGORICAL_COLS + NUMERICAL_COLS:
        assert col in df.columns


def test_02_target_is_binary(dataset):
    assert set(dataset["Final_Prediction"].unique()).issubset({0, 1})


def test_03_class_balance_not_degenerate(dataset):
    rate = dataset["Final_Prediction"].mean()
    assert 0.20 <= rate <= 0.80, f"Unusable class balance: {rate:.2%}"


def test_04_age_bounds(dataset):
    assert dataset["Age"].min() >= 18
    assert dataset["Age"].max() <= 100


def test_05_no_nulls_in_core_columns(dataset):
    cols = CATEGORICAL_COLS + NUMERICAL_COLS + ["Final_Prediction"]
    assert dataset[cols].isna().sum().sum() == 0


def test_06_save_dataset_roundtrip(tmp_path):
    df = generate_cancer_dataset(50, random_state=1)
    out = tmp_path / "tmp.csv"
    save_dataset(df, str(out))
    loaded = pd.read_csv(out)
    assert loaded.shape == df.shape


# ---- Artifacts / loading (7-11) ----

def test_07_all_artifacts_exist():
    for path in ARTIFACTS:
        assert Path(path).exists(), f"Missing {path}"


def test_08_models_dict_has_all_required(pipeline):
    for name in REQUIRED_MODELS:
        assert name in pipeline.models


def test_09_feature_names_match_training_schema(pipeline):
    assert len(pipeline.feature_names) == 24
    assert "Final_Prediction" not in pipeline.feature_names


def test_10_metrics_file_consistent(pipeline):
    metrics = joblib.load("model_metrics.pkl")
    assert metrics["best_model_name"] == pipeline.best_model_name
    assert abs(metrics["best_accuracy"] - pipeline.best_accuracy) < 1e-9
    for name in REQUIRED_MODELS:
        assert name in metrics["model_scores"]


def test_11_encoders_cover_categoricals(pipeline):
    for col in CATEGORICAL_COLS:
        assert col in pipeline.label_encoders
        assert len(pipeline.label_encoders[col].classes_) >= 2


# ---- Model quality (12-17) ----

def test_12_all_models_accuracy_above_floor(pipeline):
    for name, score in pipeline.model_scores.items():
        assert score >= 0.80, f"{name} accuracy too low: {score}"


def test_13_best_model_in_90_95_band_or_better(pipeline):
    assert pipeline.best_accuracy >= 0.90, (
        f"Best accuracy {pipeline.best_accuracy:.2%} below 90%"
    )


def test_14_rf_xgb_ensemble_present_and_strong(pipeline):
    score = pipeline.model_scores["RF + XGBoost Ensemble"]
    assert score >= 0.88


def test_15_individual_boosters_trained(pipeline):
    assert pipeline.model_scores["XGBoost"] >= 0.85
    assert pipeline.model_scores["LightGBM"] >= 0.85
    assert pipeline.model_scores["Random Forest"] >= 0.85


def test_16_metrics_include_precision_recall_f1(pipeline):
    for name in REQUIRED_MODELS:
        m = pipeline.model_metrics[name]
        for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
            assert key in m


def test_17_confusion_matrix_shape(pipeline):
    cm = np.array(pipeline.model_metrics[pipeline.best_model_name]["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert cm.sum() > 0


# ---- Real prediction behavior (18-26) ----

def test_18_predict_returns_binary_and_proba(pipeline):
    pred, proba = pipeline.predict(_sample_patient())
    assert pred.shape == (1,)
    assert pred[0] in (0, 1)
    assert proba.shape == (1, 2)
    assert abs(proba.sum(axis=1)[0] - 1.0) < 1e-5


def test_19_prediction_is_deterministic(pipeline):
    row = _sample_patient()
    p1, pr1 = pipeline.predict(row)
    p2, pr2 = pipeline.predict(row)
    assert p1[0] == p2[0]
    np.testing.assert_allclose(pr1, pr2)


def test_20_not_dummy_random_prediction(pipeline):
    """Repeated identical inputs must not behave like np.random."""
    row = _sample_patient(Age=40, Smoking_Status="Never", Family_History="No")
    probs = [float(pipeline.predict(row)[1][0][1]) for _ in range(8)]
    assert len(set(round(p, 8) for p in probs)) == 1


def test_21_high_risk_profile_elevated_probability(pipeline):
    high = _sample_patient(
        Age=78,
        Smoking_Status="Current",
        Family_History="Yes",
        Air_Pollution_Exposure=9.5,
        Stage_at_Diagnosis=4,
        Second_Hand_Smoke="Yes",
        Occupation_Exposure="High",
        Indoor_Smoke_Exposure="Yes",
        Delay_in_Diagnosis=120,
    )
    _, proba = pipeline.predict(high)
    assert proba[0][1] >= 0.55


def test_22_low_risk_profile_lower_than_high(pipeline):
    high = _sample_patient(
        Age=78,
        Smoking_Status="Current",
        Family_History="Yes",
        Air_Pollution_Exposure=9.5,
        Stage_at_Diagnosis=4,
    )
    low = _sample_patient(
        Age=25,
        Smoking_Status="Never",
        Family_History="No",
        Air_Pollution_Exposure=1.0,
        Stage_at_Diagnosis=1,
        Second_Hand_Smoke="No",
        Occupation_Exposure="Low",
        Indoor_Smoke_Exposure="No",
        Delay_in_Diagnosis=3,
        Mortality_Risk=0.05,
    )
    _, ph = pipeline.predict(high)
    _, pl = pipeline.predict(low)
    assert pl[0][1] < ph[0][1]


def test_23_each_model_can_predict(pipeline):
    row = _sample_patient()
    for name in REQUIRED_MODELS:
        pred, proba = pipeline.predict(row, model_name=name)
        assert pred[0] in (0, 1)
        assert 0.0 <= proba[0][1] <= 1.0


def test_24_batch_prediction(pipeline):
    batch = pd.concat(
        [_sample_patient(Age=30), _sample_patient(Age=70), _sample_patient(Age=50)],
        ignore_index=True,
    )
    pred, proba = pipeline.predict(batch)
    assert len(pred) == 3
    assert proba.shape == (3, 2)


def test_25_unseen_category_does_not_crash(pipeline):
    row = _sample_patient(Country="Atlantis")
    pred, proba = pipeline.predict(row)
    assert pred[0] in (0, 1)
    assert proba.shape == (1, 2)


def test_26_transform_input_feature_order(pipeline):
    X = pipeline.transform_input(_sample_patient())
    assert list(X.columns) == pipeline.feature_names
    assert X.shape == (1, len(pipeline.feature_names))


# ---- Integration / app wiring (27-30) ----

def test_27_streamlit_app_imports():
    import streamlit_app  # noqa: F401

    assert hasattr(streamlit_app, "main")
    assert hasattr(streamlit_app, "load_pipeline")


def test_28_streamlit_has_no_random_dummy_prediction():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "np.random.uniform(0.1, 0.9)" not in source
    assert "pipeline.predict" in source


def test_29_pipeline_retrain_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = generate_cancer_dataset(400, random_state=99)
    pipe = CancerPredictionPipeline()
    processed = pipe.preprocess_data(df)
    X = processed.drop("Final_Prediction", axis=1)
    y = processed["Final_Prediction"]
    scores, _, _ = pipe.train_models(X, y)
    assert set(REQUIRED_MODELS).issubset(scores.keys())
    pipe.save_models()
    loaded = CancerPredictionPipeline().load_artifacts()
    pred, _ = loaded.predict(_sample_patient())
    assert pred[0] in (0, 1)


def test_30_end_to_end_dataset_to_prediction(pipeline, dataset):
    # Take a real row from CSV (raw categorical strings) and predict
    raw = dataset.drop(columns=["Final_Prediction"]).iloc[[0]].copy()
    pred, proba = pipeline.predict(raw)
    assert pred[0] in (0, 1)
    assert 0.0 <= float(proba[0][1]) <= 1.0
