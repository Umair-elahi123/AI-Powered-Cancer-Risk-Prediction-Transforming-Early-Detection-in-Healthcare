# cancer_ml_pipeline.py
import utils.session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings("ignore")

CATEGORICAL_COLS = [
    "Country",
    "Gender",
    "Smoking_Status",
    "Second_Hand_Smoke",
    "Occupation_Exposure",
    "Rural_or_Urban",
    "Socioeconomic_Status",
    "Healthcare_Access",
    "Insurance_Coverage",
    "Screening_Availability",
    "Cancer_Type",
    "Mutation_Type",
    "Treatment_Access",
    "Clinical_Trial_Access",
    "Language_Barrier",
    "Family_History",
    "Indoor_Smoke_Exposure",
    "Tobacco_Marketing_Exposure",
]

NUMERICAL_COLS = [
    "Age",
    "Air_Pollution_Exposure",
    "Stage_at_Diagnosis",
    "Mortality_Risk",
    "5_Year_Survival_Probability",
    "Delay_in_Diagnosis",
]

TARGET_COL = "Final_Prediction"


class CancerPredictionPipeline:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        self.feature_names = []
        self.model_scores = {}
        self.model_metrics = {}

    def preprocess_data(self, df):
        """Fit preprocessors and transform training data."""
        print("Starting data preprocessing...")
        df = df.copy()
        df = df.fillna(df.mode().iloc[0])

        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        existing_num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
        if existing_num_cols:
            df[existing_num_cols] = self.scaler.fit_transform(df[existing_num_cols])

        self.feature_names = [c for c in df.columns if c != TARGET_COL]
        return df

    def _safe_encode_value(self, encoder, value):
        value = str(value)
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        # Unseen category -> most frequent class (index 0 after LabelEncoder fit order)
        return 0

    def transform_input(self, input_data):
        """Transform raw patient rows using fitted encoders/scaler."""
        df = input_data.copy()

        for col in self.feature_names:
            if col not in df.columns:
                if col in NUMERICAL_COLS:
                    df[col] = 0.0
                else:
                    df[col] = "Unknown"

        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(
                    lambda v, enc=encoder: self._safe_encode_value(enc, v)
                )

        existing_num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
        if existing_num_cols:
            df[existing_num_cols] = self.scaler.transform(df[existing_num_cols])

        return df[self.feature_names]

    def _build_models(self):
        """Build RF, XGBoost, LightGBM with CPU-safe settings."""
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        xgb_kwargs = dict(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        try:
            xgb_model = xgb.XGBClassifier(**xgb_kwargs, tree_method="hist")
        except TypeError:
            xgb_model = xgb.XGBClassifier(**xgb_kwargs)

        lgb_kwargs = dict(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
        )
        try:
            lgb_model = lgb.LGBMClassifier(**lgb_kwargs, device="cpu")
        except TypeError:
            lgb_model = lgb.LGBMClassifier(**lgb_kwargs)

        return {
            "Random Forest": rf,
            "XGBoost": xgb_model,
            "LightGBM": lgb_model,
        }

    def train_models(self, X, y):
        """Train individual models and soft-voting ensembles."""
        print("Training models...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = self._build_models()
        model_scores = {}
        model_metrics = {}

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_scores[name] = float(accuracy)
            model_metrics[name] = {
                "accuracy": float(accuracy),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
            self.models[name] = model
            print(f"{name} Accuracy: {accuracy:.4f}")

        print("Training ensemble models...")
        ensembles = {
            "RF + XGBoost Ensemble": VotingClassifier(
                [("rf", models["Random Forest"]), ("xgb", models["XGBoost"])],
                voting="soft",
            ),
            "RF + LightGBM Ensemble": VotingClassifier(
                [("rf", models["Random Forest"]), ("lgb", models["LightGBM"])],
                voting="soft",
            ),
        }

        for name, ensemble in ensembles.items():
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_scores[name] = float(accuracy)
            model_metrics[name] = {
                "accuracy": float(accuracy),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
            self.models[name] = ensemble
            print(f"{name} Accuracy: {accuracy:.4f}")

        self.model_scores = model_scores
        self.model_metrics = model_metrics
        self.best_model_name = max(model_scores, key=model_scores.get)
        self.best_accuracy = model_scores[self.best_model_name]
        self.best_model = self.models[self.best_model_name]

        print(
            f"\nBest Model: {self.best_model_name} "
            f"with accuracy: {self.best_accuracy:.4f}"
        )
        return model_scores, X_test, y_test

    def save_models(self, prefix=""):
        """Persist models, preprocessors, and evaluation metrics."""
        joblib.dump(self.models, f"{prefix}cancer_models.pkl")
        joblib.dump(self.label_encoders, f"{prefix}label_encoders.pkl")
        joblib.dump(self.scaler, f"{prefix}scaler.pkl")
        joblib.dump(self.feature_names, f"{prefix}feature_names.pkl")
        joblib.dump(
            {
                "model_scores": self.model_scores,
                "model_metrics": self.model_metrics,
                "best_model_name": self.best_model_name,
                "best_accuracy": self.best_accuracy,
            },
            f"{prefix}model_metrics.pkl",
        )
        print("Models saved successfully!")

    def load_artifacts(self, prefix=""):
        """Load persisted artifacts into this pipeline instance."""
        self.models = joblib.load(f"{prefix}cancer_models.pkl")
        self.label_encoders = joblib.load(f"{prefix}label_encoders.pkl")
        self.scaler = joblib.load(f"{prefix}scaler.pkl")
        self.feature_names = joblib.load(f"{prefix}feature_names.pkl")
        metrics = joblib.load(f"{prefix}model_metrics.pkl")
        self.model_scores = metrics["model_scores"]
        self.model_metrics = metrics["model_metrics"]
        self.best_model_name = metrics["best_model_name"]
        self.best_accuracy = metrics["best_accuracy"]
        self.best_model = self.models[self.best_model_name]
        return self

    def predict(self, input_data, model_name=None):
        """Make real model predictions on raw (unencoded) input rows."""
        X = self.transform_input(input_data)
        model = self.models.get(model_name, self.best_model)
        if model is None:
            raise RuntimeError("No trained model available for prediction.")
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        return prediction, probability


if __name__ == "__main__":
    print("Pipeline setup complete!")
