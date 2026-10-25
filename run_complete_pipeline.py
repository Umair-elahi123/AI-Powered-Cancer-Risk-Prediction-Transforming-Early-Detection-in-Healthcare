# run_complete_pipeline.py
import pandas as pd
from cancer_ml_pipeline import CancerPredictionPipeline
from generate_sample_data import generate_cancer_dataset, save_dataset
import warnings

warnings.filterwarnings("ignore")


def main():
    print("Cancer Prediction System - Complete Pipeline")
    print("=" * 50)

    print("\nStep 1: Generating sample dataset...")
    df = generate_cancer_dataset(5000)
    df = save_dataset(df, "cancer_dataset.csv")

    print("\nStep 2: Initializing ML Pipeline...")
    pipeline = CancerPredictionPipeline()

    print("\nStep 3: Preprocessing data...")
    df_processed = pipeline.preprocess_data(df.copy())

    print("\nStep 4: Preparing features and target...")
    X = df_processed.drop("Final_Prediction", axis=1)
    y = df_processed["Final_Prediction"]
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    print("\nStep 5: Training models...")
    model_scores, X_test, y_test = pipeline.train_models(X, y)

    print("\nStep 6: Model Performance Results")
    print("-" * 40)
    for model_name, score in model_scores.items():
        marker = "BEST" if model_name == pipeline.best_model_name else "    "
        print(f"[{marker}] {model_name}: {score:.4f} ({score:.2%})")

    print(f"\nBest Model: {pipeline.best_model_name}")
    print(f"Best Accuracy: {pipeline.best_accuracy:.4f} ({pipeline.best_accuracy:.2%})")

    print("\nStep 7: Saving trained models...")
    pipeline.save_models()

    print("\nStep 8: Testing prediction on sample data...")
    sample_input = pd.DataFrame(
        {
            "Country": ["USA"],
            "Age": [55],
            "Gender": ["Male"],
            "Smoking_Status": ["Current"],
            "Second_Hand_Smoke": ["Yes"],
            "Air_Pollution_Exposure": [8.5],
            "Occupation_Exposure": ["High"],
            "Rural_or_Urban": ["Urban"],
            "Socioeconomic_Status": ["Medium"],
            "Healthcare_Access": ["Good"],
            "Insurance_Coverage": ["Partial"],
            "Screening_Availability": ["Yes"],
            "Stage_at_Diagnosis": [2],
            "Cancer_Type": ["Lung"],
            "Mutation_Type": ["No Mutation"],
            "Treatment_Access": ["Medium"],
            "Clinical_Trial_Access": ["No"],
            "Language_Barrier": ["No"],
            "Mortality_Risk": [0.6],
            "5_Year_Survival_Probability": [0.7],
            "Delay_in_Diagnosis": [45],
            "Family_History": ["Yes"],
            "Indoor_Smoke_Exposure": ["Yes"],
            "Tobacco_Marketing_Exposure": ["High"],
        }
    )

    high_risk = sample_input.copy()
    high_risk.loc[0, "Age"] = 78
    high_risk.loc[0, "Smoking_Status"] = "Current"
    high_risk.loc[0, "Family_History"] = "Yes"
    high_risk.loc[0, "Air_Pollution_Exposure"] = 9.5
    high_risk.loc[0, "Stage_at_Diagnosis"] = 4
    high_risk.loc[0, "Second_Hand_Smoke"] = "Yes"
    high_risk.loc[0, "Occupation_Exposure"] = "High"
    high_risk.loc[0, "Indoor_Smoke_Exposure"] = "Yes"
    high_risk.loc[0, "Delay_in_Diagnosis"] = 120

    low_risk = sample_input.copy()
    low_risk.loc[0, "Age"] = 28
    low_risk.loc[0, "Smoking_Status"] = "Never"
    low_risk.loc[0, "Family_History"] = "No"
    low_risk.loc[0, "Air_Pollution_Exposure"] = 1.0
    low_risk.loc[0, "Stage_at_Diagnosis"] = 1
    low_risk.loc[0, "Second_Hand_Smoke"] = "No"
    low_risk.loc[0, "Occupation_Exposure"] = "Low"
    low_risk.loc[0, "Indoor_Smoke_Exposure"] = "No"
    low_risk.loc[0, "Delay_in_Diagnosis"] = 5

    high_pred, high_prob = pipeline.predict(high_risk)
    low_pred, low_prob = pipeline.predict(low_risk)
    print(
        f"High-risk profile: "
        f"{'Cancer Risk' if high_pred[0] == 1 else 'No Cancer Risk'} "
        f"({high_prob[0][1]:.2%})"
    )
    print(
        f"Low-risk profile: "
        f"{'Cancer Risk' if low_pred[0] == 1 else 'No Cancer Risk'} "
        f"({low_prob[0][1]:.2%})"
    )

    print("\nPipeline completed successfully!")
    print("\nRun the Streamlit app with:")
    print("streamlit run streamlit_app.py")

    rf = pipeline.models.get("Random Forest")
    if rf is not None and hasattr(rf, "feature_importances_"):
        print("\nTop 10 Most Important Features (Random Forest):")
        ranked = sorted(
            zip(pipeline.feature_names, rf.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        for i, (feature, importance) in enumerate(ranked[:10], 1):
            print(f"{i:2d}. {feature:<30} {importance:.4f}")

    return pipeline, df_processed, model_scores


if __name__ == "__main__":
    pipeline, df_processed, model_scores = main()
