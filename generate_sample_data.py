# generate_sample_data.py
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def generate_cancer_dataset(n_samples=5000, random_state=42):
    """Generate a balanced, learnable cancer risk dataset (~30% positive)."""
    rng = np.random.default_rng(random_state)

    countries = [
        "USA", "UK", "Germany", "France", "Canada", "Australia", "Japan",
        "India", "Brazil", "China", "Italy", "Spain", "Netherlands",
        "Sweden", "Norway", "Denmark", "Finland",
    ]
    genders = ["Male", "Female"]
    smoking_statuses = ["Never", "Former", "Current"]
    binary_options = ["No", "Yes"]
    levels = ["Low", "Medium", "High"]
    healthcare_levels = ["Poor", "Fair", "Good", "Excellent"]
    cancer_types = [
        "Lung", "Breast", "Colorectal", "Prostate", "Liver",
        "Stomach", "Cervical", "Bladder", "Kidney", "Other",
    ]
    mutation_types = [
        "EGFR", "KRAS", "ALK", "ROS1", "BRAF",
        "PIK3CA", "TP53", "BRCA1", "BRCA2", "No Mutation",
    ]

    age = np.clip(rng.normal(58, 12, n_samples).astype(int), 18, 95)
    smoking = rng.choice(smoking_statuses, n_samples, p=[0.4, 0.35, 0.25])
    family = rng.choice(binary_options, n_samples, p=[0.7, 0.3])
    air = rng.uniform(0, 10, n_samples).round(2)
    stage = rng.integers(1, 5, n_samples)
    second_hand = rng.choice(binary_options, n_samples, p=[0.6, 0.4])
    occupation = rng.choice(levels, n_samples, p=[0.6, 0.3, 0.1])
    indoor = rng.choice(binary_options, n_samples, p=[0.6, 0.4])
    delay = rng.exponential(30, n_samples).astype(int)

    # Learnable risk signal tuned for ~35-45% positives and ~90-95% accuracy
    # (no direct leakage into mortality/survival features)
    logit = (
        -3.15
        + (age > 55).astype(float) * 0.75
        + (age > 65).astype(float) * 0.9
        + (age > 75).astype(float) * 0.6
        + (smoking == "Current").astype(float) * 1.7
        + (smoking == "Former").astype(float) * 0.75
        + (family == "Yes").astype(float) * 1.15
        + (air / 10.0) * 1.25
        + (stage >= 3).astype(float) * 1.3
        + (second_hand == "Yes").astype(float) * 0.55
        + (occupation == "High").astype(float) * 0.7
        + (occupation == "Medium").astype(float) * 0.28
        + (indoor == "Yes").astype(float) * 0.5
        + (delay > 45).astype(float) * 0.5
        + rng.normal(0, 0.38, n_samples)
    )
    probability = 1 / (1 + np.exp(-logit))
    final_prediction = (probability > 0.5).astype(int)

    # Clinical score fields: weakly related only (avoid label leakage)
    mortality = np.clip(
        0.05 * (stage - 1)
        + 0.04 * (smoking == "Current").astype(float)
        + rng.uniform(0.05, 0.55, n_samples),
        0,
        1,
    ).round(3)
    survival = np.clip(
        0.75
        - 0.05 * (stage - 1)
        + rng.uniform(-0.15, 0.2, n_samples),
        0,
        1,
    ).round(3)

    data = {
        "Country": rng.choice(countries, n_samples),
        "Age": age,
        "Gender": rng.choice(genders, n_samples),
        "Smoking_Status": smoking,
        "Second_Hand_Smoke": second_hand,
        "Air_Pollution_Exposure": air,
        "Occupation_Exposure": occupation,
        "Rural_or_Urban": rng.choice(["Rural", "Urban"], n_samples, p=[0.3, 0.7]),
        "Socioeconomic_Status": rng.choice(levels, n_samples, p=[0.3, 0.5, 0.2]),
        "Healthcare_Access": rng.choice(healthcare_levels, n_samples, p=[0.15, 0.25, 0.35, 0.25]),
        "Insurance_Coverage": rng.choice(["Uninsured", "Partial", "Full"], n_samples, p=[0.2, 0.3, 0.5]),
        "Screening_Availability": rng.choice(binary_options, n_samples, p=[0.3, 0.7]),
        "Stage_at_Diagnosis": stage,
        "Cancer_Type": rng.choice(cancer_types, n_samples),
        "Mutation_Type": rng.choice(mutation_types, n_samples),
        "Treatment_Access": rng.choice(levels, n_samples, p=[0.2, 0.5, 0.3]),
        "Clinical_Trial_Access": rng.choice(binary_options, n_samples, p=[0.7, 0.3]),
        "Language_Barrier": rng.choice(binary_options, n_samples, p=[0.8, 0.2]),
        "Mortality_Risk": mortality,
        "5_Year_Survival_Probability": survival,
        "Delay_in_Diagnosis": delay,
        "Family_History": family,
        "Indoor_Smoke_Exposure": indoor,
        "Tobacco_Marketing_Exposure": rng.choice(levels, n_samples, p=[0.4, 0.4, 0.2]),
        "Final_Prediction": final_prediction,
    }

    return pd.DataFrame(data)


def save_dataset(df, filename="cancer_dataset.csv"):
    """Save the dataset to CSV."""
    df.to_csv(filename, index=False)
    print(f"Dataset saved as {filename}")
    print(f"Shape: {df.shape}")
    print(
        f"Cancer cases: {df['Final_Prediction'].sum()} "
        f"({df['Final_Prediction'].mean():.2%})"
    )
    return df


if __name__ == "__main__":
    df = generate_cancer_dataset(5000)
    df = save_dataset(df)
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
