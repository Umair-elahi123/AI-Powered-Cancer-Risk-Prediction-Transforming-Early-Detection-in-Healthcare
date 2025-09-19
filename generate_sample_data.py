# generate_sample_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def generate_cancer_dataset(n_samples=5000):
    """Generate a comprehensive cancer dataset for testing"""
    
    np.random.seed(42)
    
    # Define possible values for categorical variables
    countries = ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia', 'Japan', 'India', 'Brazil', 'China', 
                'Italy', 'Spain', 'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Finland']
    
    genders = ['Male', 'Female']
    smoking_statuses = ['Never', 'Former', 'Current']
    binary_options = ['No', 'Yes']
    levels = ['Low', 'Medium', 'High']
    healthcare_levels = ['Poor', 'Fair', 'Good', 'Excellent']
    cancer_types = ['Lung', 'Breast', 'Colorectal', 'Prostate', 'Liver', 'Stomach', 'Cervical', 'Bladder', 'Kidney', 'Other']
    mutation_types = ['EGFR', 'KRAS', 'ALK', 'ROS1', 'BRAF', 'PIK3CA', 'TP53', 'BRCA1', 'BRCA2', 'None']
    
    # Generate data
    data = {
        # Demographics
        'Country': np.random.choice(countries, n_samples),
        'Age': np.random.normal(58, 12, n_samples).astype(int),
        'Gender': np.random.choice(genders, n_samples),
        
        # Risk factors
        'Smoking_Status': np.random.choice(smoking_statuses, n_samples, p=[0.4, 0.35, 0.25]),
        'Second_Hand_Smoke': np.random.choice(binary_options, n_samples, p=[0.6, 0.4]),
        'Air_Pollution_Exposure': np.random.uniform(0, 10, n_samples).round(2),
        'Occupation_Exposure': np.random.choice(levels, n_samples, p=[0.6, 0.3, 0.1]),
        'Rural_or_Urban': np.random.choice(['Rural', 'Urban'], n_samples, p=[0.3, 0.7]),
        
        # Socioeconomic factors
        'Socioeconomic_Status': np.random.choice(levels, n_samples, p=[0.3, 0.5, 0.2]),
        'Healthcare_Access': np.random.choice(healthcare_levels, n_samples, p=[0.15, 0.25, 0.35, 0.25]),
        'Insurance_Coverage': np.random.choice(['None', 'Partial', 'Full'], n_samples, p=[0.2, 0.3, 0.5]),
        
        # Screening and diagnosis
        'Screening_Availability': np.random.choice(binary_options, n_samples, p=[0.3, 0.7]),
        'Stage_at_Diagnosis': np.random.randint(1, 5, n_samples),
        'Cancer_Type': np.random.choice(cancer_types, n_samples),
        'Mutation_Type': np.random.choice(mutation_types, n_samples),
        
        # Access to care
        'Treatment_Access': np.random.choice(levels, n_samples, p=[0.2, 0.5, 0.3]),
        'Clinical_Trial_Access': np.random.choice(binary_options, n_samples, p=[0.7, 0.3]),
        'Language_Barrier': np.random.choice(binary_options, n_samples, p=[0.8, 0.2]),
        
        # Risk assessments
        'Mortality_Risk': np.random.uniform(0, 1, n_samples).round(3),
        '5_Year_Survival_Probability': np.random.uniform(0, 1, n_samples).round(3),
        'Delay_in_Diagnosis': np.random.exponential(30, n_samples).astype(int),
        
        # Additional factors
        'Family_History': np.random.choice(binary_options, n_samples, p=[0.7, 0.3]),
        'Indoor_Smoke_Exposure': np.random.choice(binary_options, n_samples, p=[0.6, 0.4]),
        'Tobacco_Marketing_Exposure': np.random.choice(levels, n_samples, p=[0.4, 0.4, 0.2])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure age is within reasonable bounds
    df['Age'] = np.clip(df['Age'], 18, 95)
    
    # Create realistic correlations for Final_Prediction
    risk_score = (
        (df['Age'] > 60).astype(int) * 0.3 +
        (df['Smoking_Status'] == 'Current').astype(int) * 0.4 +
        (df['Smoking_Status'] == 'Former').astype(int) * 0.2 +
        (df['Family_History'] == 'Yes').astype(int) * 0.25 +
        (df['Air_Pollution_Exposure'] > 7).astype(int) * 0.15 +
        (df['Stage_at_Diagnosis'] >= 3).astype(int) * 0.35 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Convert to probability and then binary prediction
    probability = 1 / (1 + np.exp(-risk_score))  # Sigmoid function
    df['Final_Prediction'] = (probability > 0.5).astype(int)
    
    return df

def save_dataset(df, filename='cancer_dataset.csv'):
    """Save the dataset to CSV"""
    df.to_csv(filename, index=False)
    print(f"Dataset saved as {filename}")
    print(f"Shape: {df.shape}")
    print(f"Cancer cases: {df['Final_Prediction'].sum()} ({df['Final_Prediction'].mean():.2%})")
    
    return df

if __name__ == "__main__":
    # Generate and save dataset
    df = generate_cancer_dataset(5000)
    df = save_dataset(df)
    
    # Display basic info
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())