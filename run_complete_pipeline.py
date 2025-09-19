# run_complete_pipeline.py
import pandas as pd
import numpy as np
from cancer_ml_pipeline import CancerPredictionPipeline
from generate_sample_data import generate_cancer_dataset, save_dataset
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🫁 Cancer Prediction System - Complete Pipeline")
    print("=" * 50)
    
    # Step 1: Generate sample data (if you don't have real data)
    print("\n📊 Step 1: Generating sample dataset...")
    df = generate_cancer_dataset(5000)
    df = save_dataset(df, 'cancer_dataset.csv')
    
    # Step 2: Initialize pipeline
    print("\n🔧 Step 2: Initializing ML Pipeline...")
    pipeline = CancerPredictionPipeline()
    
    # Step 3: Preprocess data
    print("\n🛠️ Step 3: Preprocessing data...")
    df_processed = pipeline.preprocess_data(df.copy())
    
    # Step 4: Prepare features and target
    print("\n🎯 Step 4: Preparing features and target...")
    X = df_processed.drop('Final_Prediction', axis=1)
    y = df_processed['Final_Prediction']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Step 5: Train models
    print("\n🚀 Step 5: Training models...")
    model_scores, X_test, y_test = pipeline.train_models(X, y)
    
    # Step 6: Display results
    print("\n📈 Step 6: Model Performance Results")
    print("-" * 40)
    for model_name, score in model_scores.items():
        emoji = "🏆" if model_name == pipeline.best_model_name else "📊"
        print(f"{emoji} {model_name}: {score:.4f} ({score:.2%})")
    
    print(f"\n🎉 Best Model: {pipeline.best_model_name}")
    print(f"🔥 Best Accuracy: {pipeline.best_accuracy:.4f} ({pipeline.best_accuracy:.2%})")
    
    # Step 7: Save models
    print("\n💾 Step 7: Saving trained models...")
    pipeline.save_models()
    
    # Step 8: Test prediction
    print("\n🔮 Step 8: Testing prediction on sample data...")
    
    # Create a sample input for testing
    sample_input = pd.DataFrame({
        'Country': ['USA'],
        'Age': [55],
        'Gender': ['Male'],
        'Smoking_Status': ['Current'],
        'Second_Hand_Smoke': ['Yes'],
        'Air_Pollution_Exposure': [8.5],
        'Occupation_Exposure': ['High'],
        'Rural_or_Urban': ['Urban'],
        'Socioeconomic_Status': ['Medium'],
        'Healthcare_Access': ['Good'],
        'Insurance_Coverage': ['Partial'],
        'Screening_Availability': ['Yes'],
        'Stage_at_Diagnosis': [2],
        'Cancer_Type': ['Lung'],
        'Mutation_Type': ['EGFR'],
        'Treatment_Access': ['Medium'],
        'Clinical_Trial_Access': ['No'],
        'Language_Barrier': ['No'],
        'Mortality_Risk': [0.6],
        '5_Year_Survival_Probability': [0.7],
        'Delay_in_Diagnosis': [45],
        'Family_History': ['Yes'],
        'Indoor_Smoke_Exposure': ['Yes'],
        'Tobacco_Marketing_Exposure': ['High']
    })
    
    try:
        prediction, probability = pipeline.predict(sample_input)
        print(f"Sample Prediction: {'Cancer Risk' if prediction[0] == 1 else 'No Cancer Risk'}")
        print(f"Probability: {probability[0][1]:.2%} cancer risk")
    except Exception as e:
        print(f"Prediction test failed: {e}")
        print("This is normal if you're using the sample data generator")
    
    print("\n✅ Pipeline completed successfully!")
    print("\n🚀 Now you can run the Streamlit app with:")
    print("streamlit run streamlit_app.py")
    
    # Feature importance analysis
    if hasattr(pipeline.best_model, 'feature_importances_'):
        print("\n🔍 Top 10 Most Important Features:")
        feature_importance = zip(pipeline.feature_names, pipeline.best_model.feature_importances_)
        sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:<25} {importance:.4f}")
    
    return pipeline, df_processed, model_scores

if __name__ == "__main__":
    pipeline, df_processed, model_scores = main()