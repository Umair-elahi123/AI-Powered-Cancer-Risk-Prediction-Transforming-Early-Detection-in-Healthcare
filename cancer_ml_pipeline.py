# cancer_ml_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

class CancerPredictionPipeline:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_accuracy = 0
        self.feature_names = []
        
    def preprocess_data(self, df):
        """Comprehensive data preprocessing"""
        print("Starting data preprocessing...")
        
        # Handle missing values
        df = df.fillna(df.mode().iloc[0])
        
        # Separate categorical and numerical columns
        categorical_cols = ['Country', 'Gender', 'Smoking_Status', 'Second_Hand_Smoke',
                          'Occupation_Exposure', 'Rural_or_Urban', 'Socioeconomic_Status',
                          'Healthcare_Access', 'Insurance_Coverage', 'Screening_Availability',
                          'Cancer_Type', 'Mutation_Type', 'Treatment_Access', 
                          'Clinical_Trial_Access', 'Language_Barrier', 'Family_History',
                          'Indoor_Smoke_Exposure', 'Tobacco_Marketing_Exposure']
        
        numerical_cols = ['Age', 'Air_Pollution_Exposure', 'Stage_at_Diagnosis',
                         'Mortality_Risk', '5_Year_Survival_Probability', 'Delay_in_Diagnosis']
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale numerical features
        if numerical_cols:
            existing_num_cols = [col for col in numerical_cols if col in df.columns]
            if existing_num_cols:
                df[existing_num_cols] = self.scaler.fit_transform(df[existing_num_cols])
        
        self.feature_names = [col for col in df.columns if col != 'Final_Prediction']
        
        return df
    
    def train_models(self, X, y):
        """Train multiple models and ensembles"""
        print("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Individual Models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method='gpu_hist',  # GPU acceleration
                gpu_id=0
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                device='gpu',  # GPU acceleration
                gpu_platform_id=0,
                gpu_device_id=0
            )
        }
        
        # Train individual models and track performance
        model_scores = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_scores[name] = accuracy
            self.models[name] = model
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Ensemble Models
        print("Training ensemble models...")
        
        # Random Forest + XGBoost Ensemble
        rf_xgb_ensemble = VotingClassifier([
            ('rf', models['Random Forest']),
            ('xgb', models['XGBoost'])
        ], voting='soft')
        rf_xgb_ensemble.fit(X_train, y_train)
        rf_xgb_pred = rf_xgb_ensemble.predict(X_test)
        rf_xgb_accuracy = accuracy_score(y_test, rf_xgb_pred)
        model_scores['RF + XGBoost Ensemble'] = rf_xgb_accuracy
        self.models['RF + XGBoost Ensemble'] = rf_xgb_ensemble
        
        # Random Forest + LightGBM Ensemble
        rf_lgb_ensemble = VotingClassifier([
            ('rf', models['Random Forest']),
            ('lgb', models['LightGBM'])
        ], voting='soft')
        rf_lgb_ensemble.fit(X_train, y_train)
        rf_lgb_pred = rf_lgb_ensemble.predict(X_test)
        rf_lgb_accuracy = accuracy_score(y_test, rf_lgb_pred)
        model_scores['RF + LightGBM Ensemble'] = rf_lgb_accuracy
        self.models['RF + LightGBM Ensemble'] = rf_lgb_ensemble
        
        # Find best model
        self.best_model_name = max(model_scores, key=model_scores.get)
        self.best_accuracy = model_scores[self.best_model_name]
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nBest Model: {self.best_model_name} with accuracy: {self.best_accuracy:.4f}")
        
        return model_scores, X_test, y_test
    
    def save_models(self):
        """Save trained models and preprocessors"""
        joblib.dump(self.models, 'cancer_models.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.feature_names, 'feature_names.pkl')
        print("Models saved successfully!")
    
    def predict(self, input_data):
        """Make predictions on new data"""
        # Preprocess input data
        for col, encoder in self.label_encoders.items():
            if col in input_data.columns:
                input_data[col] = encoder.transform(input_data[col].astype(str))
        
        # Scale numerical features
        numerical_cols = ['Age', 'Air_Pollution_Exposure', 'Stage_at_Diagnosis',
                         'Mortality_Risk', '5_Year_Survival_Probability', 'Delay_in_Diagnosis']
        existing_num_cols = [col for col in numerical_cols if col in input_data.columns]
        if existing_num_cols:
            input_data[existing_num_cols] = self.scaler.transform(input_data[existing_num_cols])
        
        # Make prediction
        prediction = self.best_model.predict(input_data)
        probability = self.best_model.predict_proba(input_data)
        
        return prediction, probability

# Usage Example
if __name__ == "__main__":
    # Load your dataset
    # df = pd.read_csv('cancer_dataset.csv')
    
    # Initialize pipeline
    pipeline = CancerPredictionPipeline()
    
    # Preprocess data
    # df_processed = pipeline.preprocess_data(df)
    
    # Prepare features and target
    # X = df_processed.drop('Final_Prediction', axis=1)
    # y = df_processed['Final_Prediction']
    
    # Train models
    # model_scores, X_test, y_test = pipeline.train_models(X, y)
    
    # Save models
    # pipeline.save_models()
    
    print("Pipeline setup complete!")