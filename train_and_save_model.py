import pandas as pd
import numpy as np
import joblib
from data_preprocessing import clean_transform
from label_encoders_split_data_model import label_encoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_and_save():
    try:
        # Load and preprocess data
        print("Loading data...")
        df = pd.read_csv('Clean Student Depression Dataset.csv')
        
        # Add any missing columns expected by preprocessing
        if 'id' not in df.columns:
            df['id'] = range(len(df))
        
        print("Preprocessing data...")
        df_cleaned = clean_transform(df)
        
        # Prepare for training
        X = df_cleaned.drop('depression', axis=1)
        y = df_cleaned['depression']
        
        # Split the data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        print("Training model...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        print("\nModel Performance:")
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Testing Accuracy: {test_score:.4f}")
        
        # Save the model and scaler
        print("\nSaving model and scaler...")
        joblib.dump(model, 'trained_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        
        # Save feature names for reference
        with open('feature_names.txt', 'w') as f:
            f.write('\n'.join(X.columns))
        
        print("Model and scaler saved successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    train_and_save() 