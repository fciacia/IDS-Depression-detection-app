from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def label_encoder(df_model):
    print("Dataset shape:", df_model.shape)
    print("Columns:", df_model.columns)
    print(df_model.head())

    label_encoders = {}
    for col in df_model.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    X = df_model.drop('depression', axis=1)
    y = df_model['depression']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the encoders and scaler
    joblib.dump(label_encoders, 'label_encoders.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_label_encoders():
    """Load the label encoders and scaler from disk"""
    try:
        label_encoders = joblib.load('label_encoders.joblib')
        scaler = joblib.load('scaler.joblib')
        return {'encoders': label_encoders, 'scaler': scaler}
    except Exception as e:
        print(f"Error loading encoders: {str(e)}")
        return None