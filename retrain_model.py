import pandas as pd
import joblib
from data_preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your cleaned data
csv_path = 'Clean Student Depression Dataset.csv'
df = pd.read_csv(csv_path)

# Preprocess
data_preprocessor = DataPreprocessor()
df_cleaned, _ = data_preprocessor.clean_transform(df)

# Remove cgpa_bin from features
df_cleaned = df_cleaned.drop('cgpa_bin', axis=1)

# Prepare features and target
X = df_cleaned.drop('depression', axis=1)
y = df_cleaned['depression']

# Save feature names (excluding target)
with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(X.columns))

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'trained_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print('Model, scaler, and feature_names.txt saved and synchronized!') 