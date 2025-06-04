import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataPreprocessor:
    """Class to handle data preprocessing for depression analysis."""
    
    def __init__(self):
        self.valid_ranges = {
            'academic_pressure': (1, 5),
            'study_satisfaction': (1, 5),
            'cgpa': (0, 10),
            'work_study_hours': (0, 24),
            'financial_stress': (1, 5)
        }
        
        self.categorical_mappings = {
            'dietary_habits': {
                'Unhealthy': 1,
                'Moderate': 2,
                'Healthy': 3
            },
            'sleep_duration': {
                'Less than 5 hours': 1,
                '5-6 hours': 2,
                '7-8 hours': 3,
                'More than 8 hours': 4
            },
            'family_mental_history': {'No': 0, 'Yes': 1},
            'suicidal_thoughts': {'No': 0, 'Yes': 1},
            'gender': {'Female': 0, 'Male': 1}
        }
        
        self.degree_mapping = {
            'Pre-U': 0,
            'Class 12': 0,  # Mapped to Pre-U
            'Diploma': 1,
            'Undergraduate': 2,
            'Postgraduate': 3,
            'PhD': 4
        }
        
        self.columns_to_drop = ['profession', 'work_pressure', 'job_satisfaction', 'city']
        self.label_encoder = LabelEncoder()
        
    def validate_numeric_range(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Validate and clean numeric values in a column."""
        if column not in self.valid_ranges:
            return df[column]
            
        min_val, max_val = self.valid_ranges[column]
        series = df[column].copy()
        median_val = series[(series >= min_val) & (series <= max_val)].median()
        
        invalid_mask = (series < min_val) | (series > max_val)
        if invalid_mask.any():
            logging.warning(f"{invalid_mask.sum()} invalid values found in {column}. Replacing with median: {median_val}")
            series[invalid_mask] = median_val
        
        return series
    
    def process_categorical(self, df: pd.DataFrame, column: str, mapping: Dict) -> pd.Series:
        """Process categorical variables with proper error handling."""
        series = df[column].copy()
        unknown_categories = series[~series.isin(mapping.keys())].unique()
        
        if len(unknown_categories) > 0:
            logging.warning(f"Unknown categories in {column}: {unknown_categories}")
            default_value = max(mapping.values()) + 1
            return series.map(lambda x: mapping.get(x, default_value))
        
        return series.map(mapping)
    
    def process_cgpa(self, cgpa_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Process CGPA values and create bins."""
        cgpa_scaled = (cgpa_series * 0.4).round(1)  # Convert to 4.0 scale
        bin_edges = np.arange(0.0, 4.01, 0.1)
        cgpa_bin = pd.cut(cgpa_scaled, bins=bin_edges, labels=False, include_lowest=True)
        return cgpa_scaled, cgpa_bin
    
    def clean_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean and transform the depression dataset.
        
        Args:
            df: Input DataFrame containing depression-related data
            
        Returns:
            Tuple containing:
            - Cleaned DataFrame
            - Dictionary of transformation metadata
        """
        logging.info("Starting data preprocessing")
        metadata = {
            'original_shape': df.shape,
            'transformations': [],
            'warnings': [],
            'dropped_columns': []
        }
        
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # 1. Validate and clean numeric columns
            for column in self.valid_ranges.keys():
                if column in df.columns:
                    df[column] = self.validate_numeric_range(df, column)
                    metadata['transformations'].append(f"Validated {column}")
            
            # 2. Process categorical variables
            for column, mapping in self.categorical_mappings.items():
                if column in df.columns:
                    df[column] = self.process_categorical(df, column, mapping)
                    metadata['transformations'].append(f"Encoded {column}")
            
            # 3. Special handling for degree
            if 'degree' in df.columns:
                df['degree'] = self.process_categorical(df, 'degree', self.degree_mapping)
                metadata['transformations'].append("Encoded degree")
            
            # 4. Process CGPA
            if 'cgpa' in df.columns:
                df['cgpa_scaled'], df['cgpa_bin'] = self.process_cgpa(df['cgpa'])
                metadata['transformations'].append("Processed CGPA")
            
            # 5. Drop unnecessary columns
            columns_to_drop = [col for col in self.columns_to_drop if col in df.columns]
            if columns_to_drop:
                df = df.drop(columns_to_drop, axis=1)
                metadata['dropped_columns'] = columns_to_drop
            
            # 6. Handle missing values
            missing_data = df.isnull().sum()
            columns_with_missing = missing_data[missing_data > 0]
            
            if not columns_with_missing.empty:
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                metadata['transformations'].append("Filled missing values")
            
            # 7. Final validation
            if df.isnull().any().any():
                raise ValueError("NaN values remain after preprocessing")
            
            # 8. Convert all columns to numeric
            for col in df.columns:
                if col != 'depression':  # Don't convert target variable
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True)
            
            metadata['final_shape'] = df.shape
            logging.info("Preprocessing completed successfully")
            
            return df, metadata
            
        except Exception as e:
            logging.error(f"Error during preprocessing: {str(e)}")
            raise
    
    def get_feature_names(self) -> list:
        """Get list of feature names after preprocessing."""
        return [
            'gender', 'academic_pressure', 'study_satisfaction', 'sleep_duration',
            'dietary_habits', 'degree', 'suicidal_thoughts', 'work_study_hours',
            'financial_stress', 'family_mental_history', 'age_bin', 'cgpa_scaled',
            'cgpa_bin', 'depression'
        ]