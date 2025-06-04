from typing import Optional, Union
import pandas as pd
import os

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate the structure and content of the loaded dataframe.
    
    Args:
        df: The pandas DataFrame to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    required_columns = {
        'gender': str,
        'age': int,
        'academic_pressure': int,
        'cgpa': float,
        'study_satisfaction': int,
        'sleep_duration': str,
        'dietary_habits': str,
        'degree': str,
        'suicidal_thoughts': str,
        'work_study_hours': float,
        'financial_stress': int,
        'family_mental_history': str,
        'depression': int
    }
    
    # Check if all required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {', '.join(missing_cols)}")
        return False
    
    # Check for empty dataframe
    if df.empty:
        print("DataFrame is empty")
        return False
    
    # Check for too many missing values
    missing_threshold = 0.5
    missing_percentages = df.isnull().mean()
    problematic_columns = missing_percentages[missing_percentages > missing_threshold].index
    if not problematic_columns.empty:
        print(f"Columns with too many missing values (>{missing_threshold*100}%): {', '.join(problematic_columns)}")
        return False
    
    return True

def load_data(raw_data_path: str) -> Optional[pd.DataFrame]:
    """
    Load data from a CSV file with comprehensive error handling and validation.
    
    Args:
        raw_data_path: Path to the CSV file
        
    Returns:
        Optional[pd.DataFrame]: Loaded and validated DataFrame or None if loading fails
    """
    try:
        # Check if file exists
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"File not found: {raw_data_path}")
        
        # Check file extension
        if not raw_data_path.lower().endswith('.csv'):
            raise ValueError("File must be a CSV file")
        
        # Load the data
        df = pd.read_csv(raw_data_path)
        
        # Validate the loaded data
        if not validate_dataframe(df):
            raise ValueError("Data validation failed")
        
        print(f"Data loaded successfully from: {raw_data_path}")
        print(f"Shape: {df.shape}")
        print("\nSample of loaded data:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        
        return df
    
    except pd.errors.EmptyDataError:
        print(f"Error: The file {raw_data_path} is empty")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse {raw_data_path}. Make sure it's a valid CSV file")
    except Exception as e:
        print(f"Unexpected error loading data: {str(e)}")
    
    return None

def save_data(df: pd.DataFrame, clean_data_path: str) -> bool:
    """
    Save DataFrame to CSV with error handling and validation.
    
    Args:
        df: DataFrame to save
        clean_data_path: Path where to save the CSV file
        
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("Cannot save empty DataFrame")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(clean_data_path)), exist_ok=True)
        
        # Save the data
        df.to_csv(clean_data_path, index=False)
        
        # Verify the save was successful
        if not os.path.exists(clean_data_path):
            raise IOError("File was not created successfully")
        
        # Verify file size is reasonable
        file_size = os.path.getsize(clean_data_path)
        if file_size == 0:
            raise ValueError("Saved file is empty")
        
        print(f"Data saved successfully to: {clean_data_path}")
        print(f"File size: {file_size/1024:.2f} KB")
        return True
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False