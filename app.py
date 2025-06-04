# depression_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page settings
st.set_page_config(
    page_title="Depression Detection App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #e3f2fd;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Card-like containers */
    .content-container {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        margin: 1rem 0;
        border-radius: 2rem;
        background: linear-gradient(45deg, #2196F3, #64B5F6);
        color: white;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        background: linear-gradient(45deg, #1976D2, #2196F3);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #81C784);
    }
    
    /* Alert boxes */
    .error-box {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: #ffebee;
        border: none;
        box-shadow: 0 2px 4px rgba(239, 83, 80, 0.2);
        margin: 1rem 0;
    }
    
    .success-box {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: #e8f5e9;
        border: none;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2);
        margin: 1rem 0;
    }
    
    .warning-box {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: #fff3e0;
        border: none;
        box-shadow: 0 2px 4px rgba(255, 152, 0, 0.2);
        margin: 1rem 0;
    }
    
    .info-message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 1rem;
        background-color: #e3f2fd;
        box-shadow: 0 2px 4px rgba(33, 150, 243, 0.2);
    }
    
    /* Titles and Headers */
    .main-title {
        text-align: center;
        margin-bottom: 2rem;
        color: #1565C0;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-title {
        color: #1976D2;
        font-size: 1.8rem;
        margin: 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #90CAF9;
    }
    
    /* Center content */
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    
    /* Form styling */
    .stSelectbox {
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 0.5rem;
        border: 2px solid #90CAF9;
    }
    
    .stSlider > div > div {
        background-color: #BBDEFB;
    }
    
    /* Markdown text */
    .markdown-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #37474F;
    }
    
    /* Custom list styling */
    .custom-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    .custom-list li {
        padding: 0.5rem 0;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .custom-list li:before {
        content: "•";
        color: #2196F3;
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Statistics Cards */
    .stats-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        flex: 1;
        min-width: 200px;
        border-left: 4px solid #2196F3;
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1565C0;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #546E7A;
        font-size: 1rem;
    }
    
    /* Story Cards */
    .story-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #FF9800;
    }
    
    .story-title {
        color: #FF9800;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .quote-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-style: italic;
    }
    
    .quote-text {
        color: #1565C0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .quote-author {
        color: #546E7A;
        font-size: 0.9rem;
        margin-top: 1rem;
        text-align: right;
    }
    
    /* Back Button */
    .back-button {
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 1000;
    }
    
    .back-button button {
        background: linear-gradient(45deg, #90CAF9, #64B5F6) !important;
        border-radius: 2rem !important;
        padding: 0.5rem 1.5rem !important;
        font-size: 1rem !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
        width: auto !important;
        margin: 0 !important;
    }
    
    .back-button button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
        background: linear-gradient(45deg, #64B5F6, #42A5F5) !important;
    }
    
    /* Main CTA Button */
    .main-cta {
        text-align: center;
        padding: 1rem;
        margin: 2rem auto;
        max-width: 500px;
    }
    
    .main-cta button {
        background: linear-gradient(45deg, #2196F3, #1976D2) !important;
        padding: 1rem 3rem !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        border-radius: 3rem !important;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3) !important;
        transition: all 0.3s ease !important;
        border: none !important;
        color: white !important;
        width: auto !important;
    }
    
    .main-cta button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4) !important;
        background: linear-gradient(45deg, #1976D2, #1565C0) !important;
    }
    
    .main-cta-description {
        color: #546E7A;
        font-size: 1.1rem;
        margin-top: 1rem;
        text-align: center;
    }
    
    /* Content Sections */
    .main-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .content-section {
        margin-bottom: 2rem;
    }
    
    .info-section, .process-section {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-section .section-title,
    .process-section .section-title {
        color: #1976D2;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #90CAF9;
    }
    
    .custom-list {
        list-style-type: none;
        padding-left: 0;
        margin: 1rem 0;
    }
    
    .custom-list li {
        padding: 0.5rem 0;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
        color: #37474F;
        font-size: 1.1rem;
    }
    
    .custom-list li:before {
        content: "•";
        color: #2196F3;
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    
    ol.custom-list {
        counter-reset: item;
    }
    
    ol.custom-list li {
        counter-increment: item;
    }
    
    ol.custom-list li:before {
        content: counter(item) ".";
        color: #2196F3;
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    
    /* Enhanced Card Animations */
    .content-container, .story-card, .stat-card, .quote-card {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .content-container:hover, .story-card:hover, .stat-card:hover, .quote-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(120deg, #1976D2, #64B5F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Enhanced Form Fields */
    .stSelectbox > div > div {
        background: white;
        border-radius: 1rem !important;
        border: 2px solid #E3F2FD !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #90CAF9 !important;
        box-shadow: 0 0 0 4px rgba(33, 150, 243, 0.1);
    }
    
    /* Slider Improvements */
    .stSlider > div > div > div {
        background-color: #2196F3 !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #1976D2 !important;
        box-shadow: 0 0 10px rgba(33, 150, 243, 0.3);
    }
    
    /* Progress Steps */
    .progress-steps {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        padding: 1rem;
        background: white;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .step {
        display: flex;
        align-items: center;
        margin: 0 1rem;
    }
    
    .step-number {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 50%;
        background: linear-gradient(45deg, #42a5f5, #1976d2);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 0.5rem;
        box-shadow: 0 4px 6px rgba(33, 150, 243, 0.2);
    }
    
    .step-text {
        color: #1976D2;
        font-weight: 500;
    }
    
    /* Results Page Cards */
    .result-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        transition: all 0.3s ease;
    }
    
    .result-card.success {
        border-color: #4CAF50;
        background: linear-gradient(to right, #E8F5E9 0%, white 100%);
    }
    
    .result-card.warning {
        border-color: #FFC107;
        background: linear-gradient(to right, #FFF8E1 0%, white 100%);
    }
    
    .result-card.danger {
        border-color: #F44336;
        background: linear-gradient(to right, #FFEBEE 0%, white 100%);
    }
    
    /* Animated Icons */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .icon-pulse {
        animation: pulse 2s infinite;
        display: inline-block;
    }
    
    /* Chart Container Enhancement */
    .chart-container {
        background: white;
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        transform: translateY(-5px);
    }

    /* Main button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #42a5f5 0%, #1976d2 100%) !important;
        color: white !important;
        font-size: 2rem !important;
        font-weight: 900 !important;
        padding: 1.5rem 4rem !important;
        border: none !important;
        border-radius: 4rem !important;
        box-shadow: 0 8px 32px 0 rgba(33, 150, 243, 0.25) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        margin: 1rem 0 !important;
        animation: button-glow 2s infinite !important;
    }

    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 12px 48px 0 rgba(33, 150, 243, 0.35), 0 0 48px 8px #42a5f5 !important;
        background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%) !important;
    }

    @keyframes button-glow {
        0% { box-shadow: 0 8px 32px 0 rgba(33, 150, 243, 0.25), 0 0 32px 4px #90caf9; }
        50% { box-shadow: 0 12px 48px 0 rgba(33, 150, 243, 0.35), 0 0 48px 8px #42a5f5; }
        100% { box-shadow: 0 8px 32px 0 rgba(33, 150, 243, 0.25), 0 0 32px 4px #90caf9; }
    }

    /* Container styling */
    .main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }

    .title-section {
        text-align: center;
        margin-bottom: 2rem;
    }

    .subtitle {
        font-size: 1.2rem;
        color: #37474F;
        margin-bottom: 2.5rem;
        text-align: center;
        max-width: 800px;
    }

    .cta-text {
        font-size: 1.3rem;
        color: #1976D2;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: 1px;
    }

    /* Simplified Personal Information Section */
    .personal-info-section {
        background: white;
        border-radius: 1.5rem;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }

    .personal-info-header {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E3F2FD;
    }

    .personal-info-content {
        padding: 1rem 0;
    }

    /* Improved CGPA Slider */
    .cgpa-slider {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    .cgpa-slider .slider-label {
        color: #1976D2;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    .cgpa-value {
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        color: #1976D2;
        margin: 1rem 0;
        padding: 0.5rem;
        background: #E3F2FD;
        border-radius: 1rem;
        transition: all 0.3s ease;
    }

    /* Improved Selection Box Styling */
    .simple-select .stSelectbox > div > div {
        background: white !important;
        border-radius: 0.5rem !important;
        border: 2px solid #E3F2FD !important;
        padding: 0.5rem !important;
        min-height: 42px !important;
        color: #2C3E50 !important;
        font-size: 1rem !important;
    }

    .simple-select .stSelectbox > div > div:hover {
        border-color: #90CAF9 !important;
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.1) !important;
    }

    /* Label Styling */
    .simple-select label {
        color: #1976D2 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
    }

    /* Hide double labels */
    .simple-select > div > label {
        display: none !important;
    }

    /* Dropdown menu styling */
    div[data-baseweb="select"] > div {
        color: #2C3E50 !important;
        background: white !important;
    }

    /* Fix placeholder color */
    div[data-baseweb="select"] span {
        color: #2C3E50 !important;
    }

    /* Selected option color */
    div[role="option"][aria-selected="true"] {
        color: #2C3E50 !important;
        background: #E3F2FD !important;
    }

    /* Input field styling */
    .stNumberInput > div > div > input {
        color: #2C3E50 !important;
        font-size: 1rem !important;
    }

    /* Add these styles to the existing CSS */
    <style>
    /* Consistent box sizing and spacing for all form elements */
    .simple-select {
        margin-bottom: 2.5rem !important;  /* Increased margin bottom */
        width: 100%;
        position: relative;
        display: flex;
        flex-direction: column;
        height: 90px !important;
    }

    /* Label styling with consistent spacing */
    .simple-select label {
        color: #1976D2 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
        flex-shrink: 0;
    }

    /* Make all select boxes and inputs the same size */
    .simple-select .stSelectbox,
    .simple-select .stNumberInput {
        width: 100% !important;
        flex-grow: 1;
    }

    /* Adjust the alignment for study satisfaction row */
    .study-row {
        display: flex !important;
        gap: 1.5rem !important;
        margin-bottom: 2.5rem !important;
    }

    .study-row > div {
        flex: 1;
        min-width: 0;  /* Prevents flex items from overflowing */
    }

    .study-row .simple-select {
        margin-bottom: 0 !important;  /* Remove margin for items in the row */
    }

    /* Column styling for even spacing */
    [data-testid="column"] {
        padding: 0 0.5rem !important;
    }

    /* Reset Streamlit's default spacing */
    [data-testid="stVerticalBlock"] {
        gap: 0 !important;
        padding: 0 !important;
    }

    /* Add these styles for the 2x2 grid layout */
    .assessment-grid {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 1.5rem !important;
        margin-bottom: 2rem !important;
    }

    .assessment-grid .simple-select {
        margin-bottom: 0 !important;
    }

    /* Reset any column specific margins */
    [data-testid="column"] {
        padding: 0 0.5rem !important;
    }

    /* Ensure consistent height for selection boxes */
    .simple-select .stSelectbox,
    .simple-select .stNumberInput {
        width: 100% !important;
        min-height: 42px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
REQUIRED_FILES = {
    "trained_model.joblib": "Main prediction model",
    "scaler.joblib": "Data scaling preprocessor",
    "feature_names.txt": "Feature names configuration"
}

VALID_RANGES = {
    'academic_pressure': (1, 5),
    'study_satisfaction': (1, 5),
    'work_study_hours': (0, 24),
    'financial_stress': (1, 5),
    'cgpa': (0.0, 4.0)
}

@st.cache_resource
def load_model_and_resources() -> Tuple[Optional[Any], Optional[Any], Optional[list], Optional[str]]:
    """
    Load model and related resources with comprehensive error handling.
    
    Returns:
        Tuple containing:
        - model: Trained model object or None if loading fails
        - scaler: Data scaler object or None if loading fails
        - feature_names: List of feature names or None if loading fails
        - error: Error message string or None if loading succeeds
    """
    try:
        # Verify all required files exist
        missing_files = []
        for file_path, description in REQUIRED_FILES.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_path} ({description})")
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files:\n" + "\n".join(missing_files))
        
        # Load and validate model
        model = joblib.load("trained_model.joblib")
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            raise ValueError("Invalid model: missing required methods")
        
        # Load and validate scaler
        scaler = joblib.load("scaler.joblib")
        if not hasattr(scaler, 'transform'):
            raise ValueError("Invalid scaler: missing transform method")
        
        # Load and validate feature names
        with open('feature_names.txt', 'r') as f:
            feature_names = f.read().splitlines()
        
        if not feature_names:
            raise ValueError("Feature names file is empty")
        
        # Log successful loading
        logger.info("Successfully loaded model and resources")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Number of features: {len(feature_names)}")
        
        return model, scaler, feature_names, None
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        return None, None, None, str(e)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return None, None, None, str(e)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None, None, None, f"An unexpected error occurred: {str(e)}"

def validate_input(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate user input data against defined ranges and rules.
    
    Args:
        data: Dictionary containing user input data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        for field, (min_val, max_val) in VALID_RANGES.items():
            if field in data:
                value = data[field]
                if not isinstance(value, (int, float)):
                    return False, f"{field} must be a number"
                if value < min_val or value > max_val:
                    return False, f"{field} must be between {min_val} and {max_val}"
        
        required_fields = [
            'gender', 'sleep_duration', 'dietary_habits', 'degree',
            'family_mental_history', 'suicidal_thoughts'
        ]
        
        for field in required_fields:
            if field not in data or data[field] is None:
                return False, f"Missing required field: {field}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

# Load model and resources
model, scaler, feature_names, load_error = load_model_and_resources()

# Initialize session state for page flow
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Main content
if load_error:
    st.error(f"Error loading required files: {load_error}")
    st.stop()

# Page 1: Home
if st.session_state.page == "Home":
    # Add custom CSS for the button
    st.markdown("""
        <style>
        /* Main button styling */
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #42a5f5 0%, #1976d2 100%) !important;
            color: white !important;
            font-size: 2rem !important;
            font-weight: 900 !important;
            padding: 1.5rem 4rem !important;
            border: none !important;
            border-radius: 4rem !important;
            box-shadow: 0 8px 32px 0 rgba(33, 150, 243, 0.25) !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
            margin: 1rem 0 !important;
            animation: button-glow 2s infinite !important;
        }

        .stButton > button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 12px 48px 0 rgba(33, 150, 243, 0.35), 0 0 48px 8px #42a5f5 !important;
            background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%) !important;
        }

        @keyframes button-glow {
            0% { box-shadow: 0 8px 32px 0 rgba(33, 150, 243, 0.25), 0 0 32px 4px #90caf9; }
            50% { box-shadow: 0 12px 48px 0 rgba(33, 150, 243, 0.35), 0 0 48px 8px #42a5f5; }
            100% { box-shadow: 0 8px 32px 0 rgba(33, 150, 243, 0.25), 0 0 32px 4px #90caf9; }
        }

        /* Container styling */
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }

        .title-section {
            text-align: center;
            margin-bottom: 2rem;
        }

        .subtitle {
            font-size: 1.2rem;
            color: #37474F;
            margin-bottom: 2.5rem;
            text-align: center;
            max-width: 800px;
        }

        .cta-text {
            font-size: 1.3rem;
            color: #1976D2;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
            letter-spacing: 1px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main content structure
    st.markdown("""
        <div class="main-container">
            <div class="title-section">
                <h1 class="gradient-text" style="font-size: 3rem; margin-bottom: 1rem;">
                    Depression Detection App
                </h1>
                <p class="subtitle">
                    Your mental health matters. Let's take care of it together.
                </p>
            </div>
            <div style="width: 100%; max-width: 500px; text-align: center;">
                <p class="cta-text">Ready to begin your journey?</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Button in a centered column
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Analysis", key="start-analysis", use_container_width=True):
            st.session_state.page = "Input Form"
            st.rerun()

    # Rest of the content
    st.markdown("""
        <!-- Hero Section -->
        <div class='hero-section' style='
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
            border-radius: 2rem;
            padding: 3rem 2rem;
            margin: 2rem 0;
            box-shadow: 0 8px 32px rgba(33,150,243,0.15);
            text-align: center;
            position: relative;
            overflow: hidden;
        '>
            <div class='icon-pulse' style='font-size: 3rem; margin-bottom: 1rem;'>🧠</div>
            <h2 style='color: #1976D2; font-size: 2.4rem; font-weight: bold; margin-bottom: 1.5rem;'>
                Understanding Your Mental Health
            </h2>
            <p style='color: #37474F; font-size: 1.3rem; line-height: 1.6; margin-bottom: 2rem;'>
                Every student faces unique challenges. This tool is here to help you reflect, 
                understand, and take positive steps for your well-being.
                <br><br>
                <span style='color: #1976D2; font-weight: 600;'>You are not alone.</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Motivational Quote Card
    st.markdown("""
        <div class='content-section'>
            <div class='quote-card' style='background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);'>
                <div class='quote-text' style='font-size: 1.3rem;'>
                    "You don't have to control your thoughts. You just have to stop letting them control you."
                </div>
                <div class='quote-author' style='margin-top: 1rem; color: #1976D2; font-weight: 600;'>- Dan Millman</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Story/Statistics Section
    st.markdown("""
        <div class='content-section' style='display: flex; flex-wrap: wrap; gap: 2rem; justify-content: space-between;'>
            <div class='story-card' style='flex: 2; min-width: 260px;'>
                <div class='story-title' style='color: #FF9800; font-size: 1.2rem; font-weight: bold; margin-bottom: 1rem;'>A Student's Story</div>
                <p style='color: #37474F; font-size: 1.1rem;'>
                    "When I started university, I felt overwhelmed by expectations and the pressure to succeed. I struggled with sleep, lost interest in things I loved, and felt isolated. It wasn't until I reached out for help that I realized how many others felt the same way. Seeking support changed my life."
                </p>
                <div style='margin-top: 1rem; color: #1976D2; font-weight: 600;'>
                    Remember: <span style='color: #FF9800;'>Asking for help is a sign of strength.</span>
                </div>
            </div>
            <div style='flex: 1; min-width: 220px;'>
                <div class='stat-card' style='background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%); border-left: 4px solid #2196F3; margin-bottom: 1rem;'>
                    <div class='stat-number' style='font-size: 2rem; color: #1565C0; font-weight: bold;'>1 in 3</div>
                    <div class='stat-label' style='color: #546E7A;'>Students experience significant depression symptoms</div>
                </div>
                <div class='stat-card' style='background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%); border-left: 4px solid #2196F3;'>
                    <div class='stat-number' style='font-size: 2rem; color: #1565C0; font-weight: bold;'>80%</div>
                    <div class='stat-label' style='color: #546E7A;'>Depression is treatable with proper support</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # How it works section
    st.markdown("""
    <div class='main-content'>
        <div class='content-section'>
            <div class='process-section'>
                <h3 class='section-title'>How it works:</h3>
                <ol class='custom-list'>
                    <li>Fill out a simple questionnaire about your current situation</li>
                    <li>Our AI model analyzes your responses</li>
                    <li>Receive personalized feedback and recommendations</li>
                </ol>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Page 2: Input Form
elif st.session_state.page == "Input Form":
    # Container for better width control
    st.markdown("""
        <div style='max-width: 1000px; margin: 0 auto; padding: 0 1rem;'>
    """, unsafe_allow_html=True)
    
    # Back Button with better positioning
    st.markdown("""
        <div class='back-button' style='margin-bottom: 2rem;'>
            <style>
            .fancy-back-btn button {
                background: linear-gradient(90deg, #90CAF9, #1976D2) !important;
                color: white !important;
                font-size: 1.1rem !important;
                font-weight: bold !important;
                border-radius: 2rem !important;
                padding: 0.7rem 2.2rem !important;
                box-shadow: 0 2px 8px rgba(33,150,243,0.18) !important;
                border: none !important;
                transition: all 0.2s cubic-bezier(.4,2,.3,1) !important;
                margin: 0 !important;
            }
            .fancy-back-btn button:hover {
                background: linear-gradient(90deg, #1976D2, #42A5F5) !important;
                transform: scale(1.05) !important;
                box-shadow: 0 4px 16px rgba(33,150,243,0.25) !important;
            }
            </style>
            <div class='fancy-back-btn'>
    """, unsafe_allow_html=True)
    if st.button("← Back to Home", key="back-to-home-btn"):
        st.session_state.page = "Home"
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Main title and description with improved styling
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 class='main-title' style='margin-bottom: 1rem;'>Assessment Form</h1>
            <p style='color: #37474F; font-size: 1.1rem; max-width: 600px; margin: 0 auto;'>
                Your responses will help us better understand your current situation and provide appropriate recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # First blue box - Use exact HTML and style for the two paragraphs
    st.markdown("""
        <div style='
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
            border-radius: 1.5rem;
            padding: 2.5rem;
            margin: 2rem auto;
            max-width: 800px;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.1);
        '>
            <h1 style='
                color: #2196F3;
                font-size: 2rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
            '>You're taking a positive step!</h1>
            <p style='
                color: #37474F;
                font-size: 1.2rem;
                line-height: 1.6;
                margin-bottom: 2rem;
            '>
                This short assessment is private and confidential. There are no right or wrong answers—just be honest with yourself.
            </p>
            <p style='
                color: #2196F3;
                font-size: 1.3rem;
                margin-bottom: 1rem;
            '>
                Remember: You are not alone on this journey.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Progress indicator with improved styling
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2.5rem;'>
            <div style='
                display: inline-flex;
                align-items: center;
                padding: 0.5rem 1.5rem;
                background: linear-gradient(90deg, #42a5f5, #1976d2);
                color: white;
                font-weight: bold;
                border-radius: 2rem;
                box-shadow: 0 2px 8px rgba(33,150,243,0.2);
            '>
                <span style='margin-right: 0.5rem;'>Step 2 of 3:</span>
                <span>Assessment</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Did you know section with lighter cream background
    st.markdown("""
        <div style='
            background: #FFFDE7;
            border-radius: 1rem;
            padding: 2rem;
            margin: 2rem auto;
            max-width: 600px;
            border-left: 4px solid #FFC107;
        '>
            <h3 style='
                color: #FF9800;
                font-size: 1.4rem;
                margin-bottom: 1rem;
            '>Did you know?</h3>
            <p style='
                color: #37474F;
                font-size: 1.1rem;
                margin-bottom: 1rem;
                line-height: 1.5;
            '>
                More than half of students who struggle with mental health never seek help.
            </p>
            <p style='
                color: #2196F3;
                font-size: 1.1rem;
                font-weight: 500;
            '>
                Taking this assessment is a sign of courage and self-care.
            </p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("depression_assessment"):
        # Personal Information Section (2 columns, 3 rows)
        st.markdown("""
            <div style='
                display: flex;
                align-items: center;
                margin-bottom: 2rem;
                padding-bottom: 1rem;
                border-bottom: 2px solid #E3F2FD;
                background: #fff;
                border-radius: 1rem;
                box-shadow: 0 2px 8px rgba(33,150,243,0.06);
                padding: 1rem 1.5rem;
            '>
                <span style='font-size: 1.5rem; margin-right: 0.8rem;'>👤</span>
                <h2 style='
                    margin: 0;
                    color: #1976D2;
                    font-size: 1.5rem;
                    font-weight: 600;
                '>Personal Information</h2>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label>Gender</label>", unsafe_allow_html=True)
            gender = st.selectbox("Gender Select", ["Female", "Male"], label_visibility="collapsed", key="gender_select")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label>Degree Level</label>", unsafe_allow_html=True)
            degree = st.selectbox("Degree Level Select", ["Pre-U", "Diploma", "Undergraduate", "Postgraduate", "PhD"], label_visibility="collapsed", key="degree_select")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label>Dietary Habits</label>", unsafe_allow_html=True)
            dietary_habits = st.selectbox("Dietary Habits Select", ["Unhealthy", "Moderate", "Healthy"], 
                                        label_visibility="collapsed", key="diet_select",
                                        help="How would you rate your overall eating habits?")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label style='margin-bottom: 2rem; color: #37474F; font-size: 1.0rem; font-weight: 500; display: block;'>Age</label>", unsafe_allow_html=True)
            age = st.number_input("Age Input", min_value=18, max_value=60, value=20, label_visibility="collapsed", help="Your current age", key="age_input")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label style='margin-top: 1.0rem; margin-bottom: 0.5rem; color: #37474F; font-size: 1.0rem; font-weight: 500; display: block;'>Sleep Duration</label>", unsafe_allow_html=True)
            sleep_duration = st.selectbox("Sleep Duration Select", [
                "Less than 5 hours",
                "5-6 hours",
                "7-8 hours",
                "More than 8 hours"
            ], label_visibility="collapsed", key="sleep_select")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label>CGPA</label>", unsafe_allow_html=True)
            cgpa = st.selectbox("CGPA Select", [
                "0.0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0",
                "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0"
            ], index=4, label_visibility="collapsed", key="cgpa_select",
            help="Your current Cumulative Grade Point Average")
            st.markdown("</div>", unsafe_allow_html=True)

        # Academic and Work Section with 2x2 grid (aligned)
        st.markdown("""
            <div style='
                margin: 1rem 0;
                width: 100%;
                background: #e3f2fd;
                border-radius: 1.0rem;
            '>
                <div class='section-header' style='
                    display: flex;
                    align-items: center;
                    margin-top: 2rem;
                    margin-bottom: 2rem;
                    padding-bottom: 1rem;
                    border-bottom: 2px solid #E3F2FD;
                    background: #fff;
                    border-radius: 1rem;
                    box-shadow: 0 2px 8px rgba(33,150,243,0.06);
                    padding: 1rem 1.5rem;
                '>
                    <span style='font-size: 1.5rem; margin-right: 0.5rem;'>📚</span>
                    <h2 style='
                        margin: 0;
                        color: #1976D2;
                        font-size: 1.5rem;
                        font-weight: 600;
                        text-align: left;
                    '>Academic and Well-being Assessment</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 2x2 grid in a single row for perfect alignment
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label>Academic Pressure</label>", unsafe_allow_html=True)
            academic_pressure = st.selectbox("Academic Pressure Select", [
                "Very Low", "Low", "Moderate", "High", "Very High"
            ], index=2, label_visibility="collapsed", key="academic_pressure_select",
            help="Rate your level of academic pressure")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label style='margin-bottom: 2rem; display: block;'>Daily Work/Study Hours</label>", unsafe_allow_html=True)
            work_study_hours = st.number_input("Daily Hours Select", 0, 24, 8,
                                             label_visibility="collapsed",
                                             help="Average hours spent on work/study per day",
                                             key="work_hours_input")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label>Study Satisfaction</label>", unsafe_allow_html=True)
            study_satisfaction = st.selectbox("Study Satisfaction Select", [
                "Very Dissatisfied", "Dissatisfied", "Neutral",
                "Satisfied", "Very Satisfied"
            ], index=2, label_visibility="collapsed", key="study_satisfaction_select",
            help="How satisfied are you with your studies?")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label>Financial Stress Level</label>", unsafe_allow_html=True)
            financial_stress = st.selectbox("Financial Stress Select", [
                "No Stress", "Mild", "Moderate", "High", "Severe"
            ], index=2, label_visibility="collapsed", key="financial_stress_select",
            help="Rate your level of financial stress")
            st.markdown("</div>", unsafe_allow_html=True)

        # Mental Health Section with improved styling
        st.markdown("""
            <div class='section-container' style='margin: 3rem 0;'>
                <div class='section-header' style='
                    display: flex;
                    align-items: center;
                    margin-bottom: 1.5rem;
                    border-bottom: 2px solid #E3F2FD;
                    background: #fff;
                    border-radius: 1rem;
                    box-shadow: 0 2px 8px rgba(33,150,243,0.06);
                    padding: 1rem 1rem;
                '>
                    <span style='font-size: 1.5rem; margin-right: 0.5rem;'>🧠</span>
                    <h2 style='
                        margin: 0;
                        color: #1976D2;
                        font-size: 1.5rem;
                        font-weight: 600;
                    '>Mental Health History</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label>Family History of Mental Health Issues</label>", unsafe_allow_html=True)
            family_mental_history = st.selectbox("##", ["No", "Yes"], key="family_history_select",
                                            help="Has anyone in your immediate family been diagnosed with mental health issues?")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='simple-select'>", unsafe_allow_html=True)
            st.markdown("<label>Thoughts of Self-harm</label>", unsafe_allow_html=True)
            suicidal_thoughts = st.selectbox("##", ["No", "Yes"], key="selfharm_select",
                                        help="This information is confidential and helps in risk assessment")
            st.markdown("</div>", unsafe_allow_html=True)

        # Privacy Notice with improved styling
        st.markdown("""
            <div style='
                background: #FFF3E0;
                border-radius: 1rem;
                padding: 1.5rem;
                margin: 2rem 0;
                border-left: 4px solid #FF9800;
            '>
                <h4 style='
                    color: #E65100;
                    margin: 0 0 0.5rem 0;
                    display: flex;
                    align-items: center;
                    font-size: 1.1rem;
                '>
                    ⚠️ Privacy Notice
                </h4>
                <p style='
                    color: #424242;
                    margin: 0;
                    font-size: 1rem;
                    line-height: 1.5;
                '>
                    Your responses are confidential and used only for assessment purposes.
                    If you're experiencing severe symptoms, please seek immediate professional help.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Submit button with improved styling
        st.markdown("""
            <style>
            div[data-testid="stFormSubmitButton"] > button {
                background: linear-gradient(90deg, #1976D2, #42A5F5) !important;
                color: white !important;
                font-size: 1.2rem !important;
                font-weight: bold !important;
                padding: 1rem 2rem !important;
                width: 100% !important;
                max-width: 400px !important;
                margin: 1rem auto !important;
                display: block !important;
                border-radius: 2rem !important;
                border: none !important;
                box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3) !important;
                transition: all 0.3s ease !important;
            }
            div[data-testid="stFormSubmitButton"] > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4) !important;
                background: linear-gradient(90deg, #42A5F5, #1976D2) !important;
            }
            </style>
        """, unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit Assessment")

        if submitted:
            with st.spinner("Processing your responses..."):
                try:
                    # Encode categorical variables
                    gender_encoded = 0 if gender == "Female" else 1
                    sleep_mapping = {
                        "Less than 5 hours": 1,
                        "5-6 hours": 2,
                        "7-8 hours": 3,
                        "More than 8 hours": 4
                    }
                    dietary_mapping = {
                        "Unhealthy": 1,
                        "Moderate": 2,
                        "Healthy": 3
                    }
                    degree_mapping = {
                        "Pre-U": 0,
                        "Diploma": 1,
                        "Undergraduate": 2,
                        "Postgraduate": 3,
                        "PhD": 4
                    }

                    # Map age to age bin
                    def get_age_bin(age):
                        if age <= 19:
                            return 0  # "[18.00, 19.00]"
                        elif age <= 21:
                            return 1  # "(19.00, 21.00]"
                        elif age <= 23:
                            return 2  # "(21.00, 23.00]"
                        elif age <= 24:
                            return 3  # "(23.00, 24.00]"
                        elif age <= 25:
                            return 4  # "(24.00, 25.00]"
                        elif age <= 28:
                            return 5  # "(25.00, 28.00]"
                        elif age <= 29:
                            return 6  # "(28.00, 29.00]"
                        elif age <= 31:
                            return 7  # "(29.00, 31.00]"
                        elif age <= 33:
                            return 8  # "(31.00, 33.00]"
                        else:
                            return 9  # "(33.00, 59.00]"

                    # Create input data dictionary with exact feature names from feature_names.txt
                    input_data = {
                        'gender': gender_encoded,
                        'academic_pressure': ["Very Low", "Low", "Moderate", "High", "Very High"].index(academic_pressure) + 1,
                        'study_satisfaction': ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"].index(study_satisfaction) + 1,
                        'sleep_duration': sleep_mapping[sleep_duration],
                        'dietary_habits': dietary_mapping[dietary_habits],
                        'degree': degree_mapping[degree],
                        'suicidal_thoughts': 1 if suicidal_thoughts == "Yes" else 0,
                        'work_study_hours': work_study_hours,
                        'financial_stress': ["No Stress", "Mild", "Moderate", "High", "Severe"].index(financial_stress) + 1,
                        'family_mental_history': 1 if family_mental_history == "Yes" else 0,
                        'age_bin': get_age_bin(age),
                        'cgpa_scaled': float(cgpa.split("-")[0])  # Get the lower bound of the CGPA range
                    }
                    
                    # Create DataFrame and ensure column order matches feature_names.txt exactly
                    input_df = pd.DataFrame([input_data])
                    
                    # Get the feature names from feature_names.txt
                    with open('feature_names.txt', 'r') as f:
                        feature_names = f.read().splitlines()
                    
                    # Reorder columns to match feature_names.txt exactly
                    input_df = input_df[feature_names]  # Use all features from feature_names.txt
                    
                    # Validate that all required features are present
                    missing_features = set(feature_names) - set(input_df.columns)
                    if missing_features:
                        raise ValueError(f"Missing features: {', '.join(missing_features)}")
                    
                    # Store in session state
                    st.session_state["input_data"] = input_df
                    st.session_state["raw_input"] = {
                        'gender': gender,
                        'age': age,
                        'degree': degree,
                        'cgpa': cgpa,
                        'sleep_duration': sleep_duration,
                        'dietary_habits': dietary_habits,
                        'academic_pressure': academic_pressure,
                        'study_satisfaction': study_satisfaction,
                        'work_study_hours': work_study_hours,
                        'financial_stress': financial_stress,
                        'family_mental_history': family_mental_history,
                        'suicidal_thoughts': suicidal_thoughts
                    }
                    
                    # Navigate to prediction page
                    st.session_state.page = "Prediction"
                    st.rerun()
                    
                except Exception as e:
                    st.error("An error occurred while processing your input.")
                    st.error(f"Error details: {str(e)}")
                    logger.error(f"Form processing error: {str(e)}")
                    st.write("Please try again or contact support if the error persists.")

    st.markdown("</div>", unsafe_allow_html=True)  # Close main container

# Page 3: Prediction
elif st.session_state.page == "Prediction":
    # Fancy Back Button
    st.markdown("""
        <div class='back-button' style='position: relative; margin-bottom: 1.5rem;'>
            <style>
            .fancy-back-btn button {
                background: linear-gradient(90deg, #90CAF9, #1976D2) !important;
                color: white !important;
                font-size: 1.1rem !important;
                font-weight: bold !important;
                border-radius: 2rem !important;
                padding: 0.7rem 2.2rem !important;
                box-shadow: 0 2px 8px rgba(33,150,243,0.18) !important;
                border: none !important;
                transition: all 0.2s cubic-bezier(.4,2,.3,1) !important;
                margin: 0 !important;
            }
            .fancy-back-btn button:hover {
                background: linear-gradient(90deg, #1976D2, #42A5F5) !important;
                transform: scale(1.05) !important;
                box-shadow: 0 4px 16px rgba(33,150,243,0.25) !important;
            }
            </style>
            <div class='fancy-back-btn'>
    """, unsafe_allow_html=True)
    if st.button("← Back to Form", key="back-to-form-btn"):
        st.session_state.page = "Input Form"
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>Step 3: Your Results</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class='content-section' style='background: linear-gradient(90deg, #e3f2fd 60%, #bbdefb 100%); border-radius: 1.5rem; padding: 1.5rem 2rem; margin-bottom: 2rem; box-shadow: 0 4px 16px rgba(33,150,243,0.10);'>
            <h2 style='color: #1976D2; font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;'>Thank you for sharing your story.</h2>
            <p style='color: #37474F; font-size: 1.1rem;'>
                Your responses help us provide you with personalized feedback and support. <br><br>
                <span style='color: #1976D2; font-weight: 600;'>Remember: You are not alone, and support is always available.</span>
            </p>
        </div>
    """, unsafe_allow_html=True)
    # Storytelling/hope card
    st.markdown("""
        <div class='story-card' style='max-width: 500px; margin: 0 auto 2rem auto;'>
            <div class='story-title' style='color: #FF9800;'>A Message of Hope</div>
            <p style='color: #37474F; font-size: 1.05rem;'>
                "Recovery is not a straight line. Many students have faced similar struggles and found their way to brighter days. Reaching out is a sign of strength, not weakness."
            </p>
            <div style='margin-top: 1rem; color: #1976D2; font-weight: 600;'>
                You matter. Your story matters.
            </div>
        </div>
    """, unsafe_allow_html=True)

    if "input_data" not in st.session_state:
        st.warning("Please complete the assessment form first.")
        if st.button("Go to Assessment Form"):
            st.session_state.page = "Input Form"
            st.rerun()
    else:
        try:
            with st.spinner("Analyzing your responses..."):
                # Get input data and scale it
                input_df = st.session_state["input_data"]
                input_scaled = scaler.transform(input_df)
                
                # Make prediction with error handling
                try:
                    prediction = model.predict(input_scaled)[0]
                    prediction_proba = model.predict_proba(input_scaled)[0]
                    confidence = prediction_proba[prediction]
                except Exception as e:
                    logger.error(f"Prediction error: {str(e)}")
                    st.error("Error making prediction. Please try again.")
                    st.stop()

            # Display results in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                    <h3 style='
                        color: #1976D2;
                        font-size: 1.8rem;
                        margin: 1.5rem 0;
                    '>Risk Assessment</h3>
                """, unsafe_allow_html=True)
                
                # Enhanced result card with confidence display
                st.markdown(f"""
                    <div class='result-card {"danger" if prediction == 1 else "success"}'>
                        <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                            <div class='icon-pulse' style='font-size: 2rem; margin-right: 1rem;'>
                                {"⚠️" if prediction == 1 else "✅"}
                            </div>
                            <h3 style='margin: 0; color: {"#F44336" if prediction == 1 else "#4CAF50"};'>
                                {prediction == 1 and "Depression Risk Detected" or "Low Risk Detected"}
                            </h3>
                        </div>
                        <div style='margin: 1.5rem 0;'>
                            <div style='font-weight: bold; color: #1976D2; margin-bottom: 0.5rem;'>Analysis Confidence</div>
                            <div style='background: #E3F2FD; border-radius: 2rem; height: 38px; width: 100%; box-shadow: 0 2px 8px rgba(33,150,243,0.10); position: relative; overflow: hidden;'>
                                <div style='
                                    background: linear-gradient(90deg, #42A5F5, #1976D2);
                                    height: 100%;
                                    width: {confidence*100}%;
                                    border-radius: 2rem;
                                    transition: width 1s cubic-bezier(.4,2,.3,1);
                                    display: flex;
                                    align-items: center;
                                    justify-content: flex-end;
                                    padding-right: 1rem;
                                '>
                                    <span style='color: white; font-weight: bold;'>{confidence*100:.1f}%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                    <h3 style='
                        color: #1976D2;
                        font-size: 1.8rem;
                        margin: 1.5rem 0;
                    '>Key Contributing Factors</h3>
                """, unsafe_allow_html=True)
                # Removed empty chart container box
                
                # Get feature names from feature_names.txt
                with open('feature_names.txt', 'r') as f:
                    feature_names = f.read().splitlines()
                
                # Create feature importance DataFrame with proper alignment
                feature_importance = pd.DataFrame({
                    'Factor': feature_names,
                    'Value': input_df.iloc[0].values,
                    'Importance': model.feature_importances_
                })
                
                # Sort by importance and get top 5
                feature_importance = feature_importance.sort_values('Importance', ascending=False).head(5)
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = plt.barh(feature_importance['Factor'], feature_importance['Importance'])
                
                # Customize the plot
                plt.title("Top 5 Most Influential Factors", pad=20)
                plt.xlabel("Relative Importance")
                
                # Add value labels on the bars
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    plt.text(width, bar.get_y() + bar.get_height()/2,
                            f'{width:.3f}',
                            ha='left', va='center', fontweight='bold')
                
                # Adjust layout
                plt.tight_layout()
                
                # Display the plot
                st.pyplot(fig)
                
                if st.checkbox("Show Your Responses"):
                    st.markdown("""
                        <h3 style='
                            color: #1976D2;
                            font-size: 1.8rem;
                            margin: 1.5rem 0;
                        '>Your Responses</h3>
                    """, unsafe_allow_html=True)
                    raw_input = st.session_state.get("raw_input", {})
                    for key, value in raw_input.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            with col2:
                st.markdown("""
                    <h3 style='
                        color: #1976D2;
                        font-size: 1.8rem;
                        margin: 1.5rem 0;
                    '>Recommendations</h3>
                """, unsafe_allow_html=True)
                if prediction == 1:
                    st.error("""
                    #### Immediate Actions:
                    1. **Seek Professional Help**
                       - Contact a mental health professional
                       - Visit your university counseling center
                       - Schedule a doctor's appointment
                    
                    2. **Emergency Resources**
                       - Malaysia Emergency Response Services (MERS) 999.
                                           
                    3. **Support System**
                       - Talk to trusted friends or family
                       - Join support groups
                       - Connect with student services
                    """)
                    
                    st.markdown("""
                        <div class='warning-box'>
                            <h4>🚨 Important Note</h4>
                            <p>If you're having thoughts of self-harm, please seek immediate help:</p>
                            <ul>
                                <li>Call emergency services</li>
                                <li>Go to the nearest emergency room</li>
                                <li>Contact a crisis hotline</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("""
                    #### Maintain Well-being:
                    1. **Mental Health**
                       - Practice stress management
                       - Maintain work-life balance
                       - Regular exercise
                    
                    2. **Academic Success**
                       - Use university resources
                       - Join study groups
                       - Take regular breaks
                       
                    3. **Social Connection**
                       - Stay connected with friends
                       - Participate in activities
                       - Build support network
                    """)
                    
                    st.info("""
                        #### Prevention Tips:
                        - Maintain regular sleep schedule
                        - Practice healthy eating habits
                        - Exercise regularly
                        - Stay connected with others
                        - Seek help early if needed
                    """)
            
            # Navigation - single button to start over
            st.markdown("---")
            st.markdown("<div class='center-content'>", unsafe_allow_html=True)
            if st.button("Start New Assessment"):
                st.session_state.pop("input_data", None)
                st.session_state.pop("raw_input", None)
                st.session_state.page = "Home"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            logger.error(f"Error in prediction page: {str(e)}")
            st.error("An error occurred while processing your results.")
            st.error(f"Error details: {str(e)}")
            st.markdown("<div class='center-content'>", unsafe_allow_html=True)
            if st.button("Try Again"):
                st.session_state.page = "Input Form"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
