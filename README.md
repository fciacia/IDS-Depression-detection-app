# Student Depression Prediction - Reproducible ML Pipeline

This project builds a machine learning pipeline to predict student depression using survey data. The pipeline is modular, reproducible, and supports multiple classification models.

## 📁 Project Structure
- `data/`
  - `raw/`: Original untouched datasets
  - `processed/`: Cleaned and transformed datasets
- `scripts/`
  - Modular Python scripts including:
    - `data_loader.py`: Data loading and saving
    - `data_preprocessing.py`: Cleaning and transformation
    - `eda.py`: Exploratory data analysis (plots, summaries)
    - `label_encoders_split_data_model.py`: Label encoding and data splitting
    - `model_training_testing.py`: ML model training and testing
    - `model_selection.py`: Model comparison and selection logic
- `models/`: Serialized trained models (e.g., `.pkl`, `.joblib`)
- `output/`: 
  - Evaluation metrics (e.g., accuracy, F1-score)
  - Confusion matrix plots
  - Log files (optional)
- `main.py`: Entry point to orchestrate the entire pipeline
- `requirements.txt`: List of required Python libraries for reproducibility

## 🚀 How to Run
1. Clone this repository or download the project folder
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## 📁 Project Structure

├── data/  
│   ├── raw/                 # Original raw datasets (do not modify)  
│   ├── processed/           # Cleaned and transformed datasets  
│   └── external/            # External datasets (if any)  

├── models/                  # Trained and saved machine learning models  

├── scripts/                 # Python modules for each pipeline stage  
│   ├── data_loader.py             # load_data() and save_data() functions  
│   ├── data_preprocessing.py      # clean_and_transform(), feature engineering, handling missing values  
│   ├── eda.py                     # Exploratory Data Analysis: count plots, depression rate plots, KDEs  
│   ├── label_encoders_split_data_model.py # Label encoding and train-test splitting logic  
│   ├── model_training_testing.py         # Training and evaluating ML models (SVM, RF, XGBoost, NN, etc.)  
│   ├── model_selection.py                # Model comparison and selection based on performance metrics  
│   └── config.py                         # (Optional) configuration: file paths, constants, hyperparameters  

├── main.py                  # Main orchestration script to run the full pipeline  

├── output/  
│   ├── results.json         # Final evaluation metrics (accuracy, precision, recall, F1)  
│   ├── confusion_matrix.png # Saved confusion matrix or other plots  
│   └── logs/                # Runtime logs or debug information  

├── requirements.txt         # Python library dependencies  

└── README.md                # Project overview, setup, and usage instructions  

# data_loader.py
This module is responsible for handling data input and output operations. It includes:

load_data(): Loads datasets from .csv or other formats into Pandas DataFrames for use in the pipeline.

save_data(): Saves cleaned or processed datasets back to disk.
It acts as the first step in the workflow, ensuring all subsequent modules receive data in the correct format.

# data_preprocessing.py
This script performs critical cleaning and transformation tasks on raw data, including:

Handling missing values

Standardizing or normalizing numerical fields

Encoding categorical variables (if done manually or with custom logic)

Creating new features (feature engineering)
It prepares the dataset for machine learning models by ensuring the input is clean, consistent, and numerically encoded.

# eda.py (Exploratory Data Analysis)
This file generates exploratory visualizations and statistical summaries to help understand data patterns and distributions. It includes:

Count plots and bar charts by category (e.g., education level vs depression rate)

KDE plots and box plots to explore distributions

Comparative graphs that display both frequency and depression rate
These insights guide modeling decisions by revealing class imbalance, outliers, and feature-target relationships.

# label_encoders_split_data_model.py
This module handles:

Label encoding of categorical features using LabelEncoder, converting string labels into numeric values.

Train-test split of the dataset (typically 80-20 or 70-30) using train_test_split.

Feature scaling using StandardScaler, transforming feature values to a common scale for optimal model performance.
It ensures the data is properly preprocessed before entering the training phase.

# model_training_testing.py
This is the core modeling engine of your pipeline. It includes:

Training multiple ML algorithms (e.g., Decision Tree, Random Forest, XGBoost, SVM, Neural Network).

Evaluating model performance using metrics like accuracy, precision, recall, and F1-score.

Generating and saving confusion matrices as heatmaps for visual assessment.

Optionally includes hyperparameter tuning with HalvingRandomSearchCV for models like Random Forest or SVM.
This file encapsulates the training logic and performance evaluation of all ML models.

# model_selection.py
This script compares the performance of all trained models based on metrics such as:

Accuracy

F1-score

Recall / Precision
It selects the best-performing model for deployment or further analysis. The comparison may be printed to console or logged into an output file (e.g., results.json). It ensures objective and data-driven model choice.

# config.py (Optional / Not implemented yet)
Acts as a central location to store:

File paths to raw/processed data

Hyperparameters used across models

Output directories

Flags for turning on/off specific features or debug modes
It improves modularity and makes future edits easier without changing multiple script files.

# __init__
This is the orchestration script that glues everything together. It:

Loads the dataset using data_loader.py

Preprocesses data using data_preprocessing.py and label_encoders_split_data_model.py

Performs exploratory analysis using eda.py

Trains and tests models using model_training_testing.py

Selects the best model via model_selection.py

Stores results and visualizations into the output/ folder
This is the single script you execute to run the full reproducible pipeline from raw data to model evaluation.