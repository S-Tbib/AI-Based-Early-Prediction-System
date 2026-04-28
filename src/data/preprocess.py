import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_RAW = BASE_DIR / "data" / "raw" / "diabetes.csv"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "model"


# LOAD DATA
def load_data(filepath=DATA_RAW):
    df = pd.read_csv(filepath)
    print(f" Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df



# CLEAN DATA
def handle_zeros(df):
    """
    Replace physiologically impossible zeros with NaN
    """
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    df_clean = df.copy()
    df_clean[zero_columns] = df_clean[zero_columns].replace(0, np.nan)
    
    print(" Zeros replaced with NaN")
    print(f"   Total missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean


# SPLIT DATA
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets
    Uses stratification to preserve class distribution
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(" Data split:")
    print(f"   Train: {X_train.shape[0]}")
    print(f"   Test: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

#  IMPUTATION (NO LEAKAGE)

def impute_data(X_train, X_test):
    """
    Fill missing values using median (robust to outliers)
    """
    imputer = SimpleImputer(strategy='median')
    
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),  #  FIT on TRAIN only
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),  # APPLY on TEST
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("Missing values imputed (median)")
    
    return X_train_imputed, X_test_imputed, imputer

# SCALING (NO LEAKAGE)

def standardize_features(X_train, X_test):
    """
    Standardize features (mean=0, std=1)
    """
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)  # TRAIN
    X_test_scaled = scaler.transform(X_test)        # TEST
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(" Features standardized")
    
    return X_train_scaled, X_test_scaled, scaler

# SAVE DATA
def create_dataset_with_outcome(X, y):
    """
    Create dataframe by adding outcome column to features
    """
    df = X.copy()
    df['Outcome'] = y
    return df


def save_processed_data(X_train, X_test, y_train, y_test, scaler, imputer):
    """
    Save processed datasets and transformers
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train set
    train_df = create_dataset_with_outcome(X_train, y_train)
    train_df.to_csv(DATA_PROCESSED / "train.csv", index=False)
    
    # Test set
    test_df = create_dataset_with_outcome(X_test, y_test)
    test_df.to_csv(DATA_PROCESSED / "test.csv", index=False)
    
    # Full cleaned dataset
    full_df = pd.concat([train_df, test_df])
    full_df.to_csv(DATA_PROCESSED / "cleaned_diabetes.csv", index=False)
    
    # Save transformers
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
    
    print("Data successfully saved")



# GLOBAL PIPELINE

def preprocess_pipeline():
    print("="*60)
    print(" STARTING PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Load
    print("\n[1/6] Loading data...")
    df = load_data()
    
    # 2. Clean
    print("\n[2/6] Cleaning data...")
    df = handle_zeros(df)
    
    # 3. Split (VERY IMPORTANT)
    print("\n[3/6] Splitting data...")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 4. Impute
    print("\n[4/6] Imputing missing values...")
    X_train, X_test, imputer = impute_data(X_train, X_test)
    
    # 5. Scale
    print("\n[5/6] Scaling features...")
    X_train, X_test, scaler = standardize_features(X_train, X_test)
    
    # 6. Save
    print("\n[6/6] Saving data...")
    save_processed_data(X_train, X_test, y_train, y_test, scaler, imputer)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return X_train, X_test, y_train, y_test, scaler, imputer



# EXECUTION
if __name__ == "__main__":
    preprocess_pipeline()