import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from preprocess import load_and_preprocess_data

def get_data_path():
    # Automatically locate the data file whether running from root or src/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check possible paths
    path_from_src = os.path.join(current_dir, '../data/kaggle_no_shows.csv')
    path_from_root = os.path.join(current_dir, 'data/kaggle_no_shows.csv')
    
    if os.path.exists(path_from_src):
        return os.path.normpath(path_from_src)
    elif os.path.exists(path_from_root):
        return os.path.normpath(path_from_root)
    else:
        # Fallback default
        return '../data/kaggle_no_shows.csv'

def train_model():
    print("Loading and preprocessing data...")
    data_path = get_data_path()
    print(f"Using data path: {data_path}")
    
    df = load_and_preprocess_data(data_path)
    
    # Select available numeric and categorical features automatically
    exclude_cols = ['patientid', 'appointmentid', 'scheduledday', 'appointmentday', 'no_show', 'target', 'neighbourhood']
    features = [col for col in df.columns if col not in exclude_cols and col != 'target']
    
    print(f"Features selected for training: {features}")
    
    X = df[features]
    y = df['target']
    
    if len(X) == 0:
        raise ValueError("Feature matrix X is empty. Cannot train model.")

    # Identify categorical vs numerical features
    cat_features = [col for col in ['gender'] if col in X.columns]
    num_features = [col for col in features if col not in cat_features]
    
    # Train-test split
    print("Splitting data into train and test sets...")
    stratify_target = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_target
    )
    
    # Preprocessor pipeline
    transformers = []
    if cat_features:
        transformers.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features))
    if num_features:
        transformers.append(('num', StandardScaler(), num_features))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    # Create model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    print("Training the Random Forest model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    if y.nunique() > 1:
        y_prob = model.predict_proba(X_test)[:, 1]
        print("\n--- Model Evaluation Results ---")
        print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    else:
        print("\n--- Model Evaluation Results ---")
        print("Note: Target column has only one unique class value in this file version.")
        
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    train_model()