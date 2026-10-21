import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess_data

def analyze_features():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(current_dir, '../data/kaggle_no_shows.csv'))
    
    df = load_and_preprocess_data(data_path)
    
    exclude_cols = ['patientid', 'appointmentid', 'scheduledday', 'appointmentday', 'no_show', 'target', 'neighbourhood']
    features = [col for col in df.columns if col not in exclude_cols and col != 'target']
    
    if 'gender' in df.columns:
        df['gender_binary'] = df['gender'].apply(lambda x: 1 if str(x).strip().upper() == 'M' else 0)
        features = [f if f != 'gender' else 'gender_binary' for f in features]

    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Extract feature importances
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    print("\n--- TOP FACTORS DRIVING NO-SHOWS ---")
    print(feature_imp_df.to_string(index=False))

if __name__ == "__main__":
    analyze_features()
    