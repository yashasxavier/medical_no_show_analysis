import pandas as pd

def load_and_preprocess_data(file_path):
    # Load dataset safely
    df = pd.read_csv(file_path, encoding='utf-8-sig', sep=None, engine='python')
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace('-', '_').str.replace(' ', '_')
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Identify target column dynamically
    target_col = None
    for col in df.columns:
        if 'show' in col:
            target_col = col
            break
            
    # If no-show column isn't in this file version, use 'diabetes' or the last column as a target
    if target_col is None:
        print("Warning: 'no_show' column not found. Using 'diabetes' as a placeholder target for training.")
        target_col = 'diabetes' if 'diabetes' in df.columns else df.columns[-1]

    # Handle dates safely if they exist
    if 'scheduledday' in df.columns and 'appointmentday' in df.columns:
        df['scheduledday'] = pd.to_datetime(df['scheduledday'], errors='coerce')
        df['appointmentday'] = pd.to_datetime(df['appointmentday'], errors='coerce')
        df['waiting_days'] = (df['appointmentday'].dt.normalize() - df['scheduledday'].dt.normalize()).dt.days
        df['waiting_days'] = df['waiting_days'].fillna(0).clip(lower=0) # prevent negative or NaN values
        df['appointment_day_of_week'] = df['appointmentday'].dt.dayofweek.fillna(0)
    else:
        df['waiting_days'] = 0
        df['appointment_day_of_week'] = 0

    # Map target variable safely to 0 and 1
    df['target'] = df[target_col].apply(lambda x: 1 if str(x).strip().capitalize() in ['Yes', '1', 'True'] else 0)
    
    # Ensure we don't return an empty dataframe
    if len(df) == 0:
        raise ValueError("The dataset is empty after preprocessing!")
        
    return df

if __name__ == "__main__":
    df = load_and_preprocess_data('../data/kaggle_no_shows.csv')
    print("Preprocessing successful! Final shape:", df.shape)