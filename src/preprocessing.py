import pandas as pd
import numpy as np

def clean_data(df):
    # 1. Standardize column names (remove quotes/spaces)
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    
    # 2. Drop the index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # 3. Handle 'LineOfCode' (Convert "null" strings to NaN)
    df['LineOfCode'] = pd.to_numeric(df['LineOfCode'], errors='coerce')
    df['LineOfCode'] = df['LineOfCode'].fillna(df['LineOfCode'].median())
    
    # 4. Clean 'NoOfImage' (Zero out negatives)
    df['NoOfImage'] = df['NoOfImage'].apply(lambda x: max(0, x))

    # 5. !!! THE CRITICAL FIX !!!
    # Pop the label out entirely so it cannot be touched by dummy encoding
    y = df['label'].astype(int).values 
    X = df.drop(columns=['label'])
    
    # 6. Encode only the features
    X = pd.get_dummies(X, drop_first=True)
    
    # Ensure X and y are returned as a clean pair
    return X, y