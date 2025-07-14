import os
import kaggle
import pandas as pd
from sklearn.model_selection import train_test_split

def download_dataset():
    """Downloads the News Category Dataset from Kaggle"""
    print("Starting dataset download process...")
    os.makedirs('data/raw', exist_ok=True)
    print(f"Raw data directory created at: {os.path.abspath('data/raw')}")
    
    if not os.path.exists('data/raw/News_Category_Dataset_v3.json'):
        print("Dataset not found, attempting Kaggle download...")
        try:
            kaggle.api.authenticate()
            print("Kaggle authentication successful")
            kaggle.api.dataset_download_files(
                'rmisra/news-category-dataset',
                path='data/raw',
                unzip=True
            )
            print("Dataset downloaded and extracted successfully")
        except Exception as e:
            print(f"Error during download: {str(e)}")
            raise
    else:
        print("Dataset already exists in data/raw directory")

def prepare_dataset():
    """Prepares and splits the dataset"""
    # Download if not exists
    if not os.path.exists('data/raw/News_Category_Dataset_v3.json'):
        download_dataset()
    
    # Read dataset
    df = pd.read_json('data/raw/News_Category_Dataset_v3.json', lines=True)
    df = df.dropna(subset=['headline', 'category'])
    
    # Create train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Save splits
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_parquet('data/processed/train.parquet')
    val_df.to_parquet('data/processed/validation.parquet')
    test_df.to_parquet('data/processed/test.parquet')
    
    return train_df, val_df, test_df