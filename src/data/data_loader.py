import os
import kaggle
import pandas as pd

def download_dataset():
    """
    Downloads the News Category Dataset from Kaggle
    Requires Kaggle API credentials to be set up
    """
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'rmisra/news-category-dataset',
        path='data/raw',
        unzip=True
    )

def prepare_dataset():
    """
    Reads and prepares the dataset for processing
    """
    df = pd.read_json('data/raw/News_Category_Dataset_v3.json', lines=True)
    
    # Basic cleaning
    df = df.dropna(subset=['headline', 'category'])
    
    # Save to processed directory
    os.makedirs('data/processed', exist_ok=True)
    df.to_parquet('data/processed/news_dataset.parquet')
    
    return df