from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to Python path
sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_split_pipeline',
    default_args=default_args,
    description='Data splitting pipeline for news classification',
    schedule_interval='@daily',
    catchup=False
)

def split_data():
    """Split the processed dataset into train, validation, and test sets"""
    df = pd.read_parquet('data/processed/news_dataset.parquet')
    
    # Create train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Save splits
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_parquet('data/processed/train.parquet')
    val_df.to_parquet('data/processed/validation.parquet')
    test_df.to_parquet('data/processed/test.parquet')
    
    print(f"Data split completed: Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

start = DummyOperator(task_id='start', dag=dag)

split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

start >> split_data_task >> end