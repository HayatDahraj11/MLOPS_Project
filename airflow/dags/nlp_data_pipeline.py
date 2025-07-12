from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
import sys
import os

# Add project root to Python path
sys.path.insert(0, '/opt/airflow')

from src.data.data_loader import download_dataset, prepare_dataset
from src.data.processor import TextProcessor

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
    'nlp_data_pipeline',
    default_args=default_args,
    description='NLP data processing pipeline',
    schedule_interval='@daily',
    catchup=False
)

def process_and_split_data():
    """Process and split the dataset"""
    processor = TextProcessor()
    train_df, val_df, test_df = prepare_dataset()
    
    # Process each split
    for df, split_name in [(train_df, 'train'), (val_df, 'validation'), (test_df, 'test')]:
        processed_df = processor.process_dataset(df)
        processed_df.to_parquet(f'data/processed/processed_{split_name}.parquet')

# Define tasks
start = DummyOperator(task_id='start', dag=dag)

download_data = PythonOperator(
    task_id='download_data',
    python_callable=download_dataset,
    dag=dag
)

process_data = PythonOperator(
    task_id='process_data',
    python_callable=process_and_split_data,
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

# Set task dependencies
start >> download_data >> process_data >> end