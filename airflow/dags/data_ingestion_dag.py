from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import sys
import os

# Add project root to Python path
sys.path.insert(0, '/opt/airflow')

from src.data.data_loader import download_dataset

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
    'data_ingestion_pipeline',
    default_args=default_args,
    description='Data ingestion pipeline for news classification',
    schedule_interval='@daily',
    catchup=False
)

start = DummyOperator(task_id='start', dag=dag)

download_data = PythonOperator(
    task_id='download_data',
    python_callable=download_dataset,
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

start >> download_data >> end