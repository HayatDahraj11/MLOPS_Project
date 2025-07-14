FROM apache/airflow:2.7.1

USER root

RUN pip install --no-cache-dir kaggle pandas scikit-learn pyarrow

USER airflow