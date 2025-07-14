FROM apache/airflow:2.7.1
USER root
# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    net-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER airflow
# Install Python packages as the airflow user
RUN pip install --no-cache-dir kaggle pandas scikit-learn pyarrow nltk mlflow prometheus-client
# Download NLTK data (updated for newer NLTK versions)
RUN python -c "import nltk; \
    nltk.download('punkt_tab'); \
    nltk.download('punkt'); \
    nltk.download('stopwords'); \
    nltk.download('wordnet'); \
    nltk.download('omw-1.4'); \
    print('All NLTK data downloaded successfully!')"