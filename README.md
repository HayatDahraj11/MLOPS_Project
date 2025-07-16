# MLOps News Classification Pipeline

A production-ready MLOps pipeline for news article classification using modern DevOps practices and machine learning workflows. This project demonstrates end-to-end ML pipeline implementation with automated data processing, model training, experiment tracking, API serving, and comprehensive monitoring.

## 🚀 Features

- **Automated Data Pipeline**: Apache Airflow orchestrates data ingestion and preprocessing
- **Experiment Tracking**: MLflow tracks all model experiments, parameters, and metrics
- **Model Serving**: FastAPI provides RESTful API endpoints with graceful error handling
- **Containerization**: Fully Dockerized microservices architecture
- **Monitoring**: Prometheus metrics collection with Grafana dashboards
- **Database**: PostgreSQL for data persistence
- **Scalable Architecture**: Docker Compose orchestration for easy deployment

## 📊 Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Apache Airflow │────▶│     MLflow      │────▶│    FastAPI      │
│   (Port 8080)   │     │   (Port 5001)   │     │   (Port 8000)   │
│                 │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PostgreSQL    │     │   PostgreSQL    │     │   Prometheus    │
│   (Airflow DB)  │     │   (MLflow DB)   │     │   (Port 9090)   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │     Grafana     │
                                                │   (Port 3000)   │
                                                └─────────────────┘
```

## 🛠️ Tech Stack

- **Orchestration**: Apache Airflow 2.x
- **ML Tracking**: MLflow 2.x
- **API Framework**: FastAPI
- **Containerization**: Docker & Docker Compose
- **Monitoring**: Prometheus + Grafana
- **Database**: PostgreSQL
- **ML Libraries**: scikit-learn, NLTK, pandas, numpy

## 📋 Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ (for local development)
- 8GB RAM minimum
- 10GB free disk space

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <(https://github.com/HayatDahraj11/MLOPS_Project)>
   cd MLOPS_project
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Access the services**
   - Airflow UI: http://localhost:8080 (Username: `airflow`, Password: `airflow`)
   - MLflow UI: http://localhost:5001
   - FastAPI Docs: http://localhost:8000/docs
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (Username: `admin`, Password: `admin`)

## 📁 Project Structure

```
mlops-news-classification/
├── airflow/
│   ├── dags/
│   │   ├── data_preprocessing_pipeline.py
│   │   └── model_training_pipeline.py
│   ├── Dockerfile
│   └── requirements.txt
├── api/
│   ├── serve_api.py
│   ├── Dockerfile
│   └── requirements.txt
├── mlflow/
│   └── artifacts/
├── data/
│   ├── raw/
│   └── processed/
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboards/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔄 Pipeline Workflow

### 1. Data Preprocessing Pipeline
- Triggered via Airflow DAG
- Downloads and processes news articles
- Performs text cleaning and feature extraction
- Stores processed data for model training

### 2. Model Training Pipeline
- Trains multiple ML models:
  - Neural Network (Best: 54.7% accuracy)
  - Logistic Regression (54.5% accuracy)
  - Naive Bayes (49.3% accuracy)
  - Random Forest (22.7% accuracy)
- Logs experiments to MLflow
- Saves best model for serving

### 3. Model Serving
- FastAPI loads the best model from MLflow
- Provides `/predict` endpoint for real-time inference
- Implements graceful error handling
- Exposes Prometheus metrics

### 4. Monitoring
- Prometheus collects metrics:
  - API request count
  - Response latency
  - Prediction distribution
- Grafana visualizes metrics in real-time

## 🎯 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Breaking news about technology..."}'
```

### View Metrics
```bash
curl http://localhost:8000/metrics
```

## 📊 Model Performance

Based on MLflow experiments:

| Model | Validation Accuracy | Test Accuracy | Test F1-Score |
|-------|-------------------|---------------|---------------|
| Neural Network | 54.7% | 54.7% | 0.547 |
| Logistic Regression | 54.5% | 54.5% | 0.545 |
| Naive Bayes | 49.3% | 49.3% | 0.493 |
| Random Forest | 22.7% | 22.7% | 0.227 |

## 🔧 Configuration

### Airflow Configuration
- DAG refresh interval: 30 seconds
- Max active runs: 1
- Retry attempts: 2

### MLflow Configuration
- Backend store: PostgreSQL
- Artifact store: Local filesystem
- Default experiment: "news-classification"

### API Configuration
- Workers: 4
- Timeout: 30 seconds
- Model refresh: On startup

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```bash
   docker-compose exec airflow-scheduler python -c "
   import nltk
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   "
   ```

2. **MLflow Connection Issues**
   ```bash
   # Check MLflow logs
   docker-compose logs mlflow
   
   # Restart MLflow service
   docker-compose restart mlflow
   ```

3. **Permission Issues**
   ```bash
   # Fix MLflow directory permissions
   sudo chown -R $(whoami):$(whoami) ./mlflow
   chmod -R 755 ./mlflow
   ```

4. **Port Conflicts**
   ```bash
   # Check if ports are in use
   netstat -tulpn | grep -E '(8080|5001|8000|9090|3000)'
   ```

## 📈 Monitoring Dashboard

Access Grafana at http://localhost:3000 and import the provided dashboard to monitor:
- API request rate
- Response time percentiles
- Error rate
- Model prediction distribution
- System resource usage

## 🚀 Deployment

### Production Deployment Checklist
- [ ] Update environment variables for production
- [ ] Configure proper database credentials
- [ ] Set up SSL/TLS certificates
- [ ] Configure resource limits in docker-compose.yml
- [ ] Set up backup strategies for databases
- [ ] Configure log aggregation
- [ ] Set up alerts in Grafana

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Apache Airflow for workflow orchestration
- MLflow for experiment tracking
- FastAPI for high-performance API serving
- The open-source community for amazing tools

## 📞 Contact

For questions or support, please open an issue in the GitHub repository.

---

