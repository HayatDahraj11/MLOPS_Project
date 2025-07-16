#!/bin/bash

echo "=================================================="
echo "🚀 MLOps News Classification Pipeline Demo"
echo "=================================================="
echo ""
echo "Model: Neural Network (MLPClassifier)"
echo "Accuracy: 55.2% on test data"
echo "Categories: business, entertainment, politics, sport, tech, media, etc."
echo ""

# Check services
echo "📊 Service Health Check:"
health=$(curl -s http://localhost:8000/health)
echo "API Status: $(echo $health | jq -r .status)"
echo "Model Type: $(echo $health | jq -r .model_type)"
echo ""

# Make predictions
echo "🔮 Making Predictions:"
echo ""

predictions=(
    '{"text": "Apple unveils revolutionary new AI chip for machine learning"}'
    '{"text": "Stock market hits record high as tech companies surge"}'
    '{"text": "Lakers win NBA championship in overtime thriller"}'
    '{"text": "New Marvel movie breaks box office records worldwide"}'
    '{"text": "President announces major climate policy changes"}'
    '{"text": "Google launches quantum computing breakthrough"}'
)

labels=("Tech News" "Business News" "Sports News" "Entertainment" "Politics" "Technology")

for i in {0..5}; do
    echo "Test $((i+1)): ${labels[$i]}"
    echo "Input: $(echo ${predictions[$i]} | jq -r .text)"
    
    result=$(curl -s -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d "${predictions[$i]}")
    
    category=$(echo $result | jq -r .category)
    confidence=$(echo $result | jq -r '.confidence * 100 | floor / 100')
    
    echo "→ Predicted: $category (${confidence}% confidence)"
    echo ""
done

echo "=================================================="
echo "✅ Demo Complete!"
echo ""
echo "📌 Access Points:"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • MLflow Experiments: http://localhost:5001"
echo "  • Airflow DAGs: http://localhost:8080 (airflow/airflow123)"
echo "  • Prometheus Metrics: http://localhost:9090"
echo "  • Grafana Dashboards: http://localhost:3000 (admin/admin)"
echo "=================================================="

