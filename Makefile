.PHONY: install data train anomalies costs api dashboard test lint docker-up docker-down all

install:
	pip install -r requirements.txt

data:
	python src/generate_data.py

train:
	python src/train.py

anomalies:
	python src/anomaly_detection.py

costs:
	python src/cost_savings.py

# Full pipeline: data -> train -> anomalies -> cost simulation
all: data train anomalies costs
	@echo "Pipeline complete. Artifacts in models/. Now run 'make api' or 'make dashboard'."

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v

lint:
	flake8 src/ api/ --max-line-length=120 --ignore=E203,W503

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-up:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down
