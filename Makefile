# Makefile

.PHONY: all setup run_project run_docker

# Default target
all: setup run_project

# Set up the project environment
setup:
	pip install -r requirements.txt
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 &
	prefect server start & prefect cloud login -k pnu_BSMcUokn

# Check if Docker is running, start if not
start_docker:
	if ! docker info >/dev/null 2>&1; then \
		service docker start; \
	fi
	
# Run the project, (it will also build the docker image with name 'myproject-image')
run_project:
	python src/training_flow.py
	python src/runtest_deploy.py

# Run the Docker container
run_docker:
	docker run -d myproject-image
