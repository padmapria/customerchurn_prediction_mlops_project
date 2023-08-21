# Use a base image, for example, Python
FROM python:3.11.1

# Set the working directory inside the Docker container
WORKDIR /app

COPY requirements.txt ./

# Copy the contents of the 'data' directory from your local machine to the 'data' directory in the container
COPY data /app/data

# Copy the contents of the 'data_processing' directory
COPY data_processing /app/data_processing

# Copy the contents of the 'config' directory
COPY config /app/config

# Copy the contents of the 'model' directory
COPY model /app/model

# Copy the contents of the 'model_artifacts' directory
COPY model_artifacts /app/model_artifacts

# Install any required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run when the container starts
CMD ["python", "model/predict.py"]
