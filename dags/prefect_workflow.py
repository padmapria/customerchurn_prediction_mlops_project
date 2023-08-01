# prefect_workflow.py

from prefect import task
from workflow import download_data, download_data_for_the_day, main

# Use the @task decorator to register the functions as Prefect tasks
download_data = task(download_data)
download_data_for_the_day = task(download_data_for_the_day)

# Define Prefect-specific DAG here (if applicable)
# ...
