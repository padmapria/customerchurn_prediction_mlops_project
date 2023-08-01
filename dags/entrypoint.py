# For Prefect
from prefect import Flow
import prefect_workflow

flow = Flow("My Prefect Workflow")
flow.schedule(cron="0 0 * * *")  # Schedule daily at midnight
flow.set_dependencies(prefect_workflow.download_data_for_the_day, prefect_workflow.download_data, flow_start)

# For Airflow
import airflow_dag

# ... Airflow-specific setup ...

# Your Airflow DAG definition goes here
# ...

