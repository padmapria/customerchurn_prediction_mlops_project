import pandas as pd
from sklearn.preprocessing import LabelEncoder
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset

# Load historical and new data

historical_data = pd.read_csv("historical_data.csv")
new_data = pd.read_csv("new_data.csv")



data_drift = Report(metrics = [DataDriftPreset()])
data_drift.run(current_data = current.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00'],
               reference_data = reference,
               column_mapping=column_mapping)

data_drift.show()

data_drift.save("data_drift_dashboard_after_week1.html")

