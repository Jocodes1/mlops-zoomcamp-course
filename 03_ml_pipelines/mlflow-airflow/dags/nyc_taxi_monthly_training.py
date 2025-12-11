from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from include.pipeline import run_training

with DAG(
    dag_id="nyc_taxi_monthly_training",
    schedule=None,                    # ← no schedule = manual only
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["ml", "taxi"],
) as dag:

    def train_fixed_month():
        # ONE month that definitely exists
        run_training(year=2023, month=1)   # train 2023-01 → validate 2023-02

    PythonOperator(
        task_id="train_jan_2023",
        python_callable=train_fixed_month,
    )