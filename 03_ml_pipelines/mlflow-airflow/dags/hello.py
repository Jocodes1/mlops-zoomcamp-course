from airflow import DAG
from airflow.operators.python import PythonOperator # pyright: ignore[reportMissingImports]
from datetime import datetime

def hello():
    print("Hello World from Astronomer Airflow")

with DAG(
    "hello_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
):
    t1 = PythonOperator(
        task_id="hello_task",
        python_callable=hello
    )
