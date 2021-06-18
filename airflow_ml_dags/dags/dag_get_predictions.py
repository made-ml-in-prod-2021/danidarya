from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

default_args = {
    "owner": "danidarya",
    "email": ["babina.daria@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "3_dag_get_predictions",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:

    model_path = Variable.get("model_path")
    get_predictions = DockerOperator(
        image="danidarya/airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/predictions/{{ ds }}",
        task_id="docker-airflow-get-predictions",
        do_xcom_push=False,
        environment={
            'MODEL_PATH': model_path
        },
        volumes=["/home/daria/PycharmProjects/airflow_ml_dags/data:/data"]
    )

    get_predictions