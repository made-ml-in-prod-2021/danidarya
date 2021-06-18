from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "danidarya",
    "email": ["babina.daria@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "2_dag_train_model",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(5),
) as dag:

    preprocessing = DockerOperator(
        image="danidarya/airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=["/home/daria/PycharmProjects/airflow_ml_dags/data:/data"]
    )

    splitting = DockerOperator(
        image="danidarya/airflow-split",
        command="--data_for_split_dir /data/processed/{{ ds }}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=["/home/daria/PycharmProjects/airflow_ml_dags/data:/data"]
    )

    training = DockerOperator(
        image="danidarya/airflow-model-training",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}",
        task_id="docker-airflow-model-training",
        do_xcom_push=False,
        volumes=["/home/daria/PycharmProjects/airflow_ml_dags/data:/data"]
    )

    validation = DockerOperator(
        image="danidarya/airflow-model-validation",
        command="--data-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }} --metrics-dir /data/metrics/{{ ds }}",
        task_id="docker-airflow-model-validation",
        do_xcom_push=False,
        volumes=["/home/daria/PycharmProjects/airflow_ml_dags/data:/data"]
    )

    preprocessing >> splitting >> training >> validation
