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
    "1_dag_data_generation",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(3),
) as dag:

    generate = DockerOperator(
        image="danidarya/airflow-data-generation",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-data-generation",
        do_xcom_push=False,
        volumes=["/home/daria/PycharmProjects/airflow_ml_dags/data:/data"]
    )


    generate
