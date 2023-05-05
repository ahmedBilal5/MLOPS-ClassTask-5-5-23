from datetime import timedelta
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization

default_args = {
    'owner': 'Ahmed Bilal ',
    'depends_on_past': False,
    'start_date': days_ago(31),
    'email': ['ahmedbilal2015@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

#instantiates a directed acyclic graph
dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='A Machine Learning Pipeline for training a decision tree.',
    schedule_interval=timedelta(days=30),
)


preprocess_data = BashOperator(
    task_id='preprocess_data',
    bash_command='python3 /home/ahmed/mlops/classtask/preprocess_data.py',
    dag=dag,
)

# instantiate tasks using Operators.
#BashOperator defines tasks that execute bash scripts. In this case, we run Python scripts for each task.
train_model = BashOperator(
    task_id='train_model',
    depends_on_past=True,
    bash_command='python3 /home/ahmed/mlops/classtask/train.py',
    retries=3,
    dag=dag,
)

evaluate_model = BashOperator(
    task_id='evaluate_model',
    bash_command='python3 /home/ahmed/mlops/classtask/evaluate_model.py',
    dag=dag,
)

build_docker_image = DockerOperator(
    task_id='build_docker_image',
    image='my_ml_model:latest',
    dockerfile='Dockerfile',
    build_kwargs={
        'path': '/home/ahmed/mlops/classtask/deploy_model.py',
    },
    dag=dag,
)

run_docker_container = DockerOperator(
    task_id='run_docker_container',
    image='my_ml_model:latest',
    command='/bin/bash -c "gunicorn -w 4 -b 0.0.0.0:5000 model_deployment:app"',
    ports=[5000],
    dag=dag,
)

preprocess_data >> train_model >> evaluate_model >> build_docker_image >> run_docker_container
