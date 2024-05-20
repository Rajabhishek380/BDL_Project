from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 20),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}

dag = DAG(
    'simplified_car_price_prediction_dag',
    default_args=default_args,
    description='A simplified DAG for car price data processing',
    schedule_interval=None,
    catchup=False
)

download_extract_task = BashOperator(
    task_id='download_dataset',
    bash_command='curl -o /tmp/cars.zip "https://www.kaggle.com/datasets/sujithmandala/second-hand-car-price-prediction/download" && unzip -o /tmp/cars.zip -d /tmp/',
    dag=dag
)

spark_job_task = SparkSubmitOperator(
    task_id='process_cars_data',
    application='/path/to/your/spark_script.py',  # Path to the Spark script
    name='simple_car_price_prediction_job',
    conn_id='spark_default',
    application_args=['/tmp/cars.csv', '/tmp/processed_cars.csv'],
    dag=dag
)

download_extract_task >> spark_job_task