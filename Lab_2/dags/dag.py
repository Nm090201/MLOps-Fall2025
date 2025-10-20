"""
Airflow DAG for ML Pipeline
Runs daily model training
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow/src')
from model import train_pipeline

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': ['nidhi.mallik2001@gmail.com'],
    'email_on_failure': True,
}

'''dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Daily ML training',
    schedule_interval='35 11 * * *',  # Cron: 11:35 AM every day
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml'],
)'''


dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Daily ML training',
    schedule_interval=None,  # No automatic schedule
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml'],
)

def train_task():
    """Train model"""
    metrics = train_pipeline()
    print(f"Training complete: {metrics}")
    return metrics

train = PythonOperator(
    task_id='train_model',
    python_callable=train_task,
    dag=dag,
)

email = EmailOperator(
    task_id='send_email',
    to='nidhi.mallik2001@gmail.com',
    subject='Model Training Complete',
    html_content='<p>Model trained successfully on {{ ds }}</p>',
    dag=dag,
)

train >> email