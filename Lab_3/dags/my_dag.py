from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.model_development import load_data, data_preprocessing, build_model
from src.email_notification import send_success_email

# DAG configuration
default_args = {
    'owner': 'data-team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

# Create DAG
with DAG(
    dag_id='ml_ad_click_prediction',
    default_args=default_args,
    description='ML Pipeline: Ad Click Prediction',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['machine-learning', 'classification']
) as dag:
    
    # Task 1: Load data
    task_load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )
    
    # Task 2: Preprocess data
    task_preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=data_preprocessing,
        op_args=[task_load_data.output],
    )
    
    # Task 3: Train and save model
    task_train_model = PythonOperator(
        task_id='train_model',
        python_callable=build_model,
        op_args=[task_preprocess.output, "model.pkl"],
    )
    
    # Task 4: Send success email
    task_email = PythonOperator(
        task_id='send_email_notification',
        python_callable=send_success_email,
        provide_context=True,
    )
    
    # Define task dependencies
    task_load_data >> task_preprocess >> task_train_model >> task_email