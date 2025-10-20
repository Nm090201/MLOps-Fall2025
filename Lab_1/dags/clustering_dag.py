from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow/dags/src')

from lab import (
    load_data, 
    scale_features, 
    find_optimal_clusters,
    train_and_save_model, 
    predict_test_data
)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'customer_clustering_pipeline',
    default_args=default_args,
    description='Customer segmentation using KMeans clustering',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'clustering', 'kmeans'],
)

# Task 1: Load data
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

# Task 2: Scale features
def scale_wrapper(**context):
    ti = context['ti']
    data = ti.xcom_pull(task_ids='load_data')
    return scale_features(data)

scale_task = PythonOperator(
    task_id='scale_features',
    python_callable=scale_wrapper,
    provide_context=True,
    dag=dag,
)

# Task 3: Find optimal clusters
def find_k_wrapper(**context):
    ti = context['ti']
    data = ti.xcom_pull(task_ids='scale_features')
    return find_optimal_clusters(data)

find_k_task = PythonOperator(
    task_id='find_optimal_k',
    python_callable=find_k_wrapper,
    provide_context=True,
    dag=dag,
)

# Task 4: Train and save model
def train_wrapper(**context):
    ti = context['ti']
    data = ti.xcom_pull(task_ids='find_optimal_k')
    return train_and_save_model(data, "kmeans_model.pkl")

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_wrapper,
    provide_context=True,
    dag=dag,
)

# Task 5: Predict on test data
def predict_wrapper(**context):
    ti = context['ti']
    filename = ti.xcom_pull(task_ids='train_model')
    return predict_test_data(filename)

predict_task = PythonOperator(
    task_id='predict_test',
    python_callable=predict_wrapper,
    provide_context=True,
    dag=dag,
)

# Pipeline: load_data → scale_features → find_optimal_k → train_model → predict_test
load_data_task >> scale_task >> find_k_task >> train_task >> predict_task