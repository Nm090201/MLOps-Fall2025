# ML Pipeline with Airflow & FastAPI

Automated ML pipeline that trains a model daily and serves predictions via API.

## 🎓 Apache Airflow - What I Learned

### Core Concepts

**Airflow** is a workflow orchestration platform where workflows are defined as Python code.

**DAG (Directed Acyclic Graph)**: Represents a workflow with multiple tasks and their dependencies.

**Operators**: Building blocks that perform specific actions (run Python code, send emails, execute bash commands).

**Scheduling**: Uses cron expressions to automate workflows at specified intervals.

**Dependencies**: Controls task execution order using simple operators.

### What I Implemented

**PythonOperator**: Executed ML model training function that loads data, trains logistic regression, and saves the model.

**EmailOperator**: Configured email notifications for successful and failed pipeline runs with HTML content.

**BashOperator**: Used for logging and system checks.

**Webhook/API Triggers**: Explored triggering DAGs via REST API and manual triggers through the web UI.

**Task Dependencies**: Chained training task to email notification task for sequential execution.

**Scheduling**: Set pipeline to run daily at 2 AM automatically.

### Why Airflow?

- Workflows as code (Python-based, version controlled)
- Automated scheduling with cron
- Web UI for monitoring pipeline status
- Automatic retry on failures
- Handles complex task dependencies
- Email alerts for pipeline events

### When to Use Airflow

**Best for**: Batch workflows, scheduled tasks, ETL pipelines, ML training workflows

**Not for**: Real-time streaming, event-driven processes, millisecond latency requirements

## 🏗️ My Implementation

Created an ML pipeline that:
- Trains logistic regression model daily on advertising data
- Sends email notifications on completion
- Serves predictions via FastAPI
- Monitors everything through Airflow UI

**Input**: User behavior data (time on site, age, income, internet usage)
**Output**: Prediction if user will click ad with confidence score

**Technologies**: Airflow, FastAPI, Docker, scikit-learn, PostgreSQL

## 🎯 Key Takeaway

Airflow transforms manual workflows into automated, monitored, and reliable pipelines through code, making it essential for modern MLOps practices.