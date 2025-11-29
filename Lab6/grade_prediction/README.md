# Student Grade Prediction ML Pipeline

## Overview
This project demonstrates an automated machine learning pipeline that trains a student grade prediction model and deploys it to Google Cloud Storage using GitHub Actions.

## What It Does
The pipeline predicts student final grades based on three key factors:
- Study hours per week
- Attendance rate
- Previous grade performance

## Workflow Automation
The system uses GitHub Actions to automatically:
- Train a linear regression model on synthetic student data
- Evaluate model performance using standard regression metrics
- Version and store trained models in Google Cloud Storage with timestamps
- Run on a daily schedule or manual trigger

## Key Features
- **Automated Training**: Models retrain daily at midnight UTC
- **Cloud Storage**: All model versions are preserved in GCS for tracking and rollback
- **CI/CD Integration**: Seamless deployment pipeline triggered by code changes
- **Model Versioning**: Timestamp-based naming ensures no model overwrites

## Technologies Used
- **Machine Learning**: scikit-learn for model training and evaluation
- **Cloud Platform**: Google Cloud Storage for model persistence
- **CI/CD**: GitHub Actions for automation
- **Python**: Core implementation language

## Workflow Structure
![Workflow Structure](image.png)

The diagram shows the complete CI/CD pipeline flow from code commit to model deployment in GCS.

## Requirements
- Google Cloud Platform account with Storage bucket
- GCP Service Account credentials stored as GitHub secret
- Python 3.10 with required ML libraries

## Metrics Tracked
- Mean Squared Error (MSE): Measures prediction accuracy
- R² Score: Indicates model fit quality

## Use Case
Educational institutions can use this pipeline to:
- Predict at-risk students early in the semester
- Allocate tutoring resources effectively
- Identify factors most influencing academic success
- Track model performance over time

## Future Enhancements
- Add real student data integration
- Implement model performance monitoring
- Create prediction API endpoint
- Add automated email alerts for model drift
