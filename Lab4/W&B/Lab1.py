#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost Multi-class Classification with Weights & Biases Tracking
Dataset: Iris Dataset (Flower Species Classification)
A classic dataset for beginners - classifies iris flowers into 3 species
"""

import os
import wandb
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_and_prepare_data():
    """Load the Iris dataset and split into train and test sets"""
    print("Loading Iris dataset...")
    
    # Load the famous Iris dataset
    iris = load_iris()
    X = iris.data  # Features: sepal length, sepal width, petal length, petal width
    y = iris.target  # Target: 0=setosa, 1=versicolor, 2=virginica
    
    # Split into 70% training and 30% testing
    train_X, test_X, train_Y, test_Y = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Dataset loaded: {len(iris.data)} samples, {len(iris.feature_names)} features")
    print(f"Classes: {iris.target_names}")
    print(f"Training samples: {len(train_Y)}, Test samples: {len(test_Y)}")
    
    return train_X, train_Y, test_X, test_Y, iris.target_names

def train_xgboost_model(train_X, train_Y, test_X, test_Y):
    """Train XGBoost model with wandb tracking"""
    
    # Initialize wandb run
    run = wandb.init(project="Lab1-Iris-Classification", name="xgboost-iris")
    
    # Create DMatrix for XGBoost
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    
    # Setup parameters for xgboost
    param = {
        'objective': 'multi:softmax',  # multi-class classification
        'eta': 0.3,                     # learning rate (higher for small dataset)
        'max_depth': 4,                 # maximum tree depth
        'silent': 1,                    # verbosity
        'nthread': 4,                   # number of threads
        'num_class': 3,                 # 3 iris species
        'eval_metric': 'mlogloss'       # evaluation metric
    }
    
    # Log parameters to wandb
    wandb.config.update(param)
    
    # Training watchlist
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 50  # More rounds for better convergence
    
    # Train the model with wandb callback
    print("\nTraining XGBoost model...")
    bst = xgb.train(param, xg_train, num_round, watchlist, 
                   callbacks=[wandb.xgboost.WandbCallback()])
    
    return bst, run

def evaluate_model(bst, test_X, test_Y, class_names, run):
    """Evaluate the model and log results"""
    
    # Get predictions
    xg_test = xgb.DMatrix(test_X)
    pred = bst.predict(xg_test)
    
    # Calculate accuracy and error rate
    accuracy = accuracy_score(test_Y, pred)
    error_rate = 1 - accuracy
    
    print(f'\n{"="*50}')
    print(f'Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'Test Error Rate: {error_rate:.4f}')
    print(f'{"="*50}')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_Y, pred, target_names=class_names))
    
    # Log metrics to wandb
    run.summary['Accuracy'] = accuracy
    run.summary['Error Rate'] = error_rate
    
    # Create confusion matrix
    wandb.sklearn.plot_confusion_matrix(test_Y, pred, class_names.tolist())
    
    return accuracy, error_rate

def main():
    """Main execution function"""
    
    # Login to wandb (will prompt for API key if not logged in)
    print("="*50)
    print("XGBoost Iris Classification with W&B Tracking")
    print("="*50)
    print("\nLogging into Weights & Biases...")
    wandb.login()
    
    # Load and prepare data
    train_X, train_Y, test_X, test_Y, class_names = load_and_prepare_data()
    
    # Train model
    bst, run = train_xgboost_model(train_X, train_Y, test_X, test_Y)
    
    # Evaluate model
    accuracy, error_rate = evaluate_model(bst, test_X, test_Y, class_names, run)
    
    # Finish wandb run
    run.finish()
    print("\n" + "="*50)
    print("Training complete! Check your W&B dashboard for results.")
    print("="*50)

if __name__ == "__main__":
    main()