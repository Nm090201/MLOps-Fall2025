# Lab 2: Docker-Based Machine Learning Applications

## Overview
This lab explores containerization of machine learning applications using Docker with two implementations: a simple training pipeline (L1) and a full-stack web application (L2).

---

## L1: Wine Quality Prediction Model

A containerized ML pipeline using Ridge Regression on the Wine Quality dataset. Trains the model, evaluates performance with RMSE and R² metrics, and saves the model and scaler for future use.

**Technologies**: Python, scikit-learn, Docker

---

## L2: FastAPI Wine Quality Web Service

A full-stack web application serving a TensorFlow neural network through FastAPI. Users input 11 wine properties and receive quality predictions (0-10 scale) through an interactive web interface.

**Technologies**: FastAPI, TensorFlow, HTML/CSS/JavaScript, Docker

---

## Key Learnings

### Docker Skills
- Creating Dockerfiles for ML applications
- Managing dependencies and file structures in containers
- Port mapping and container debugging

### ML Deployment
- Model training and persistence
- REST API development with FastAPI
- Separating training from inference
- Full-stack integration of ML models

---

## Challenges Solved

1. **Container accessibility** - Fixed port mapping and file paths
2. **File structure issues** - Corrected Dockerfile copying strategy
3. **Training optimization** - Reduced epochs and optimized architecture

---

## Results

Successfully deployed two containerized ML applications:
- **L1**: Trained model with evaluation metrics
- **L2**: Interactive web service with real-time predictions

Screenshots available in Results folder.

---

## Conclusion

This lab demonstrated containerizing ML applications from prototype to production, covering Docker workflows, API development, and deployment best practices essential for MLOps.

---

**Author**: Nm090201  
**Course**: MLOps Fall 2025  
**Date**: October 22, 2025
