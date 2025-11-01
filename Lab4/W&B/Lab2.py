#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple CNN Image Classification with Weights & Biases
Dataset: CIFAR-10 (10 types of objects in color images)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import wandb
from wandb.integration.keras import WandbMetricsLogger
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow import keras


# Configuration
config = {
    'learning_rate': 0.001,
    'epochs': 5,
    'batch_size': 128,
}

# Class names for CIFAR-10
class_names = ["Airplane", "Car", "Bird", "Cat", "Deer", 
               "Dog", "Frog", "Horse", "Ship", "Truck"]


def load_data():
    """Load and prepare CIFAR-10 dataset"""
    print("Loading CIFAR-10 dataset...")
    
    # Load dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values to 0-1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train.flatten(), 10)
    y_test = to_categorical(y_test.flatten(), 10)
    
    print(f"Training images: {len(X_train)}")
    print(f"Test images: {len(X_test)}")
    print(f"Image shape: {X_train.shape[1:]}")
    
    return X_train, y_train, X_test, y_test


def create_model():
    """Create a simple CNN model"""
    model = keras.Sequential([
        # First convolutional layer
        keras.layers.Conv2D(32, (3, 3), activation='relu', 
                           input_shape=(32, 32, 3), padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel created successfully!")
    model.summary()
    
    return model


def train_model():
    """Main training function"""
    print("\n" + "="*50)
    print("CIFAR-10 Image Classification")
    print("="*50 + "\n")
    
    # Login to W&B
    print("Logging into Weights & Biases...")
    wandb.login()
    
    # Initialize W&B
    run = wandb.init(
        project="Lab2-CIFAR10-Simple",
        config=config
    )
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Create model
    model = create_model()
    
    # Train the model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=[WandbMetricsLogger()],
        verbose=1
    )
    
    # Evaluate the model
    print("\n" + "="*50)
    print("Final Results")
    print("="*50)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Log final metrics
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy
    })
    
    # Save the model
    model.save('cifar10_model.h5')
    print("\nModel saved as 'cifar10_model.h5'")
    
    # Finish W&B run
    run.finish()
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("Check your W&B dashboard for results.")
    print("="*50)


if __name__ == "__main__":
    train_model()