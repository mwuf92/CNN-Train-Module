import os
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Function to load data from file
def load_data(data_file):
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    return x_train, y_train, x_test, y_test

# Function to build a CNN model based on user input
def build_model():
    model_name = input("Enter model name: ")
    num_layers = int(input("Enter number of layers: "))
    input_shape = (int(input("Enter input image height: ")), int(input("Enter input image width: ")), int(input("Enter input image channels: ")))
    
    model = keras.Sequential()
    for i in range(num_layers):
        layer_type = input("Enter layer type (Conv2D/MaxPooling2D/Dense): ")
        if layer_type == "Conv2D":
            filters = int(input("Enter number of filters: "))
            kernel_size = int(input("Enter kernel size: "))
            activation = input("Enter activation function: ")
            model.add(keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape))
        elif layer_type == "MaxPooling2D":
            pool_size = int(input("Enter pool size: "))
            model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
        elif layer_type == "Dense":
            units = int(input("Enter number of units: "))
            activation = input("Enter activation function: ")
            model.add(keras.layers.Dense(units=units, activation=activation))
            
    return model, model_name
    

# Function to train a CNN model based on user input
def train_model(model, model_name, x_train, y_train, x_test, y_test):
    epochs = int(input("Enter number of epochs: "))
    batch_size = int(input("Enter batch size: "))
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    
    return model, epochs

# Greeting message
print("Welcome to the CNN model builder and trainer program!")

# Get user input for number of models
num_models = int(input("Enter number of models to train: "))

# Initialize list to store model details
model_details = []

# Loop through each model
for i in range(num_models):
    # Build the model
    model, model_name = build_model()
    
    # Get user input for training data
    data_file = input("Enter path to data file: ")
    x_train, y_train, x_test, y_test = load_data(data_file)
    
    # Train the model
    trained_model, epochs = train_model(model, model_name, x_train, y_train, x_test, y_test)
    
    # Save the model
    trained_model.save(f"{model_name}.h5")
    
    # Save the model details
    model_details.append([model_name, trained_model.summary(), epochs])
    
# Save model details in a CSV file
csv_file = "model_details.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer
