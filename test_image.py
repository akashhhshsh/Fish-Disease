import os
import cv2
import numpy as np
import joblib

# Parameters
img_height, img_width = 128, 128  

# Load the saved SVM model
model_path = "svm_modell.joblib"
svm_model = joblib.load(model_path)

# Function to preprocess a new image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read at path: {image_path}")
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0  
    return img.flatten().reshape(1, -1)  

# Path to the new image
new_image_path = "/Users/akashadhyapak/Documents/ML/Fish Disease/test-non infected.png"  


threshold = -5  

# Preprocess and predict
try:
    new_image_features = preprocess_image(new_image_path)
   
    decision_score = svm_model.decision_function(new_image_features)
    
    # Prediction based on threshold
    prediction = 1 if decision_score > threshold else 0
    label = "Infected" if prediction == 1 else "Non-Infected"
    
    print(f"The model predicts that the fish is: {label}")
    print(f"Decision Score: {decision_score[0]}")
    
except Exception as e:
    print(f"Error: {e}")
