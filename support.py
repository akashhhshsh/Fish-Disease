import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set paths
data_dir = "/Users/akashadhyapak/Documents/ML/Fish Disease/Kaggle_upload"  
img_height, img_width = 128, 128  # Resize images to 128x128 for SVM
batch_size = 32

# Data preprocessing without augmentation, just rescaling
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create training and validation generators
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # Binary classification for Fresh vs NonFresh
    subset='training',
    shuffle=False
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Extract features and labels from the generators
def extract_features_labels(generator):
    num_samples = generator.samples
    features = np.zeros((num_samples, img_height * img_width * 3))  # Flattened size for RGB images
    labels = np.zeros((num_samples,))
    
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        features[i*batch_size:(i+1)*batch_size] = x_batch.reshape(x_batch.shape[0], -1)  # Flatten images
        labels[i*batch_size:(i+1)*batch_size] = y_batch
    
    return features, labels

# Get training and validation data
X_train, y_train = extract_features_labels(train_generator)
X_val, y_val = extract_features_labels(validation_generator)

# Train-Test Split (for the SVM)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=157)

# SVM Model
svm_model = SVC(kernel='linear', random_state=157)

# Train the SVM
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fresh', 'NonFresh'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for SVM Classifier (Fish Data)')
plt.show()

# Evaluate on the validation set
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
