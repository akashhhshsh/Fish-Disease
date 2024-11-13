import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Paths to folders
data_dir = "/Users/akashadhyapak/Documents/ML/Fish Disease/parent directory"
infected_dir = os.path.join(data_dir, "Fresh Augmented 1.0")
non_infected_dir = os.path.join(data_dir, "Infected augmented 2.0")

# Parameters
img_height, img_width = 128, 128
test_size = 0.3  
val_size = 0.1  
random_state = 42

# Load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_width, img_height))
            img = img / 255.0  # Normalize
            images.append(img.flatten())  
            labels.append(label)
    return images, labels

# Load data
infected_images, infected_labels = load_images_from_folder(infected_dir, label=1)
non_infected_images, non_infected_labels = load_images_from_folder(non_infected_dir, label=0)

# Combine and split data
X = np.array(infected_images + non_infected_images)
y = np.array(infected_labels + non_infected_labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)

# Train SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=random_state) 
svm_model.fit(X_train, y_train)

# Predict and evaluate on training and validation sets for accuracy graph
train_accuracy = accuracy_score(y_train, svm_model.predict(X_train))
val_accuracy = accuracy_score(y_val, svm_model.predict(X_val))

# Plot training vs validation accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Training Accuracy', 'Validation Accuracy'], [train_accuracy, val_accuracy], color=['skyblue', 'salmon'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy for SVM Model')
plt.show()

# Predict on test set and evaluate
y_pred = svm_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Non-Infected', 'Infected'],
            yticklabels=['Non-Infected', 'Infected'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for SVM Classifier")
plt.show()

# Save the SVM model
joblib.dump(svm_model, 'svm_model.joblib')
