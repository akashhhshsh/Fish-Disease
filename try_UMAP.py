import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to folders
data_dir = "/Users/akashadhyapak/Documents/ML/Fish Disease/parent directory"
infected_dir = os.path.join(data_dir, "Fresh Augmented 1.0")
non_infected_dir = os.path.join(data_dir, "Infected augmented 2.0")

# Parameters
img_height, img_width = 128, 128
test_size = 0.3
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
            img = img / 255.0  
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

# UMAP for dimensionality reduction
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=random_state)
X_train_umap = umap_reducer.fit_transform(X_train)
X_test_umap = umap_reducer.transform(X_test)

# Logistic Regression Model
logreg_model = LogisticRegression(random_state=random_state)
logreg_model.fit(X_train_umap, y_train)

# Predict on test set
y_pred = logreg_model.predict(X_test_umap)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Non-Infected', 'Infected'],
            yticklabels=['Non-Infected', 'Infected'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for UMAP + Logistic Regression Classifier")
plt.show()
