import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Path to the dataset folders
drowsy_dir = "dataset/drowsiness/"
active_dir = "dataset/active/"

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Load images from folders
drowsy_images = load_images_from_folder(drowsy_dir)
active_images = load_images_from_folder(active_dir)

# Label images
drowsy_labels = np.zeros(len(drowsy_images))
active_labels = np.ones(len(active_images))

# Concatenate images and labels
X = np.array(drowsy_images + active_images)
y = np.concatenate((drowsy_labels, active_labels))

# Preprocess images (resize and normalize)
X = np.array([cv2.resize(image, (100, 100)) for image in X])
X = X.astype('float32') / 255.0

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation accuracy:", accuracy)

# Save the model
model.save("drowsiness_detection_model.h5")
