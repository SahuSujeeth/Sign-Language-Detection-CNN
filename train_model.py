import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data = []
labels = []

dataset_path = "dataset"

# -----------------------------
# LOAD DATA
# -----------------------------
for label in os.listdir(dataset_path):
    path = os.path.join(dataset_path, label)

    if not os.path.isdir(path):
        continue

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))

        data.append(img)
        labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)

print("Total samples:", len(data))

# -----------------------------
# ENCODE LABELS
# -----------------------------
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

print("Total classes:", len(le.classes_))

# -----------------------------
# SHUFFLE
# -----------------------------
data, labels = shuffle(data, labels, random_state=42)

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(X_train)

# -----------------------------
# MODEL
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=25,
    validation_data=(X_test, y_test)
)

# -----------------------------
# FINAL ACCURACY
# -----------------------------
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print("\n==============================")
print(f"Final Training Accuracy: {train_acc*100:.2f}%")
print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
print("==============================")

# -----------------------------
# ACCURACY GRAPH
# -----------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.savefig("accuracy_graph.png")
plt.close()

# -----------------------------
# METRICS (PRECISION, RECALL, CONFUSION)
# -----------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification Report
print("\nClassification Report:\n")
report = classification_report(
    y_true, y_pred_classes,
    target_names=le.classes_,
    output_dict=True
)

print(classification_report(y_true, y_pred_classes, target_names=le.classes_))

# Save CSV (🔥 NEW)
df = pd.DataFrame(report).transpose()
df.to_csv("classification_report.csv")

print("Classification report saved as classification_report.csv")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("model.h5")

with open("labels.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model trained and saved!")