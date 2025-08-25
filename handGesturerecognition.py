import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ✅ Correct dataset path
dataset_path = r"D:/Moulesh/prodigy infotech tasks/hand gesture/leapGestRecog"

# ------------------------------
# Function: Load dataset
# ------------------------------
def load_dataset(path, img_size=(64, 64)):
    X, y = [], []
    print("Loading dataset...")

    # Walk through all subfolders recursively
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png"):
                filepath = os.path.join(root, file)

                # Class label is the parent folder name (00,01,...09)
                label = os.path.basename(root)

                # Read and preprocess image
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                X.append(img.flatten())
                y.append(label)

        # Show progress
        if dirs == []:  # only print when in leaf folders
            print(f"Loaded folder: {os.path.basename(root)}")

    X, y = np.array(X), np.array(y)
    print(f"✅ Loaded {len(X)} samples with {len(set(y))} gesture classes.")
    print("Classes:", set(y))
    return X, y

# ------------------------------
# Train Model
# ------------------------------
def train_model(X, y):
    print("\nTraining SVM...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.3f}")
    return model

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    X, y = load_dataset(dataset_path)
    if len(X) == 0:
        print("❌ Dataset not loaded. Check path or folder structure.")
    else:
        model = train_model(X, y)
