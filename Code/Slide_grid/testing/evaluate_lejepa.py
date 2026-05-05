import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os

def evaluate_lejepa_embeddings():
    embed_path = "data/embeddings/embeddings.npy"
    label_path = "data/label/labels.npy"
    
    # Check if the required files exist
    if not (os.path.exists(embed_path) and os.path.exists(label_path)):
        print("Error: Embedding or label files not found.")
        return

    # 1. Load data
    embeddings = np.load(embed_path)
    labels = np.load(label_path)
    
    print(f"Data loaded successfully: Embeddings shape {embeddings.shape}, Labels shape {labels.shape}")

    # 2. Train/Test Split (80/20 split for evaluating embedding quality)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 3. Train a simple Logistic Regression model (Linear Probing)
    print("Training Linear Classifier (Logistic Regression)...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Prediction and Evaluation
    y_pred = clf.predict(X_test)
    
    print("\n" + "="*50)
    print("LeJEPA Embedding Evaluation Results")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    print("="*50)

if __name__ == "__main__":
    evaluate_lejepa_embeddings()