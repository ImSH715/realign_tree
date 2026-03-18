import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# zero-division ignorance
warnings.filterwarnings('ignore')

def main():
    embedding_dir = "data/embeddings"
    label_dir = "data/label"

    print("Loading extracted embeddings and labels...")
    
    # Load data
    X_train = np.load(os.path.join(embedding_dir, "train_embeddings.npy"))
    y_train = np.load(os.path.join(label_dir, "train_labels.npy"))
    
    X_val = np.load(os.path.join(embedding_dir, "val_embeddings.npy"))
    y_val = np.load(os.path.join(label_dir, "val_labels.npy"))
    
    X_test = np.load(os.path.join(embedding_dir, "test_embeddings.npy"))
    y_test = np.load(os.path.join(label_dir, "test_labels.npy"))

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    # 1-NN
    print("\nTraining 1-NN Classifier for Evaluation...")
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(X_train, y_train)

    # Evaluate funtion
    def evaluate(name, X, y_true):
        y_pred = knn.predict(X)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\n--- {name} Set Performance ---")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f} (Weighted)")
        print(f"Recall    : {rec:.4f} (Weighted)")
        print(f"F1-Score  : {f1:.4f} (Weighted)")

    # Validation & Test Evaluation
    evaluate("Validation", X_val, y_val)
    evaluate("Test", X_test, y_test)

if __name__ == "__main__":
    main()