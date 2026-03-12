from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import gradglass as gg

RND = 42

def leakage_scaling_before_split(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # uses information from the whole dataset (including what will be test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=RND)
    gg.run_leakage_detection(X_train, y_train, X_test, y_test)
    model = LogisticRegression(solver="liblinear", random_state=RND)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def correct_scaling_after_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RND)
    gg.run_leakage_detection(X_train, y_train, X_test, y_test)  
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit only on train
    X_test_scaled = scaler.transform(X_test)        # transform test with train statistics
    model = LogisticRegression(solver="liblinear", random_state=RND)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    return accuracy_score(y_test, preds)

def main():
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                               n_redundant=2, n_repeated=0, class_sep=1.0,
                               flip_y=0.05, random_state=RND)

    acc_leak = leakage_scaling_before_split(X, y)
    acc_correct = correct_scaling_after_split(X, y)

    print(f"Accuracy (scaling BEFORE split) : {acc_leak:.4f}  <-- potential data leakage")
    print(f"Accuracy (scaling AFTER split)  : {acc_correct:.4f}  <-- correct procedure")

    scaler_full = StandardScaler().fit(X)
    X_train, _, _, _ = train_test_split(X, y, test_size=0.3, random_state=RND)
    scaler_train = StandardScaler().fit(X_train)
    mean_full = np.round(scaler_full.mean_, 3)
    mean_train = np.round(scaler_train.mean_, 3)
    print("First 5 feature means (full-data fit) :", mean_full[:5].tolist())
    print("First 5 feature means (train-only fit) :", mean_train[:5].tolist())

if __name__ == "__main__":
    main()