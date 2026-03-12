import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent))
from gradglass import gg


def demo_random_forest():
    print("\n" + "=" * 60)
    print("Demo 1 — Random Forest Classifier")
    print("=" * 60)

    X, y = make_classification(n_samples=4_000, n_features=20, n_informative=12, n_redundant=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, oob_score=True, random_state=42)

    run = gg.run("random-forest-clf", n_estimators=200, max_depth=8)
    run.watch(clf)  # detects sklearn automatically
    run.fit(
        X_train,
        y_train,  # wraps clf.fit() and captures diagnostics
        X_val=X_test,
        y_val=y_test,
    )

    y_pred = clf.predict(X_test)
    run.log(test_accuracy=accuracy_score(y_test, y_pred), oob_score=clf.oob_score_)

    run.check_leakage(X_train, y_train, X_test, y_test)
    report = run.finish(open=False)

    print(f"  Test accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  OOB score     : {clf.oob_score_:.4f}")
    print(classification_report(y_test, y_pred, target_names=["class-0", "class-1"]))


def demo_pipeline():
    print("\n" + "=" * 60)
    print("Demo 2 — sklearn Pipeline (Scaler → LogisticRegression)")
    print("=" * 60)

    X, y = make_classification(
        n_samples=3_000, n_features=15, n_informative=10, n_classes=3, n_clusters_per_class=1, random_state=7
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, C=1.0, multi_class="multinomial"))]
    )

    run = gg.run("lr-pipeline", C=1.0, max_iter=500)
    run.watch(pipe)
    run.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    y_pred = pipe.predict(X_test)
    run.log(test_accuracy=accuracy_score(y_test, y_pred))
    run.finish(open=False)

    print(f"  Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Iterations   : {pipe.named_steps['clf'].n_iter_[0]}")


def demo_clustering():
    print("\n" + "=" * 60)
    print("Demo 3 — KMeans Clustering (unsupervised)")
    print("=" * 60)

    X, y_true = make_blobs(n_samples=2_000, centers=5, cluster_std=1.2, random_state=0)

    kmeans = KMeans(n_clusters=5, n_init=10, max_iter=300, random_state=0)

    run = gg.run("kmeans-clustering", n_clusters=5)
    run.watch(kmeans)
    run.fit(X)  # no y for unsupervised

    labels = kmeans.labels_
    sil = silhouette_score(X, labels)
    run.log(inertia=kmeans.inertia_, silhouette=sil)
    run.finish(open=False)

    print(f"  Inertia     : {kmeans.inertia_:.2f}")
    print(f"  Silhouette  : {sil:.4f}")
    print(f"  Clusters    : {len(np.unique(labels))}")


def demo_gradient_boosting():
    print("\n" + "=" * 60)
    print("Demo 4 — Gradient Boosting Classifier")
    print("=" * 60)

    X, y = make_classification(n_samples=5_000, n_features=25, n_informative=15, random_state=99)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

    gbc = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, subsample=0.8, random_state=99)

    run = gg.run("gradient-boosting", n_estimators=150, lr=0.1)
    run.watch(gbc)
    run.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    y_pred = gbc.predict(X_test)
    run.log(test_accuracy=accuracy_score(y_test, y_pred))
    run.finish(open=False)

    print(f"  Test accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    demo_random_forest()
    demo_pipeline()
    demo_clustering()
    demo_gradient_boosting()
    print("\n✅  All sklearn demos complete.  Run `gg.open_last()` to explore.")
