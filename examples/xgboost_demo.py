import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from gradglass import gg

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not found. Install with:  pip install xgboost")
    sys.exit(1)


def demo_functional_api():
    print("\n" + "=" * 60)
    print("Demo 1 — XGBoost functional API (xgb.train)")
    print("=" * 60)

    X, y = make_classification(n_samples=6_000, n_features=30, n_informative=18, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    run = gg.run("xgb-binary-functional", **{k: v for k, v in params.items() if isinstance(v, (int, float, str))})

    class _XGBPlaceholder:
        _estimator_type = "classifier"

        def get_params(self, deep=True):
            return params

        def fit(self, *a, **kw):
            pass

    run.watch(_XGBPlaceholder())
    cb = run.xgboost_callback()

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dtest, "test")],
        callbacks=[cb],
        verbose_eval=False,
    )

    preds = (booster.predict(dtest) > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    run.log(final_test_accuracy=acc)
    run.finish(open=False)

    print(f"  Rounds trained : 200")
    print(f"  Test accuracy  : {acc:.4f}")


def demo_sklearn_classifier():
    print("\n" + "=" * 60)
    print("Demo 2 — XGBClassifier (sklearn API)")
    print("=" * 60)

    X, y = make_classification(
        n_samples=5_000, n_features=25, n_informative=15, n_classes=3, n_clusters_per_class=1, random_state=7
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    run = gg.run("xgb-multiclass-sklearn")
    run.watch(clf)
    run.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        # Pass the GradGlass callback so per-round metrics are captured
        callbacks=[run.xgboost_callback()],
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    acc = accuracy_score(y_test, clf.predict(X_test))
    run.log(test_accuracy=acc)
    run.finish(open=False)

    print(f"  Test accuracy  : {acc:.4f}")


def demo_sklearn_regressor():
    print("\n" + "=" * 60)
    print("Demo 3 — XGBRegressor (sklearn API)")
    print("=" * 60)

    X, y = make_regression(n_samples=4_000, n_features=20, n_informative=12, noise=0.3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    reg = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        eval_metric="rmse",
        random_state=0,
        verbosity=0,
    )

    run = gg.run("xgb-regression-sklearn")
    run.watch(reg)
    run.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        callbacks=[run.xgboost_callback()],
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    rmse = mean_squared_error(y_test, reg.predict(X_test)) ** 0.5
    run.log(test_rmse=rmse)
    run.finish(open=False)

    print(f"  Test RMSE : {rmse:.4f}")


if __name__ == "__main__":
    demo_functional_api()
    demo_sklearn_classifier()
    demo_sklearn_regressor()
    print("\n✅  All XGBoost demos complete.  Run `gg.open_last()` to explore.")
