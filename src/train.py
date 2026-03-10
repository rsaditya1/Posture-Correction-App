import os
import sys
import random
import time
import json

import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
import onnxruntime as ort


def set_all_seeds(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"All seeds set to {seed}")


def load_data(config):
    """Load train, val, test splits."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_path = os.path.join(base_dir, config["data"]["train_path"])
    val_path = os.path.join(base_dir, config["data"]["val_path"])
    test_path = os.path.join(base_dir, config["data"]["test_path"])
    features_path = os.path.join(base_dir, config["data"]["feature_names_path"])

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    with open(features_path) as f:
        feature_cols = json.load(f)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    print(f"Features: {feature_cols}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_baseline(X_train, y_train, X_val, y_val, seed):
    """Train logistic regression baseline."""
    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION (BASELINE)")
    print("=" * 50)

    lr = LogisticRegression(random_state=seed, max_iter=1000)
    lr.fit(X_train, y_train)

    train_pred = lr.predict(X_train)
    val_pred = lr.predict(X_val)

    metrics = {
        "train_acc": accuracy_score(y_train, train_pred),
        "train_f1": f1_score(y_train, train_pred),
        "val_acc": accuracy_score(y_val, val_pred),
        "val_f1": f1_score(y_val, val_pred),
    }

    print(f"Train Acc: {metrics['train_acc']:.4f}  Train F1: {metrics['train_f1']:.4f}")
    print(f"Val Acc:   {metrics['val_acc']:.4f}  Val F1:   {metrics['val_f1']:.4f}")

    return lr, metrics


def train_random_forest(X_train, y_train, X_val, y_val, seed):
    """Train random forest."""
    print("\n" + "=" * 50)
    print("RANDOM FOREST")
    print("=" * 50)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, random_state=seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)
    val_pred = rf.predict(X_val)

    metrics = {
        "train_acc": accuracy_score(y_train, train_pred),
        "train_f1": f1_score(y_train, train_pred),
        "val_acc": accuracy_score(y_val, val_pred),
        "val_f1": f1_score(y_val, val_pred),
    }

    print(f"Train Acc: {metrics['train_acc']:.4f}  Train F1: {metrics['train_f1']:.4f}")
    print(f"Val Acc:   {metrics['val_acc']:.4f}  Val F1:   {metrics['val_f1']:.4f}")

    return rf, metrics


def train_xgboost(X_train, y_train, X_val, y_val, config, seed):
    """Train XGBoost with optional grid search."""
    print("\n" + "=" * 50)
    print("XGBOOST")
    print("=" * 50)

    xgb_config = config["xgboost"]
    search_config = config.get("search", {})

    if search_config.get("enabled", False):
        print("Running grid search (this may take a minute)...")
        print(f"Grid: {search_config['param_grid']}")
        print(f"CV folds: {config['cv_folds']}")

        base_model = XGBClassifier(
            subsample=xgb_config["subsample"],
            colsample_bytree=xgb_config["colsample_bytree"],
            reg_lambda=xgb_config["reg_lambda"],
            reg_alpha=xgb_config["reg_alpha"],
            eval_metric=xgb_config["eval_metric"],
            random_state=seed,
        )

        grid = GridSearchCV(
            base_model,
            param_grid=search_config["param_grid"],
            cv=config["cv_folds"],
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )

        search_start = time.time()
        grid.fit(X_train, y_train)
        search_time = time.time() - search_start

        print(f"\nGrid search completed in {search_time:.1f}s")
        print(f"Best params: {grid.best_params_}")
        print(f"Best CV F1:  {grid.best_score_:.4f}")

        model = grid.best_estimator_
        best_params = grid.best_params_

    else:
        model = XGBClassifier(
            n_estimators=xgb_config["n_estimators"],
            max_depth=xgb_config["max_depth"],
            learning_rate=xgb_config["learning_rate"],
            subsample=xgb_config["subsample"],
            colsample_bytree=xgb_config["colsample_bytree"],
            reg_lambda=xgb_config["reg_lambda"],
            reg_alpha=xgb_config["reg_alpha"],
            eval_metric=xgb_config["eval_metric"],
            random_state=seed,
            early_stopping_rounds=xgb_config["early_stopping_rounds"],
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=20,
        )
        best_params = {
            "n_estimators": xgb_config["n_estimators"],
            "max_depth": xgb_config["max_depth"],
            "learning_rate": xgb_config["learning_rate"],
        }

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    metrics = {
        "train_acc": accuracy_score(y_train, train_pred),
        "train_f1": f1_score(y_train, train_pred),
        "val_acc": accuracy_score(y_val, val_pred),
        "val_f1": f1_score(y_val, val_pred),
        "best_params": best_params,
    }

    print(f"\nTrain Acc: {metrics['train_acc']:.4f}  Train F1: {metrics['train_f1']:.4f}")
    print(f"Val Acc:   {metrics['val_acc']:.4f}  Val F1:   {metrics['val_f1']:.4f}")

    # Feature importance
    print("\nFeature Importance:")
    importances = model.feature_importances_

    return model, metrics


def export_to_onnx(model, feature_cols, output_path):
    """Export XGBoost model to ONNX format."""
    print(f"\nExporting model to ONNX: {output_path}")

    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    n_features = len(feature_cols)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_xgboost(model, initial_types=initial_type)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Verify it loads
    session = ort.InferenceSession(output_path)
    print(f"ONNX export verified. Input shape: {session.get_inputs()[0].shape}")

    return output_path


def measure_latency(onnx_path, n_features, n_runs=200):
    """Measure inference latency on both CPU and GPU."""
    print(f"\n--- Latency Measurement ({n_runs} runs) ---")

    dummy_input = np.random.randn(1, n_features).astype(np.float32)

    # GPU inference
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")

    results = {}

    if "CUDAExecutionProvider" in providers:
        session_gpu = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider"]
        )
        input_name = session_gpu.get_inputs()[0].name

        # Warmup
        for _ in range(20):
            session_gpu.run(None, {input_name: dummy_input})

        # Measure
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            session_gpu.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            times.append((end - start) * 1000)

        results["gpu_avg_ms"] = np.mean(times)
        results["gpu_p95_ms"] = np.percentile(times, 95)
        results["gpu_min_ms"] = np.min(times)
        print(f"GPU  - Avg: {results['gpu_avg_ms']:.3f}ms, "
              f"P95: {results['gpu_p95_ms']:.3f}ms, "
              f"Min: {results['gpu_min_ms']:.3f}ms")

    # CPU inference for comparison
    session_cpu = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    input_name = session_cpu.get_inputs()[0].name

    for _ in range(20):
        session_cpu.run(None, {input_name: dummy_input})

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session_cpu.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results["cpu_avg_ms"] = np.mean(times)
    results["cpu_p95_ms"] = np.percentile(times, 95)
    results["cpu_min_ms"] = np.min(times)
    print(f"CPU  - Avg: {results['cpu_avg_ms']:.3f}ms, "
          f"P95: {results['cpu_p95_ms']:.3f}ms, "
          f"Min: {results['cpu_min_ms']:.3f}ms")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/train_config.yaml")
    parser.add_argument("--experiment", type=str, default="posture_v1")
    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    set_all_seeds(seed)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_data(config)

    # MLflow setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(f"file:///{os.path.join(base_dir, 'mlruns')}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"run_{time.strftime('%Y%m%d_%H%M%S')}"):

        mlflow.log_params({
            "seed": seed,
            "n_features": len(feature_cols),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        })

        # --- Train all three models ---

        lr_model, lr_metrics = train_baseline(X_train, y_train, X_val, y_val, seed)
        mlflow.log_metrics({f"lr_{k}": v for k, v in lr_metrics.items() if isinstance(v, float)})

        rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val, seed)
        mlflow.log_metrics({f"rf_{k}": v for k, v in rf_metrics.items() if isinstance(v, float)})

        xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val, config, seed)
        mlflow.log_metrics({f"xgb_{k}": v for k, v in xgb_metrics.items() if isinstance(v, float)})

        # --- Select best model ---

        results = {
            "logistic_regression": (lr_model, lr_metrics),
            "random_forest": (rf_model, rf_metrics),
            "xgboost": (xgb_model, xgb_metrics),
        }

        print("\n" + "=" * 50)
        print("MODEL COMPARISON (Val F1)")
        print("=" * 50)
        for name, (_, metrics) in results.items():
            print(f"  {name:<25} Val F1: {metrics['val_f1']:.4f}")

        best_name = max(results, key=lambda k: results[k][1]["val_f1"])
        best_model = results[best_name][0]
        best_metrics = results[best_name][1]

        print(f"\n>>> Best model: {best_name} (Val F1: {best_metrics['val_f1']:.4f})")

        mlflow.log_param("best_model", best_name)

        # --- Test set evaluation ---

        print("\n" + "=" * 50)
        print(f"TEST SET RESULTS ({best_name})")
        print("=" * 50)

        test_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)

        print(f"Test Acc: {test_acc:.4f}")
        print(f"Test F1:  {test_f1:.4f}")
        print(f"\n{classification_report(y_test, test_pred, target_names=['Bad', 'Good'])}")

        cm = confusion_matrix(y_test, test_pred)
        print(f"Confusion Matrix:")
        print(f"              Pred Bad  Pred Good")
        print(f"  Actual Bad    {cm[0][0]:>5}      {cm[0][1]:>5}")
        print(f"  Actual Good   {cm[1][0]:>5}      {cm[1][1]:>5}")

        mlflow.log_metrics({"test_acc": test_acc, "test_f1": test_f1})

        # --- Overfitting check ---

        print("\n--- Overfitting Check ---")
        train_f1 = best_metrics["train_f1"]
        val_f1 = best_metrics["val_f1"]
        gap = train_f1 - val_f1

        print(f"Train F1: {train_f1:.4f}")
        print(f"Val F1:   {val_f1:.4f}")
        print(f"Test F1:  {test_f1:.4f}")
        print(f"Train-Val gap: {gap:.4f}")

        if gap > 0.05:
            print("WARNING: Possible overfitting (gap > 0.05)")
        else:
            print("OK: No significant overfitting detected")

        # --- Feature importance (XGBoost) ---

        if best_name == "xgboost":
            print("\n--- Feature Importance ---")
            importances = best_model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            for i in sorted_idx:
                bar = "█" * int(importances[i] * 50)
                print(f"  {feature_cols[i]:<25} {importances[i]:.4f} {bar}")

            mlflow.log_dict(
                {feature_cols[i]: float(importances[i]) for i in sorted_idx},
                "feature_importance.json",
            )

        # --- Export to ONNX ---

        models_dir = os.path.join(base_dir, "models")
        onnx_path = os.path.join(models_dir, "posture_model.onnx")

        if best_name == "xgboost":
            # Save XGBoost native format too
            json_path = os.path.join(models_dir, "posture_model.json")
            os.makedirs(models_dir, exist_ok=True)
            best_model.save_model(json_path)
            print(f"\nXGBoost model saved to {json_path}")

        export_to_onnx(best_model, feature_cols, onnx_path)

        # Copy feature names to models dir for inference
        import shutil
        features_src = os.path.join(base_dir, config["data"]["feature_names_path"])
        features_dst = os.path.join(models_dir, "feature_names.json")
        shutil.copy2(features_src, features_dst)
        print(f"Feature names copied to {features_dst}")

        # --- Latency measurement ---

        latency = measure_latency(onnx_path, len(feature_cols))
        mlflow.log_metrics({k: v for k, v in latency.items()})

        # --- Save final summary ---

        summary = {
            "best_model": best_name,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "overfitting_gap": gap,
            "n_features": len(feature_cols),
            "train_size": len(X_train),
            "latency": latency,
            "confusion_matrix": {
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1]),
            },
        }

        if best_name == "xgboost" and "best_params" in best_metrics:
            summary["best_params"] = best_metrics["best_params"]

        summary_path = os.path.join(models_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nTraining summary saved to {summary_path}")

        mlflow.log_artifact(onnx_path)
        mlflow.log_artifact(summary_path)

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")

        print("\n" + "=" * 50)
        print("TRAINING COMPLETE")
        print("=" * 50)


if __name__ == "__main__":
    main()