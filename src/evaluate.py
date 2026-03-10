import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
import onnxruntime as ort


def load_data_and_model(base_dir):
    """Load test data, feature names, and ONNX model."""

    test_path = os.path.join(base_dir, "data", "processed", "test.csv")
    train_path = os.path.join(base_dir, "data", "processed", "train.csv")
    features_path = os.path.join(base_dir, "models", "feature_names.json")
    model_path = os.path.join(base_dir, "models", "posture_model.onnx")

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    with open(features_path) as f:
        feature_names = json.load(f)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df["label"].values

    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df["label"].values

    print(f"Test samples:  {len(X_test)}")
    print(f"Train samples: {len(X_train)} (for reference distributions)")

    return session, input_name, X_test, y_test, X_train, y_train, feature_names, test_df


def run_predictions(session, input_name, X_test):
    """Run ONNX inference and extract predictions + probabilities."""

    outputs = session.run(None, {input_name: X_test})
    y_pred = outputs[0].flatten()

    # Extract probabilities
    y_prob = None
    if len(outputs) > 1:
        raw_prob = outputs[1]
        if isinstance(raw_prob, list):
            y_prob = np.array([[d.get(0, 0), d.get(1, 0)] for d in raw_prob])
        elif isinstance(raw_prob, np.ndarray):
            y_prob = raw_prob

    return y_pred, y_prob


def print_metrics(y_test, y_pred):
    """Print all classification metrics."""

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Bad Posture", "Good Posture"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                    Predicted Bad   Predicted Good")
    print(f"  Actual Bad            {cm[0][0]:>5}            {cm[0][1]:>5}")
    print(f"  Actual Good           {cm[1][0]:>5}            {cm[1][1]:>5}")
    print()

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"True Negatives  (correctly caught bad posture):  {tn}")
    print(f"False Positives (missed bad posture — DANGEROUS): {fp}")
    print(f"False Negatives (wrongly flagged good posture):   {fn}")
    print(f"True Positives  (correctly confirmed good posture): {tp}")

    return cm


def plot_confusion_matrix(cm, output_dir):
    """Save confusion matrix plot."""

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Bad Posture", "Good Posture"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Posture Classification — Confusion Matrix", fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_roc_curve(y_test, y_prob, output_dir):
    """Save ROC curve plot."""

    if y_prob is None:
        print("No probability outputs — skipping ROC curve")
        return None

    y_prob_positive = y_prob[:, 1] if y_prob.ndim == 2 else y_prob

    fpr, tpr, thresholds = roc_curve(y_test, y_prob_positive)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="blue", linewidth=2, label=f"Model (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    print(f"AUROC: {roc_auc:.4f}")

    return roc_auc


def analyze_errors(y_test, y_pred, y_prob, feature_names, test_df, train_df, output_dir):
    """Deep dive into misclassified examples."""

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    misclassified = y_pred != y_test
    n_errors = misclassified.sum()
    print(f"\nTotal misclassified: {n_errors}/{len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")

    if n_errors == 0:
        print("No errors to analyze!")
        return

    fp_mask = (y_pred == 1) & (y_test == 0)  # Predicted good, actually bad
    fn_mask = (y_pred == 0) & (y_test == 1)  # Predicted bad, actually good

    print(f"\nFalse Positives (said GOOD but actually BAD): {fp_mask.sum()}")
    print("  → These are DANGEROUS: user thinks posture is fine but it isn't")
    print(f"\nFalse Negatives (said BAD but actually GOOD): {fn_mask.sum()}")
    print("  → These are ANNOYING: user gets unnecessary alerts")

    # Get train stats for comparison
    train_good = train_df[train_df["label"] == 1]
    train_bad = train_df[train_df["label"] == 0]

    # Analyze false positives in detail
    if fp_mask.sum() > 0:
        print("\n" + "-" * 50)
        print("FALSE POSITIVES — Detailed Analysis")
        print("-" * 50)
        print("These bad posture frames were incorrectly classified as good.")
        print("They likely represent borderline/mild slouching.\n")

        fp_df = test_df[fp_mask].copy()
        if y_prob is not None:
            fp_probs = y_prob[fp_mask]
            if fp_probs.ndim == 2:
                fp_df["prob_good"] = fp_probs[:, 1]
                fp_df["prob_bad"] = fp_probs[:, 0]

        print(f"{'Feature':<25} {'FP Mean':>10} {'Train Bad':>10} {'Train Good':>10} {'Verdict':>10}")
        print("-" * 70)
        for feat in feature_names:
            fp_mean = fp_df[feat].mean()
            bad_mean = train_bad[feat].mean()
            good_mean = train_good[feat].mean()

            # Is this FP value closer to good or bad training data?
            dist_to_good = abs(fp_mean - good_mean)
            dist_to_bad = abs(fp_mean - bad_mean)
            verdict = "→ GOOD" if dist_to_good < dist_to_bad else "→ BAD"

            print(f"  {feat:<25} {fp_mean:>8.3f}   {bad_mean:>8.3f}   {good_mean:>8.3f}   {verdict}")

        print("\nConclusion: False positives have feature values between good and bad posture.")
        print("They represent mild/borderline slouching that the model can't distinguish.")

    # Analyze false negatives in detail
    if fn_mask.sum() > 0:
        print("\n" + "-" * 50)
        print("FALSE NEGATIVES — Detailed Analysis")
        print("-" * 50)
        print("These good posture frames were incorrectly classified as bad.\n")

        fn_df = test_df[fn_mask].copy()

        print(f"{'Feature':<25} {'FN Mean':>10} {'Train Bad':>10} {'Train Good':>10} {'Verdict':>10}")
        print("-" * 70)
        for feat in feature_names:
            fn_mean = fn_df[feat].mean()
            bad_mean = train_bad[feat].mean()
            good_mean = train_good[feat].mean()

            dist_to_good = abs(fn_mean - good_mean)
            dist_to_bad = abs(fn_mean - bad_mean)
            verdict = "→ GOOD" if dist_to_good < dist_to_bad else "→ BAD"

            print(f"  {feat:<25} {fn_mean:>8.3f}   {bad_mean:>8.3f}   {good_mean:>8.3f}   {verdict}")

    # Save misclassified examples
    error_df = test_df[misclassified].copy()
    error_df["predicted"] = y_pred[misclassified]
    error_df["actual"] = y_test[misclassified]
    error_df["error_type"] = np.where(
        (y_pred[misclassified] == 1) & (y_test[misclassified] == 0),
        "false_positive",
        "false_negative",
    )

    if y_prob is not None:
        error_probs = y_prob[misclassified]
        if error_probs.ndim == 2:
            error_df["prob_bad"] = error_probs[:, 0]
            error_df["prob_good"] = error_probs[:, 1]

    error_path = os.path.join(output_dir, "misclassified_examples.csv")
    error_df.to_csv(error_path, index=False)
    print(f"\nSaved all {n_errors} misclassified examples to {error_path}")

    return error_df


def plot_feature_distributions(feature_names, test_df, y_test, y_pred, output_dir):
    """Plot feature distributions for correct vs misclassified examples."""

    misclassified = y_pred != y_test

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        if i >= len(axes):
            break

        ax = axes[i]

        correct_good = test_df[(y_test == 1) & (~misclassified)][feat]
        correct_bad = test_df[(y_test == 0) & (~misclassified)][feat]
        errors = test_df[misclassified][feat]

        ax.hist(correct_good, bins=30, alpha=0.5, color="green", label="Correct Good", density=True)
        ax.hist(correct_bad, bins=30, alpha=0.5, color="red", label="Correct Bad", density=True)
        if len(errors) > 0:
            ax.hist(errors, bins=15, alpha=0.7, color="orange", label="Misclassified", density=True)

        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=7)

    # Hide empty subplots
    for i in range(len(feature_names), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Feature Distributions: Correct vs Misclassified", fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_confidence_distribution(y_test, y_pred, y_prob, output_dir):
    """Plot prediction confidence for correct vs incorrect predictions."""

    if y_prob is None:
        return

    y_prob_positive = y_prob[:, 1] if y_prob.ndim == 2 else y_prob

    # Confidence = probability of predicted class
    confidence = np.where(y_pred == 1, y_prob_positive, 1 - y_prob_positive)

    correct = y_pred == y_test
    incorrect = ~correct

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(confidence[correct], bins=30, alpha=0.6, color="green", label=f"Correct ({correct.sum()})")
    if incorrect.sum() > 0:
        ax.hist(confidence[incorrect], bins=15, alpha=0.7, color="red", label=f"Incorrect ({incorrect.sum()})")

    ax.set_xlabel("Prediction Confidence", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "confidence_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # Print confidence stats
    if incorrect.sum() > 0:
        print(f"\nConfidence stats:")
        print(f"  Correct   — Mean: {confidence[correct].mean():.3f}, Min: {confidence[correct].min():.3f}")
        print(f"  Incorrect — Mean: {confidence[incorrect].mean():.3f}, Max: {confidence[incorrect].max():.3f}")


def save_summary(y_test, y_pred, roc_auc, output_dir):
    """Save final evaluation summary."""

    summary = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "auroc": float(roc_auc) if roc_auc else None,
        "total_samples": int(len(y_test)),
        "correct": int((y_pred == y_test).sum()),
        "misclassified": int((y_pred != y_test).sum()),
        "false_positives": int(((y_pred == 1) & (y_test == 0)).sum()),
        "false_negatives": int(((y_pred == 0) & (y_test == 1)).sum()),
    }

    path = os.path.join(output_dir, "evaluation_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {path}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Accuracy:   {summary['test_accuracy']:.4f}")
    print(f"  F1 Score:   {summary['test_f1']:.4f}")
    print(f"  Precision:  {summary['test_precision']:.4f}")
    print(f"  Recall:     {summary['test_recall']:.4f}")
    if roc_auc:
        print(f"  AUROC:      {roc_auc:.4f}")
    print(f"  Errors:     {summary['misclassified']}/{summary['total_samples']}")
    print("=" * 60)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("POSTURE MODEL EVALUATION & ERROR ANALYSIS")
    print("=" * 60)

    # Load
    session, input_name, X_test, y_test, X_train, y_train, feature_names, test_df = \
        load_data_and_model(base_dir)

    train_df = pd.read_csv(os.path.join(base_dir, "data", "processed", "train.csv"))

    # Predict
    y_pred, y_prob = run_predictions(session, input_name, X_test)

    # Metrics
    cm = print_metrics(y_test, y_pred)

    # Plots
    print("\n--- Generating Plots ---")
    plot_confusion_matrix(cm, output_dir)
    roc_auc = plot_roc_curve(y_test, y_prob, output_dir)
    plot_feature_distributions(feature_names, test_df, y_test, y_pred, output_dir)
    plot_confidence_distribution(y_test, y_pred, y_prob, output_dir)

    # Error analysis
    analyze_errors(y_test, y_pred, y_prob, feature_names, test_df, train_df, output_dir)

    # Summary
    save_summary(y_test, y_pred, roc_auc, output_dir)


if __name__ == "__main__":
    main()