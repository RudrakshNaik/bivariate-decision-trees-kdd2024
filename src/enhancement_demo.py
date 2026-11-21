"""Evaluate the EXTRA student extensions without changing original results.

Run from repository root:
    python -m src.enhancement_demo             # single split + five-fold CV
    python -m src.enhancement_demo --quick     # single split only

Output files are new (adaptive_*); original comparison tables/plots remain.
"""

from pathlib import Path
import argparse
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.bivariate_tree import BivariateDecisionTree

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
SEED = 42
BIV_KWARGS = dict(max_depth=4, min_samples_split=5,
                   min_impurity_decrease=1e-7, n_orientations=18,
                   random_seed=SEED)


def create_models():
    # Matched-complexity baseline for fairer comparison. Existing
    # unrestricted sklearn baseline remains unchanged in experiment.py.
    return {
        "Sklearn (depth=4)": DecisionTreeClassifier(
            criterion="gini", max_depth=4, min_samples_split=5,
            random_state=SEED,
        ),
        "Fixed-angle bivariate": BivariateDecisionTree(
            adaptive_refinement=False, **BIV_KWARGS,
        ),
        "Adaptive bivariate": BivariateDecisionTree(
            adaptive_refinement=True, refinement_top_k=2,
            refinement_window_degrees=8,
            refinement_step_degrees=2, **BIV_KWARGS,
        ),
    }


def describe_model(name, model, seconds, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0,
    )
    sklearn_model = isinstance(model, DecisionTreeClassifier)
    nodes = model.tree_.node_count if sklearn_model else model.get_num_nodes()
    leaves = model.tree_.n_leaves if sklearn_model else model.get_num_leaves()
    depth = model.tree_.max_depth if sklearn_model else model.get_depth()
    split_info = [] if sklearn_model else model.get_decision_nodes()
    return {
        "model": name,
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "training_time_seconds": float(seconds),
        "depth": int(depth),
        "nodes": int(nodes),
        "leaves": int(leaves),
        "bivariate_nodes": sum(node["node_type"] == "bivariate" for node in split_info),
        "refined_split_nodes": sum(node["search_stage"] == "refined" for node in split_info),
    }


def evaluate_models(X_train, y_train, X_test, y_test):
    models = create_models()
    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        rows.append(describe_model(name, model, elapsed,
                                   X_train, y_train, X_test, y_test))
    return rows, models


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def run_synthetic_demo():
    """Controlled illustration where a fine angle can help a 1-node tree.

    This is synthetic test data with a 27-degree separating direction,
    not a benchmark from the KDD paper or from the original Iris report.
    """
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-2.0, 2.0, size=(180, 2))
    true_angle = np.deg2rad(27.0)
    noisy_score = (np.cos(true_angle) * X[:, 0]
                   + np.sin(true_angle) * X[:, 1]
                   + rng.normal(0.0, 0.08, size=len(X)))
    y = (noisy_score > 0.2).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED,
    )
    rows = []
    models = {}
    for name, adaptive in [("Fixed grid", False), ("Adaptive refinement", True)]:
        model = BivariateDecisionTree(
            max_depth=1, min_samples_split=5, n_orientations=18,
            adaptive_refinement=adaptive, random_seed=SEED,
        ).fit(Xtr, ytr)
        node = model.get_decision_nodes()[0]
        rows.append({
            "model": name,
            "test_accuracy": float(accuracy_score(yte, model.predict(Xte))),
            "root_gini_gain": float(node["impurity_gain"]),
            "root_angle_degrees": float(node["angle_degrees"]),
            "root_search_stage": node["search_stage"],
        })
        models[name] = model
    write_csv(RESULTS / "adaptive_synthetic_demo.csv", rows)
    print("\nSynthetic diagonal-boundary demonstration (not Iris):")
    for row in rows:
        print(f"  {row['model']:21s}: angle={row['root_angle_degrees']:.1f}deg, "
              f"gain={row['root_gini_gain']:.4f}, "
              f"test_accuracy={row['test_accuracy']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharex=True, sharey=True)
    xx = np.linspace(-2.2, 2.2, 100)
    for ax, row in zip(axes, rows):
        ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap="coolwarm", s=22,
                   edgecolors="none", alpha=0.85)
        node = models[row["model"]].get_decision_nodes()[0]
        # The root's learned rule is w1*x1 + w2*x2 + b = 0.
        if abs(node["w2"]) > 1e-9:
            yy = -(node["w1"] * xx + node["b"]) / node["w2"]
            ax.plot(xx, yy, linewidth=2)
        else:
            ax.axvline(-node["b"] / node["w1"], linewidth=2)
        ax.set_title(f"{row['model']}\n{row['root_angle_degrees']:.1f}° / "
                     f"accuracy {row['test_accuracy']:.1%}")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("Synthetic feature 1")
        ax.set_ylabel("Synthetic feature 2")
    fig.suptitle("Synthetic diagonal boundary: coarse vs refined split")
    fig.tight_layout()
    fig.savefig(PLOTS / "adaptive_synthetic_boundaries.png", dpi=160)
    plt.close(fig)
    return rows


def run_demo(include_cv=True):
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, stratify=y, random_state=SEED,
    )

    rows, models = evaluate_models(X_train, y_train, X_test, y_test)
    write_csv(RESULTS / "adaptive_comparison.csv", rows)
    print("\nMatched-depth single split comparison (Iris)")
    for row in rows:
        print(f"  {row['model']:23s} accuracy={row['test_accuracy']:.4f} "
              f"nodes={row['nodes']:2d} time={row['training_time_seconds']:.4f}s "
              f"refined_nodes={row['refined_split_nodes']}")

    # A single held-out example, not a claim of all-prediction explainability.
    adaptive = models["Adaptive bivariate"]
    explanation = adaptive.print_prediction_explanation(
        X_test[0], feature_names=feature_names, class_names=class_names,
    )
    payload = {"sample_features": X_test[0].tolist(),
               "true_class": class_names[int(y_test[0])],
               "explanation": explanation}
    (RESULTS / "adaptive_prediction_explanation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )

    plt.figure(figsize=(8, 4.5))
    labels = [row["model"] for row in rows]
    values = [100 * row["test_accuracy"] for row in rows]
    bars = plt.bar(labels, values)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.7,
                 f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Test accuracy (%)")
    plt.title("Original vs Adaptive Bivariate Tree — Iris, same train/test split")
    plt.ylim(0, 110)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS / "adaptive_accuracy_comparison.png", dpi=160)
    plt.close()

    # Explicitly show a case where the coarse-to-fine search actually
    # selects a refined angle; Iris itself may not select any such node.
    run_synthetic_demo()

    cv_summary = []
    if include_cv:
        cv_rows = []
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for fold_no, (train_idx, test_idx) in enumerate(folds.split(X, y), 1):
            fold_rows, _ = evaluate_models(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx],
            )
            for row in fold_rows:
                cv_rows.append({"fold": fold_no, **row})
            print(f"  Finished CV fold {fold_no}/5")
        write_csv(RESULTS / "adaptive_cv_folds.csv", cv_rows)
        for name in labels:
            subset = [row for row in cv_rows if row["model"] == name]
            summary = {"model": name}
            for metric in ("test_accuracy", "precision_macro", "recall_macro",
                           "f1_macro", "training_time_seconds",
                           "depth", "nodes", "leaves", "bivariate_nodes",
                           "refined_split_nodes"):
                summary[f"mean_{metric}"] = float(np.mean([r[metric] for r in subset]))
                if metric in ("test_accuracy", "precision_macro", "recall_macro", "f1_macro"):
                    # Match the original project's pandas .std() = sample std.
                    summary[f"std_{metric}"] = float(np.std(
                        [r[metric] for r in subset], ddof=1))
            cv_summary.append(summary)
        write_csv(RESULTS / "adaptive_cv_summary.csv", cv_summary)
        print("\n5-fold mean test accuracy (Iris)")
        for summary in cv_summary:
            print(f"  {summary['model']:23s} "
                  f"{summary['mean_test_accuracy']:.4f} "
                  f"+/- {summary['std_test_accuracy']:.4f}")

    return rows, cv_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fixed and adaptive bivariate trees")
    parser.add_argument("--quick", action="store_true", help="skip five-fold CV")
    args = parser.parse_args()
    run_demo(include_cv=not args.quick)
