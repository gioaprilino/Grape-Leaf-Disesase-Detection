import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
import cv2

from . import preprocess
from .features import extract_all_features
from .train import CLASS_NAMES


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_NAMES, output_dict=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    rows = []
    for cls in CLASS_NAMES:
        rows.append([
            cls,
            f"{report[cls]['precision']:.3f}",
            f"{report[cls]['recall']:.3f}",
            f"{report[cls]['f1-score']:.3f}",
            f"{report[cls]['support']:.0f}"
        ])
    acc_val = report['accuracy']
    total_support = sum(report[cls]['support'] for cls in CLASS_NAMES)
    rows.append([
        'Accuracy', '', '', f"{acc_val:.3f}",
        f"{total_support:.0f}"
    ])
    for avg_type in ['macro avg', 'weighted avg']:
        rows.append([
            avg_type,
            f"{report[avg_type]['precision']:.3f}",
            f"{report[avg_type]['recall']:.3f}",
            f"{report[avg_type]['f1-score']:.3f}",
            f"{report[avg_type]['support']:.0f}"
        ])
    table = ax.table(cellText=rows,
                     colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Classification Report', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Classification report saved to {save_path}")


def plot_learning_curves(model, X, y, save_path):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', color='orange', label='Cross-validation score')
    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Learning curves saved to {save_path}")


def plot_cv_scores(model, X, y, save_path):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, 6), scores, color='skyblue', edgecolor='navy')
    plt.axhline(y=np.mean(scores), color='red', linestyle='--',
                label=f'Mean = {np.mean(scores):.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('5-Fold Cross-Validation Scores')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"CV scores plot saved to {save_path}")
    print(f"CV scores: {scores}")
    print(f"Mean CV: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")


def robustness_test(model, X_test, y_test, save_dir):
    print("\n" + "=" * 60)
    print("ROBUSTNESS TEST")
    print("=" * 60)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    noise_levels = [0.01, 0.05, 0.1]
    brightness_factors = [0.5, 0.75, 1.25, 1.5]

    for noise_level in noise_levels:
        X_noisy = X_test.copy()
        for i in range(len(X_noisy)):
            if isinstance(X_noisy[i], np.ndarray):
                X_noisy[i] += np.random.normal(0, noise_level, X_noisy[i].shape)
        y_pred = model.predict(X_noisy)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Noise σ={noise_level}: Accuracy = {acc:.4f}")

    for factor in brightness_factors:
        X_bright = X_test.copy()
        for i in range(len(X_bright)):
            if isinstance(X_bright[i], np.ndarray):
                X_bright[i] = X_bright[i] * factor
        y_pred = model.predict(X_bright)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Brightness ×{factor}: Accuracy = {acc:.4f}")


def evaluate_model(model_path, X_test, y_test, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    plot_confusion_matrix(y_test, y_pred, output_dir / 'confusion_matrix.png')
    plot_classification_report(y_test, y_pred, output_dir / 'classification_report.png')

    return y_pred


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/knn_model.pkl'
    data_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/train'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'results'

    from src.train import build_dataset
    X, y = build_dataset(data_dir)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(sss.split(X, y))
    X_test, y_test = X[test_idx], y[test_idx]

    evaluate_model(model_path, X_test, y_test, output_dir)
