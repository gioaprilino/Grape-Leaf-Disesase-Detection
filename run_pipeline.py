#!/usr/bin/env python3
"""
Pipeline lengkap: Training → Evaluasi TD + KNN untuk deteksi penyakit daun anggur.

Penggunaan:
  python run_pipeline.py                      # menggunakan default
  python run_pipeline.py data/train models results
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib

from src.train import build_dataset, train_knn, CLASS_NAMES
from src.evaluate import (
    evaluate_model,
    plot_learning_curves,
    plot_cv_scores,
    robustness_test
)


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/train')
    model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('models')
    result_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('results')

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TD + KNN - Deteksi Penyakit Daun Anggur")
    print("  Pipeline Training & Evaluasi")
    print("=" * 60)

    print("\n[1/5] Membangun dataset fitur...")
    X, y = build_dataset(data_dir)
    print(f"  Total sampel: {len(X)}")
    print(f"  Dimensi fitur: {X.shape[1]}")

    print("\n[2/5] Split data (70:15:15)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx2, val_idx = next(sss_val.split(X_train, y_train))
    X_train_final, X_val = X_train[train_idx2], X_train[val_idx]
    y_train_final, y_val = y_train[train_idx2], y_train[val_idx]

    split_info = {
        'train': len(y_train_final), 'val': len(y_val), 'test': len(y_test)
    }
    print(f"  Train: {split_info['train']}, Val: {split_info['val']}, "
          f"Test: {split_info['test']}")

    print("\n[3/5] GridSearchCV untuk KNN...")
    model = train_knn(X_train_final, y_train_final)
    print(f"  Best params: {model.best_params_}")
    print(f"  Best CV accuracy: {model.best_score_:.4f}")
    print(f"  Validation accuracy: {model.score(X_val, y_val):.4f}")

    model_path = model_dir / 'knn_model.pkl'
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")

    print("\n[4/5] Evaluasi model pada test set...")
    y_pred = model.predict(X_test)
    np.save(result_dir / 'y_test.npy', y_test)
    np.save(result_dir / 'y_pred.npy', y_pred)
    evaluate_model(model_path, X_test, y_test, result_dir)

    print("\n[5/5] Visualisasi tambahan...")
    scaler = model.best_estimator_.named_steps['scaler']
    X_scaled = scaler.transform(X)
    cv_model = model.best_estimator_.named_steps['knn']
    plot_learning_curves(cv_model, X_scaled, y, result_dir / 'learning_curves.png')
    plot_cv_scores(model, X, y, result_dir / 'cv_scores.png')

    print("\n" + "=" * 60)
    print("  PIPELINE SELESAI")
    print(f"  Hasil: {result_dir}/")
    print(f"  Model: {model_path}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
