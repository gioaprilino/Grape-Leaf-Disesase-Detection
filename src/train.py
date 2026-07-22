import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from tqdm import tqdm

from . import preprocess
from .features import extract_all_features

CLASS_NAMES = ['Black_Rot', 'ESCA', 'Leaf_Blight', 'Healthy']


def build_dataset(data_dir):
    data_dir = Path(data_dir)
    X, y = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue
        images, labels = preprocess.load_images_from_class_folder(
            class_dir, label_idx
        )
        for img in tqdm(images, desc=f"Extracting features - {class_name}"):
            feat = extract_all_features(img)
            if feat is not None:
                X.append(feat)
                y.append(label_idx)
    return np.array(X), np.array(y)


def train_knn(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]
    }
    grid = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    return grid


def main(data_dir, output_dir):
    print("=" * 60)
    print("MEMBANGUN DATASET FITUR")
    print("=" * 60)
    X, y = build_dataset(data_dir)

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx2, val_idx = next(sss_val.split(X_train, y_train))
    X_train_final, X_val = X_train[train_idx2], X_train[val_idx]
    y_train_final, y_val = y_train[train_idx2], y_train[val_idx]

    print(f"\nSplit: Train={len(X_train_final)}, Val={len(X_val)}, Test={len(X_test)}")

    print("\n" + "=" * 60)
    print("GRID SEARCH KNN")
    print("=" * 60)
    model = train_knn(X_train_final, y_train_final)

    print(f"\nBest parameters: {model.best_params_}")
    print(f"Best CV accuracy: {model.best_score_:.4f}")

    val_score = model.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.4f}")

    test_score = model.score(X_test, y_test)
    print(f"Test accuracy: {test_score:.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / 'knn_model.pkl')
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_test.npy', y_test)
    np.save(output_dir / 'y_pred.npy', model.predict(X_test))

    print(f"\nModel saved to {output_dir / 'knn_model.pkl'}")

    return model, (X_test, y_test)


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/train'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'models'
    main(data_dir, output_dir)
