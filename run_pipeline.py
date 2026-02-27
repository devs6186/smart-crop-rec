"""
One-command pipeline: load data → EDA → preprocess → train → save model and artifacts.
Run from project root: python run_pipeline.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_dirs, FIGURES_DIR
from src.data_loader import load_crop_data, get_data_path
from src.eda import run_full_eda
from src.preprocess import preprocess_pipeline
from src.train import train_and_select_best, save_artifacts
from src.evaluate import plot_learning_curve


def main():
    ensure_dirs()
    print("Smart Crop Recommendation — Full pipeline")
    print("=" * 50)

    # 1) Load data
    path = get_data_path()
    if not path.exists():
        print(f"ERROR: No dataset found. Please add Crop_Recommendation.csv to data/raw/")
        print("  Download from Kaggle: atharvaingle/crop-recommendation-dataset")
        sys.exit(1)
    print(f"Loading data from {path} ...")
    df = load_crop_data(path)
    print(f"  Loaded {len(df)} samples, {df['label'].nunique()} crops.")

    # 2) EDA
    print("\nRunning EDA ...")
    eda_summary = run_full_eda(df)
    print(f"  Figures saved to reports/figures/")
    print(f"  Class imbalance ratio: {eda_summary['imbalance_ratio']}")

    # 3) Preprocess
    print("\nPreprocessing (stratified split, scaling) ...")
    (
        X_train, X_test, y_train, y_test,
        scaler, label_encoder, feature_names,
    ) = preprocess_pipeline(df)
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # 4) Train and select best model
    print("\nTraining and comparing models (DT, RF, KNN, SVM, LR) ...")
    best_model, best_name, metadata, comparison = train_and_select_best(
        X_train, y_train, X_test, y_test, feature_names
    )
    print(f"\n  Best model: {best_name}")
    print(f"  Test accuracy: {metadata['test_accuracy']}, F1-macro: {metadata['test_f1_macro']}")
    print("  Comparison:")
    for row in comparison:
        print(f"    {row['model']}: acc={row['test_accuracy']}, F1={row['test_f1_macro']}")

    # 5) Learning curve for best model (overfitting check)
    print("\nPlotting learning curve ...")
    plot_learning_curve(
        best_model,
        X_train, y_train,
        title=f"Learning curve — {best_name}",
        out_path=FIGURES_DIR / "learning_curve.png",
    )

    # 6) Save artifacts
    save_artifacts(best_model, scaler, label_encoder, metadata, feature_names)
    print("\nArtifacts saved to models/")
    print("Done. Run the app with: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
