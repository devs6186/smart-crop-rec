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
from src.data_loader import load_crop_data, get_all_crop_data_paths
from src.crop_params import generate_all_new_crops, CROP_PARAMS
from src.eda import run_full_eda
from src.preprocess import preprocess_pipeline
from src.train import train_and_select_best, save_artifacts
from src.evaluate import plot_learning_curve


def main():
    ensure_dirs()
    print("Smart Crop Recommendation — Full pipeline")
    print("=" * 50)

    # 1) Load base data (Crop_Recommendation.csv + any compatible CSVs)
    paths = get_all_crop_data_paths()
    if not paths:
        print(f"ERROR: No dataset found. Add Crop_Recommendation.csv (or any CSV with columns:")
        print("  N, P, K, temperature, humidity, ph, rainfall, label) to data/raw/")
        print("  e.g. Kaggle: atharvaingle/crop-recommendation-dataset")
        sys.exit(1)
    print(f"Loading from {len(paths)} file(s): {[p.name for p in paths]}")
    df = load_crop_data(merge_all_compatible=True)
    base_crops = sorted(df["label"].unique())
    print(f"  Base data: {len(df)} samples, {len(base_crops)} crops")

    # 2) Generate synthetic data for expanded crops (from crop_params database)
    #    Only add crops NOT already in the base data.
    new_crops_in_db = set(CROP_PARAMS.keys()) - set(base_crops)
    if new_crops_in_db:
        # Match sample count to base dataset's per-crop count for balance
        base_per_crop = len(df) // len(base_crops) if base_crops else 100
        print(f"\nGenerating {len(new_crops_in_db)} additional crops ({base_per_crop} samples each):")
        print(f"  {sorted(new_crops_in_db)}")
        import pandas as pd
        from src.crop_params import generate_crop_samples
        import numpy as np
        rng = np.random.default_rng(42)
        new_frames = []
        for crop in sorted(new_crops_in_db):
            new_frames.append(generate_crop_samples(crop, base_per_crop, rng=rng))
        new_df = pd.concat(new_frames, ignore_index=True)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"  Expanded dataset: {len(df)} samples, {df['label'].nunique()} crops")
    else:
        print(f"  No new crops to add (all {len(CROP_PARAMS)} parameterized crops already in data)")

    # Show class balance summary
    vc = df["label"].value_counts()
    print(f"\n  Class balance: min={vc.min()} max={vc.max()} "
          f"(ratio={vc.max()/vc.min():.1f}x)")

    # 3) EDA
    print("\nRunning EDA ...")
    eda_summary = run_full_eda(df)
    print(f"  Figures saved to reports/figures/")
    print(f"  Class imbalance ratio: {eda_summary['imbalance_ratio']}")

    # 4) Preprocess
    print("\nPreprocessing (stratified split, scaling) ...")
    (
        X_train, X_test, y_train, y_test,
        scaler, label_encoder, feature_names,
    ) = preprocess_pipeline(df)
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"  Classes: {len(label_encoder.classes_)}")

    # 5) Train and select best model
    print("\nTraining and comparing models (DT, RF, KNN, SVM, LR) ...")
    best_model, best_name, metadata, comparison = train_and_select_best(
        X_train, y_train, X_test, y_test, feature_names
    )
    print(f"\n  Best model: {best_name}")
    print(f"  Test accuracy: {metadata['test_accuracy']}, F1-macro: {metadata['test_f1_macro']}")
    print("  Comparison:")
    for row in comparison:
        print(f"    {row['model']}: acc={row['test_accuracy']}, F1={row['test_f1_macro']}")

    # 6) Learning curve for best model (overfitting check)
    print("\nPlotting learning curve ...")
    plot_learning_curve(
        best_model,
        X_train, y_train,
        title=f"Learning curve — {best_name}",
        out_path=FIGURES_DIR / "learning_curve.png",
    )

    # 7) Save artifacts
    save_artifacts(best_model, scaler, label_encoder, metadata, feature_names)
    print("\nArtifacts saved to models/")
    print("Done. Run the app with: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
