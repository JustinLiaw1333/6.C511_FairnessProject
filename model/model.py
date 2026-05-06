"""
model.py
--------
Heat Vulnerability Risk Prediction — Modeling Module

Trains two logistic regression classifiers (unconstrained and fairness-constrained)
to predict county-level heat-related mortality risk, and generates predictions from
the CDC equal-weights baseline. Outputs predicted probabilities, binary predictions,
and learned coefficients for all models across stratified 5-fold cross-validation.

Inputs (from data.py):
    - primary_data.csv : merged national county-level dataframe

Outputs (consumed by analysis.py):
    - outputs/cv_split_indices.pkl       : saved stratified k-fold split indices
    - outputs/predictions_unconstrained.csv
    - outputs/predictions_constrained.csv
    - outputs/predictions_cdc_baseline.csv
    - outputs/coefficients_unconstrained.csv
    - outputs/coefficients_constrained.csv
"""

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

# ── Column name schema (must match data.py output) ───────────────────────────

FIPS_COL         = "fips"
LABEL_COL        = "high_risk"           # binary: 1 = high risk, 0 = low risk
SENSITIVE_COL    = "income_quartile"     # 1 = lowest income, 4 = highest income
HHI_RANKING_COL  = "overall_hhi_ranking" # CDC equal-weights score (0–1)
DATA_QUALITY_COL = "zip_code_count"      # number of ZIP codes averaged per county
SUPPRESSION_COL  = "suppressed"          # boolean flag for suppressed CDC WONDER counts

# The 25 HHI indicator columns — update these names to match data.py exactly
HHI_INDICATORS = [
    "hhi_heat_days",
    "hhi_summer_temp",
    "hhi_urban_heat_island",
    "hhi_air_conditioning_access",
    "hhi_green_space",
    "hhi_impervious_surface",
    "hhi_pct_elderly",
    "hhi_pct_children",
    "hhi_pct_disability",
    "hhi_pct_diabetes",
    "hhi_pct_heart_disease",
    "hhi_pct_mental_illness",
    "hhi_pct_no_insurance",
    "hhi_pct_poverty",
    "hhi_pct_unemployed",
    "hhi_pct_no_hs_diploma",
    "hhi_pct_minority",
    "hhi_pct_non_english",
    "hhi_pct_single_parent",
    "hhi_housing_burden",
    "hhi_pct_mobile_homes",
    "hhi_pct_crowded_housing",
    "hhi_social_vulnerability",
    "hhi_pct_rural",
    "hhi_pct_renters",
]

# ── Configuration ─────────────────────────────────────────────────────────────

N_FOLDS           = 5
RANDOM_STATE      = 42
CDC_THRESHOLD_PCT = 0.75   # top quartile = high risk for CDC baseline
OUTPUT_DIR        = "outputs"

# ── Setup ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str = "primary_data.csv") -> pd.DataFrame:
    """Load the merged county-level dataframe from data.py."""
    df = pd.read_csv(path, dtype={FIPS_COL: str})
    print(f"Loaded {len(df)} counties from {path}")
    print(f"  High-risk counties : {df[LABEL_COL].sum()} ({df[LABEL_COL].mean():.1%})")
    print(f"  Suppressed counties: {df[SUPPRESSION_COL].sum()}")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Confirm all expected columns are present before training."""
    required = [FIPS_COL, LABEL_COL, SENSITIVE_COL,
                HHI_RANKING_COL, DATA_QUALITY_COL, SUPPRESSION_COL] + HHI_INDICATORS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")
    print("Schema validation passed.")


def split_features_labels(df: pd.DataFrame):
    """
    Separate the dataframe into features, labels, sensitive attributes,
    and metadata. Returns numpy arrays for model training.
    """
    X    = df[HHI_INDICATORS].values
    y    = df[LABEL_COL].values
    A    = df[SENSITIVE_COL].values   # sensitive attribute for fairness constraint
    meta = df[[FIPS_COL, SENSITIVE_COL, DATA_QUALITY_COL, SUPPRESSION_COL]].copy()
    return X, y, A, meta


# ── Cross-Validation Splits ───────────────────────────────────────────────────

def generate_cv_splits(y: np.ndarray) -> list:
    """
    Generate stratified 5-fold cross-validation split indices.
    Stratification preserves the ~25/75 class balance in every fold.
    Saves indices to disk so all three models use identical folds.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(np.zeros(len(y)), y))

    save_path = os.path.join(OUTPUT_DIR, "cv_split_indices.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Saved {N_FOLDS}-fold CV split indices to {save_path}")
    return splits


# ── Unconstrained Logistic Regression ────────────────────────────────────────

def tune_regularization(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Tune L2 regularization strength (C parameter) on training folds only.
    Larger C = less regularization. Searches log-scale grid.
    Never touches the test fold.
    """
    param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    base_model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(base_model, param_grid, cv=inner_cv,
                          scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_params_["C"]


def train_unconstrained(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    splits: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train unconstrained logistic regression across all 5 folds.
    Features are standardized within each fold (fit on train, applied to test).

    Returns:
        predictions_df  : FIPS + predicted probabilities + binary predictions per fold
        coefficients_df : learned coefficients per fold
    """
    all_predictions  = []
    all_coefficients = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n── Unconstrained | Fold {fold + 1}/{N_FOLDS} ──")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize — fit on train only, apply to test
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Tune regularization on training fold
        best_C = tune_regularization(X_train, y_train)
        print(f"  Best C: {best_C}")

        # Train final model with best C
        model = LogisticRegression(
            penalty="l2",
            C=best_C,
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        # Generate predictions on held-out test fold
        proba  = model.predict_proba(X_test)[:, 1]
        binary = model.predict(X_test)

        # Log fold performance
        auc = roc_auc_score(y_test, proba)
        f1  = f1_score(y_test, binary)
        print(f"  AUC-ROC: {auc:.4f} | F1: {f1:.4f}")

        # Store predictions with FIPS codes
        fold_meta = meta.iloc[test_idx].copy()
        fold_meta["fold"]             = fold
        fold_meta["prob_high_risk"]   = proba
        fold_meta["pred_high_risk"]   = binary
        fold_meta["true_label"]       = y_test
        all_predictions.append(fold_meta)

        # Store coefficients
        coef_row = {name: coef for name, coef in zip(HHI_INDICATORS, model.coef_[0])}
        coef_row["fold"]      = fold
        coef_row["intercept"] = model.intercept_[0]
        coef_row["best_C"]    = best_C
        all_coefficients.append(coef_row)

    predictions_df  = pd.concat(all_predictions, ignore_index=True)
    coefficients_df = pd.DataFrame(all_coefficients)

    predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, "predictions_unconstrained.csv"), index=False)
    coefficients_df.to_csv(
        os.path.join(OUTPUT_DIR, "coefficients_unconstrained.csv"), index=False)
    print("\nUnconstrained model outputs saved.")
    return predictions_df, coefficients_df


# ── Fairness-Constrained Logistic Regression ─────────────────────────────────

def train_constrained(
    X: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    meta: pd.DataFrame,
    splits: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train fairness-constrained logistic regression using Fairlearn's
    ExponentiatedGradient with an EqualizedOdds constraint.

    EqualizedOdds requires that TPR and FPR be approximately equal across
    income quartile groups. The algorithm iteratively reweights training
    examples from disadvantaged groups until parity is satisfied.

    Sensitive attribute: income_quartile (1 = lowest income, 4 = highest)

    Returns:
        predictions_df  : FIPS + predicted probabilities + binary predictions per fold
        coefficients_df : learned coefficients per fold (averaged across estimators)
    """
    all_predictions  = []
    all_coefficients = []

    constraint = EqualizedOdds()

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n── Constrained | Fold {fold + 1}/{N_FOLDS} ──")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        A_train, A_test = A[train_idx], A[test_idx]

        # Standardize — fit on train only, apply to test
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Base estimator for ExponentiatedGradient
        base_estimator = LogisticRegression(
            penalty="l2",
            C=1.0,           # fixed C for constrained model — fairness constraint
            solver="lbfgs",  # is the primary regularizer here
            max_iter=1000,
            random_state=RANDOM_STATE
        )

        # Fairness-constrained training
        mitigator = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraint,
            eps=0.01,        # fairness tolerance — tighter = fairer but slower
            max_iter=50,
            nu=1e-6
        )
        mitigator.fit(X_train, y_train, sensitive_features=A_train)

        # Generate predictions on held-out test fold
        proba  = mitigator.predict_proba(X_test)[:, 1]
        binary = mitigator.predict(X_test)

        # Log fold performance
        auc = roc_auc_score(y_test, proba)
        f1  = f1_score(y_test, binary)
        print(f"  AUC-ROC: {auc:.4f} | F1: {f1:.4f}")

        # Store predictions with FIPS codes
        fold_meta = meta.iloc[test_idx].copy()
        fold_meta["fold"]           = fold
        fold_meta["prob_high_risk"] = proba
        fold_meta["pred_high_risk"] = binary
        fold_meta["true_label"]     = y_test
        all_predictions.append(fold_meta)

        # Extract coefficients — ExponentiatedGradient produces a weighted
        # ensemble of estimators; we extract from the best (highest weight)
        best_estimator = _get_best_estimator(mitigator)
        if best_estimator is not None:
            coef_row = {name: coef for name, coef
                        in zip(HHI_INDICATORS, best_estimator.coef_[0])}
            coef_row["fold"]      = fold
            coef_row["intercept"] = best_estimator.intercept_[0]
        else:
            coef_row = {name: np.nan for name in HHI_INDICATORS}
            coef_row["fold"]      = fold
            coef_row["intercept"] = np.nan
        all_coefficients.append(coef_row)

    predictions_df  = pd.concat(all_predictions, ignore_index=True)
    coefficients_df = pd.DataFrame(all_coefficients)

    predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, "predictions_constrained.csv"), index=False)
    coefficients_df.to_csv(
        os.path.join(OUTPUT_DIR, "coefficients_constrained.csv"), index=False)
    print("\nConstrained model outputs saved.")
    return predictions_df, coefficients_df


def _get_best_estimator(mitigator: ExponentiatedGradient):
    """
    ExponentiatedGradient stores a list of predictors and weights.
    Return the estimator with the highest weight for coefficient extraction.
    """
    try:
        weights   = np.array(mitigator.weights_)
        best_idx  = np.argmax(weights)
        return mitigator.predictors_[best_idx]
    except Exception:
        return None


# ── CDC Equal-Weights Baseline ────────────────────────────────────────────────

def generate_cdc_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate binary predictions from the CDC's equal-weights HHI ranking.
    Counties with an Overall HHI Ranking above the 75th percentile are
    classified as high-risk. No model training — this is a fixed formula.

    The threshold is computed on the full dataset (no cross-validation needed
    since the CDC score is not fit to any outcome data).
    """
    threshold = df[HHI_RANKING_COL].quantile(CDC_THRESHOLD_PCT)
    print(f"\nCDC baseline threshold (75th percentile): {threshold:.4f}")

    cdc_df = df[[FIPS_COL, SENSITIVE_COL, DATA_QUALITY_COL,
                 SUPPRESSION_COL, LABEL_COL, HHI_RANKING_COL]].copy()

    cdc_df["prob_high_risk"] = df[HHI_RANKING_COL]  # raw score as proxy for probability
    cdc_df["pred_high_risk"] = (df[HHI_RANKING_COL] >= threshold).astype(int)
    cdc_df["true_label"]     = df[LABEL_COL]

    auc = roc_auc_score(cdc_df["true_label"], cdc_df["prob_high_risk"])
    f1  = f1_score(cdc_df["true_label"], cdc_df["pred_high_risk"])
    print(f"CDC baseline — AUC-ROC: {auc:.4f} | F1: {f1:.4f}")

    save_path = os.path.join(OUTPUT_DIR, "predictions_cdc_baseline.csv")
    cdc_df.to_csv(save_path, index=False)
    print(f"CDC baseline predictions saved to {save_path}")
    return cdc_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load and validate data
    df = load_data("primary_data.csv")
    validate_schema(df)

    # 2. Separate features, labels, sensitive attributes, and metadata
    X, y, A, meta = split_features_labels(df)
    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Class balance        : {y.mean():.1%} high-risk")

    # 3. Generate and save CV splits — used by all models and analysis.py
    splits = generate_cv_splits(y)

    # 4. Train unconstrained logistic regression
    print("\n" + "="*60)
    print("UNCONSTRAINED LOGISTIC REGRESSION")
    print("="*60)
    preds_unc, coefs_unc = train_unconstrained(X, y, meta, splits)

    # 5. Train fairness-constrained logistic regression
    print("\n" + "="*60)
    print("FAIRNESS-CONSTRAINED LOGISTIC REGRESSION (EqualizedOdds)")
    print("="*60)
    preds_con, coefs_con = train_constrained(X, y, A, meta, splits)

    # 6. Generate CDC equal-weights baseline predictions
    print("\n" + "="*60)
    print("CDC EQUAL-WEIGHTS BASELINE")
    print("="*60)
    preds_cdc = generate_cdc_baseline(df)

    # 7. Summary
    print("\n" + "="*60)
    print("ALL OUTPUTS SAVED TO:", OUTPUT_DIR)
    print("="*60)
    print("  cv_split_indices.pkl")
    print("  predictions_unconstrained.csv")
    print("  predictions_constrained.csv")
    print("  predictions_cdc_baseline.csv")
    print("  coefficients_unconstrained.csv")
    print("  coefficients_constrained.csv")
    print("\nHand these files to analysis.py.")


if __name__ == "__main__":
    main()
