"""
model.py
--------
Heat Vulnerability Risk Prediction — Modeling Module

Trains two logistic regression classifiers (unconstrained and fairness-constrained)
to predict county-level heat-related mortality risk, and generates predictions from
the CDC equal-weights baseline. Outputs predicted probabilities, binary predictions,
and learned coefficients for all models across stratified 5-fold cross-validation.

Inputs (from data notebook):
    - heat_risk_dataframe.csv : merged national county-level dataframe

Outputs (consumed by analysis.py):
    - outputs/cv_split_indices.pkl
    - outputs/predictions_unconstrained.csv
    - outputs/predictions_constrained.csv
    - outputs/predictions_cdc_baseline.csv
    - outputs/predictions_unconstrained_suppressed.csv
    - outputs/predictions_constrained_suppressed.csv
    - outputs/coefficients_unconstrained.csv
    - outputs/coefficients_constrained.csv
    - outputs/cdc_implied_weights.csv
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

# ── Column name schema (matches data notebook output) ─────────────────────────

FIPS_COL         = "county_fips"        # 5-digit FIPS string
LABEL_COL        = "label"              # binary: 1 = high risk, 0 = low risk
SENSITIVE_COL    = "poverty_quartile"   # 1 = lowest poverty, 4 = highest poverty
HHI_RANKING_COL  = "OVERALL_RANK"      # CDC equal-weights score (0-1)
DATA_QUALITY_COL = "n_zctas_in_county"  # number of ZCTAs averaged per county
SUPPRESSION_COL  = "suppression_flag"   # 1 = suppressed CDC WONDER count, 0 = observed

# ── The 25 HHI percentile-ranked indicator columns ────────────────────────────
# These are the PR_ columns from the HHI dataset after population-weighted
# aggregation to county level. Sensitivity module uses raw PR_ ranks here
# rather than binary flags so all 25 features are on a comparable continuous scale.
#
# CDC implied weight per indicator (NOT equal — varies by module size):
#   Historical Heat and Health Burden (2 indicators): 1/(4*2)  = 0.125  each
#   Sensitivity                       (6 indicators): 1/(4*6)  = 0.0417 each
#   Sociodemographic                 (10 indicators): 1/(4*10) = 0.025  each
#   Natural and Built Environment     (7 indicators): 1/(4*7)  = 0.0357 each

HHI_INDICATORS = [
    # Historical Heat and Health Burden (2)
    "PR_NEHD",      # Number of Extreme Heat Days
    "PR_HRI",       # Heat-Related Illness (EMS activations)

    # Sensitivity (6)
    "PR_CHD",       # Coronary Heart Disease
    "PR_OBS",       # Obesity
    "PR_DIABETES",  # Diabetes
    "PR_COPD",      # Chronic Obstructive Pulmonary Disease
    "PR_ASTHMA",    # Asthma
    "PR_MNTHL",     # Poor Mental Health

    # Sociodemographic (10)
    "PR_UNINSUR",   # Lack of Health Insurance
    "PR_POV",       # Poverty
    "PR_UNEMP",     # Unemployment
    "PR_NOHSDP",    # No High School Diploma
    "PR_ISO",       # Living Alone
    "PR_ELP",       # Speaks English Less than Well
    "PR_DISABL",    # Civilian with a Disability
    "PR_ODW",       # Outdoor Workers
    "PR_AGE65",     # Age 65 and Older
    "PR_AGE5",      # Age 5 and Younger

    # Natural and Built Environment (7)
    "PR_IMPERV",    # Impervious Surfaces
    "PR_TREEC",     # Tree Canopy
    "PR_NOVEH",     # No Vehicle
    "PR_MOBILE",    # Mobile Homes
    "PR_RENT",      # Renters
    "PR_OZONE",     # Ozone
    "PR_PM25",      # PM2.5
]

# CDC implied weights per indicator — saved for analysis.py comparison.
# These reflect two-level equal-weighting: equal across 4 modules, then
# equal within each module. NOT the naive 1/25 = 0.04 per indicator.
CDC_IMPLIED_WEIGHTS = {
    "PR_NEHD":    0.125,  "PR_HRI":     0.125,
    "PR_CHD":     0.0417, "PR_OBS":     0.0417, "PR_DIABETES": 0.0417,
    "PR_COPD":    0.0417, "PR_ASTHMA":  0.0417, "PR_MNTHL":    0.0417,
    "PR_UNINSUR": 0.025,  "PR_POV":     0.025,  "PR_UNEMP":    0.025,
    "PR_NOHSDP":  0.025,  "PR_ISO":     0.025,  "PR_ELP":      0.025,
    "PR_DISABL":  0.025,  "PR_ODW":     0.025,  "PR_AGE65":    0.025,
    "PR_AGE5":    0.025,
    "PR_IMPERV":  0.0357, "PR_TREEC":   0.0357, "PR_NOVEH":    0.0357,
    "PR_MOBILE":  0.0357, "PR_RENT":    0.0357, "PR_OZONE":    0.0357,
    "PR_PM25":    0.0357,
}

# ── Configuration ─────────────────────────────────────────────────────────────

N_FOLDS           = 5
RANDOM_STATE      = 42
CDC_THRESHOLD_PCT = 0.75
OUTPUT_DIR        = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(path: str = "../data/heat_risk_dataframe.csv") -> pd.DataFrame:
    """Load the merged county-level dataframe from the data notebook."""
    df = pd.read_csv(path, dtype={FIPS_COL: str})
    print(f"Loaded {len(df)} total counties from {path}")
    print(f"  Suppressed counties : {df[SUPPRESSION_COL].sum()}")
    print(f"  High-risk (all)     : {df[LABEL_COL].sum()} ({df[LABEL_COL].mean():.1%})")

    before = len(df)
    df = df.dropna(subset=HHI_INDICATORS)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with missing HHI indicator values")

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Confirm all expected columns are present before training."""
    required = (
        [FIPS_COL, LABEL_COL, SENSITIVE_COL,
         HHI_RANKING_COL, DATA_QUALITY_COL, SUPPRESSION_COL]
        + HHI_INDICATORS
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")
    print("Schema validation passed.")


def split_training_prediction(df: pd.DataFrame):
    """
    Split counties into two groups:
      train_df   — observed labels (suppression_flag == 0), used for training + CV
      predict_df — suppressed labels (suppression_flag == 1), predicted but never trained on

    Keeping these separate lets analysis.py distinguish model fairness failures
    (the algorithm is wrong) from data availability failures (the CDC didn't report).
    """
    train_df   = df[df[SUPPRESSION_COL] == 0].copy().reset_index(drop=True)
    predict_df = df[df[SUPPRESSION_COL] == 1].copy().reset_index(drop=True)
    print(f"\nTraining set  : {len(train_df)} counties (observed labels)")
    print(f"Suppressed set: {len(predict_df)} counties (labels unknown)")
    print(f"  High-risk in training set: "
          f"{train_df[LABEL_COL].sum()} ({train_df[LABEL_COL].mean():.1%})")
    return train_df, predict_df


def extract_arrays(df: pd.DataFrame):
    """Extract X, y, A (sensitive attribute), and metadata from a dataframe."""
    X    = df[HHI_INDICATORS].values
    y    = df[LABEL_COL].values
    A    = df[SENSITIVE_COL].values
    meta = df[[FIPS_COL, SENSITIVE_COL, DATA_QUALITY_COL, SUPPRESSION_COL]].copy()
    return X, y, A, meta


# ── Cross-Validation Splits ───────────────────────────────────────────────────

def generate_cv_splits(y: np.ndarray) -> list:
    """
    Generate stratified 5-fold CV splits on observed training counties only.
    Stratification preserves the class balance across folds.
    Saves splits to disk so all models and analysis.py use identical folds.
    """
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(np.zeros(len(y)), y))

    save_path = os.path.join(OUTPUT_DIR, "cv_split_indices.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Saved {N_FOLDS}-fold CV split indices to {save_path}")
    return splits


# ── Helpers ───────────────────────────────────────────────────────────────────

def tune_regularization(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Tune L2 regularization strength (C) using inner 3-fold CV on training data.
    Larger C = less regularization. Grid extended to 1000 to avoid ceiling effects.
    Never touches the held-out test fold.
    """
    param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    base_model = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search   = GridSearchCV(base_model, param_grid, cv=inner_cv,
                            scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    best_C = search.best_params_["C"]
    if best_C == max(param_grid["C"]):
        print(f"  WARNING: Best C hit grid ceiling ({best_C}). Consider extending grid.")
    return best_C


def ensemble_predict_proba(mitigator: ExponentiatedGradient,
                           X: np.ndarray) -> np.ndarray:
    """
    ExponentiatedGradient has no predict_proba method directly.
    Compute probabilities as a weighted average across all estimators
    in the ensemble — this is what the algorithm is designed to produce.
    """
    proba = np.zeros(len(X))
    for est, weight in zip(mitigator.predictors_, mitigator.weights_):
        proba += weight * est.predict_proba(X)[:, 1]
    return proba


def get_best_estimator(mitigator: ExponentiatedGradient):
    """
    Return the highest-weighted estimator from the ExponentiatedGradient
    ensemble for coefficient extraction.
    """
    try:
        weights  = np.array(mitigator.weights_)
        best_idx = np.argmax(weights)
        return mitigator.predictors_[best_idx]
    except Exception:
        return None


# ── Unconstrained Logistic Regression ────────────────────────────────────────

def train_unconstrained(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    splits: list,
    X_suppressed: np.ndarray = None,
    meta_suppressed: pd.DataFrame = None,
):
    """
    Train unconstrained logistic regression across all 5 folds.
    Features standardized within each fold (scaler fit on train, applied to test).
    Also applies each fold's trained model to suppressed counties.

    Uses plain LogisticRegression — predict_proba is called directly on `model`.
    """
    all_predictions  = []
    all_coefficients = []
    all_suppressed   = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n── Unconstrained | Fold {fold + 1}/{N_FOLDS} ──")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit scaler on training fold only, apply to test
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Tune C on training fold only
        best_C = tune_regularization(X_train, y_train)
        print(f"  Best C: {best_C}")

        # Train
        model = LogisticRegression(
            penalty="l2", C=best_C, solver="lbfgs",
            max_iter=1000, random_state=RANDOM_STATE,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)

        # Predict on held-out test fold — plain LogisticRegression has predict_proba
        proba  = model.predict_proba(X_test)[:, 1]
        binary = model.predict(X_test)

        auc = roc_auc_score(y_test, proba)
        f1  = f1_score(y_test, binary)
        print(f"  AUC-ROC: {auc:.4f} | F1: {f1:.4f}")

        fold_meta = meta.iloc[test_idx].copy()
        fold_meta["fold"]           = fold
        fold_meta["prob_high_risk"] = proba
        fold_meta["pred_high_risk"] = binary
        fold_meta["true_label"]     = y_test
        all_predictions.append(fold_meta)

        coef_row = {name: coef for name, coef in zip(HHI_INDICATORS, model.coef_[0])}
        coef_row["fold"]      = fold
        coef_row["intercept"] = model.intercept_[0]
        coef_row["best_C"]    = best_C
        all_coefficients.append(coef_row)

        # Apply this fold's model to suppressed counties — true label unknown
        if X_suppressed is not None and len(X_suppressed) > 0:
            X_sup_scaled = scaler.transform(X_suppressed)
            sup_meta     = meta_suppressed.copy()
            sup_meta["fold"]           = fold
            sup_meta["prob_high_risk"] = model.predict_proba(X_sup_scaled)[:, 1]
            sup_meta["pred_high_risk"] = model.predict(X_sup_scaled)
            sup_meta["true_label"]     = np.nan
            all_suppressed.append(sup_meta)

    predictions_df  = pd.concat(all_predictions, ignore_index=True)
    coefficients_df = pd.DataFrame(all_coefficients)

    predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, "predictions_unconstrained.csv"), index=False)
    coefficients_df.to_csv(
        os.path.join(OUTPUT_DIR, "coefficients_unconstrained.csv"), index=False)

    if all_suppressed:
        sup_df = pd.concat(all_suppressed, ignore_index=True)
        sup_df.to_csv(
            os.path.join(OUTPUT_DIR, "predictions_unconstrained_suppressed.csv"), index=False)
        print(f"  Suppressed predictions: {len(meta_suppressed)} counties × {N_FOLDS} folds")

    print("\nUnconstrained model outputs saved.")
    return predictions_df, coefficients_df


# ── Fairness-Constrained Logistic Regression ─────────────────────────────────

def train_constrained(
    X: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    meta: pd.DataFrame,
    splits: list,
    X_suppressed: np.ndarray = None,
    meta_suppressed: pd.DataFrame = None,
):
    """
    Train fairness-constrained logistic regression using Fairlearn's
    ExponentiatedGradient with EqualizedOdds constraint.

    Sensitive attribute: poverty_quartile (1-4), where quartile 4 is the most
    disadvantaged group (highest poverty). The constraint enforces that TPR
    and FPR are approximately equal across all four quartile groups.

    ExponentiatedGradient produces a weighted ensemble of LogisticRegression
    estimators. It has no direct predict_proba — probabilities are computed
    via ensemble_predict_proba() as a weighted average across estimators.
    """
    all_predictions  = []
    all_coefficients = []
    all_suppressed   = []

    #Fixed AI: Put constrain inside the loop to avoid data leakage across folds. Each fold's model should be trained independently with its own constraint based on the training data of that fold only.
    

    for fold, (train_idx, test_idx) in enumerate(splits):

        print(f"\n── Constrained | Fold {fold + 1}/{N_FOLDS} ──")

        constraint = EqualizedOdds()
        
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        A_train         = A[train_idx]

        # Fit scaler on training fold only, apply to test
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        base_estimator = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        )

        mitigator = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraint,
            eps=0.1,    # max allowed TPR/FPR gap between poverty quartile groups
            max_iter=50,
            nu=1e-6
        )
        mitigator.fit(X_train, y_train, sensitive_features=A_train)

        # ExponentiatedGradient has no predict_proba — use weighted ensemble
        proba  = ensemble_predict_proba(mitigator, X_test)
        binary = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        f1  = f1_score(y_test, binary)
        print(f"  AUC-ROC: {auc:.4f} | F1: {f1:.4f}")

        fold_meta = meta.iloc[test_idx].copy()
        fold_meta["fold"]           = fold
        fold_meta["prob_high_risk"] = proba
        fold_meta["pred_high_risk"] = binary
        fold_meta["true_label"]     = y_test
        all_predictions.append(fold_meta)

        # Extract coefficients from highest-weighted estimator in ensemble
        best_est = get_best_estimator(mitigator)
        if best_est is not None:
            coef_row = {name: coef for name, coef
                        in zip(HHI_INDICATORS, best_est.coef_[0])}
            coef_row["fold"]      = fold
            coef_row["intercept"] = best_est.intercept_[0]
        else:
            coef_row = {name: np.nan for name in HHI_INDICATORS}
            coef_row["fold"]      = fold
            coef_row["intercept"] = np.nan
        all_coefficients.append(coef_row)

        # Apply this fold's ensemble to suppressed counties — true label unknown
        if X_suppressed is not None and len(X_suppressed) > 0:
            X_sup_scaled = scaler.transform(X_suppressed)
            sup_proba    = ensemble_predict_proba(mitigator, X_sup_scaled)
            sup_meta     = meta_suppressed.copy()
            sup_meta["fold"]           = fold
            sup_meta["prob_high_risk"] = sup_proba
            sup_meta["pred_high_risk"] = (sup_proba >= 0.5).astype(int)
            sup_meta["true_label"]     = np.nan
            all_suppressed.append(sup_meta)

    predictions_df  = pd.concat(all_predictions, ignore_index=True)
    coefficients_df = pd.DataFrame(all_coefficients)

    predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, "predictions_constrained.csv"), index=False)
    coefficients_df.to_csv(
        os.path.join(OUTPUT_DIR, "coefficients_constrained.csv"), index=False)

    if all_suppressed:
        sup_df = pd.concat(all_suppressed, ignore_index=True)
        sup_df.to_csv(
            os.path.join(OUTPUT_DIR, "predictions_constrained_suppressed.csv"), index=False)
        print(f"  Suppressed predictions: {len(meta_suppressed)} counties × {N_FOLDS} folds")

    print("\nConstrained model outputs saved.")
    return predictions_df, coefficients_df


# ── CDC Equal-Weights Baseline ────────────────────────────────────────────────

def generate_cdc_baseline(train_df: pd.DataFrame, predict_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate binary predictions from the CDC's OVERALL_RANK equal-weights score.
    Threshold set at 75th percentile of the training set (observed counties only).
    Applied to all counties including suppressed for completeness.

    Important: CDC weighting is NOT 1/25 per indicator. It is 1/4 per module
    then equal within modules — see CDC_IMPLIED_WEIGHTS for true per-indicator
    implied weights used in the coefficient comparison in analysis.py.
    """
    threshold = train_df[HHI_RANKING_COL].quantile(CDC_THRESHOLD_PCT)
    print(f"\nCDC baseline threshold (75th pct of training set): {threshold:.4f}")

    all_df = pd.concat([train_df, predict_df], ignore_index=True)

    cdc_df = all_df[[FIPS_COL, SENSITIVE_COL, DATA_QUALITY_COL,
                     SUPPRESSION_COL, LABEL_COL, HHI_RANKING_COL]].copy()
    cdc_df["prob_high_risk"] = all_df[HHI_RANKING_COL].values
    cdc_df["pred_high_risk"] = (all_df[HHI_RANKING_COL] >= threshold).astype(int).values
    cdc_df["true_label"]     = all_df[LABEL_COL].values

    # Evaluate on observed counties only
    obs = cdc_df[cdc_df[SUPPRESSION_COL] == 0]
    auc = roc_auc_score(obs["true_label"], obs["prob_high_risk"])
    f1  = f1_score(obs["true_label"], obs["pred_high_risk"])
    print(f"CDC baseline (observed counties) — AUC-ROC: {auc:.4f} | F1: {f1:.4f}")

    cdc_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_cdc_baseline.csv"), index=False)
    return cdc_df


def save_cdc_implied_weights() -> None:
    """Save CDC implied per-indicator weights for comparison in analysis.py."""
    weights_df = pd.DataFrame([
        {"indicator": k, "cdc_implied_weight": v}
        for k, v in CDC_IMPLIED_WEIGHTS.items()
    ])
    weights_df.to_csv(os.path.join(OUTPUT_DIR, "cdc_implied_weights.csv"), index=False)
    print("CDC implied weights saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load and validate
    df = load_data("../data/heat_risk_dataframe.csv")
    validate_schema(df)

    # 2. Separate observed from suppressed counties
    train_df, predict_df = split_training_prediction(df)

    # 3. Extract arrays
    X, y, A, meta         = extract_arrays(train_df)
    X_sup, _, _, meta_sup = extract_arrays(predict_df)
    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Class balance        : {y.mean():.1%} high-risk")

    # 4. Generate stratified CV splits on observed counties only
    splits = generate_cv_splits(y)

    # 5. Unconstrained logistic regression
    print("\n" + "="*60)
    print("UNCONSTRAINED LOGISTIC REGRESSION")
    print("="*60)
    train_unconstrained(X, y, meta, splits, X_sup, meta_sup)

    # 6. Fairness-constrained logistic regression
    print("\n" + "="*60)
    print("FAIRNESS-CONSTRAINED LOGISTIC REGRESSION (EqualizedOdds)")
    print("="*60)
    train_constrained(X, y, A, meta, splits, X_sup, meta_sup)

    # 7. CDC equal-weights baseline
    print("\n" + "="*60)
    print("CDC EQUAL-WEIGHTS BASELINE")
    print("="*60)
    generate_cdc_baseline(train_df, predict_df)

    # 8. Save CDC implied weights for analysis comparison
    save_cdc_implied_weights()

    # 9. Summary
    print("\n" + "="*60)
    print("ALL OUTPUTS SAVED TO:", OUTPUT_DIR)
    print("="*60)
    for fname in [
        "cv_split_indices.pkl",
        "predictions_unconstrained.csv",
        "predictions_constrained.csv",
        "predictions_cdc_baseline.csv",
        "predictions_unconstrained_suppressed.csv",
        "predictions_constrained_suppressed.csv",
        "coefficients_unconstrained.csv",
        "coefficients_constrained.csv",
        "cdc_implied_weights.csv",
    ]:
        print(f"  {fname}")
    print("\nHand these files to analysis.py.")


if __name__ == "__main__":
    main()
