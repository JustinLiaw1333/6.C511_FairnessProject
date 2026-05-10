"""
ModelTuned.py
-------------
Heat Vulnerability Risk Prediction — Hyperparameter Tuning Module (MICE version)

Key change from previous version:
    MICE imputation was performed in the data notebook to produce defensible labels
    for suppressed counties. This file now trains on ALL 3,108 counties using those
    MICE-imputed labels, rather than excluding suppressed counties. The suppression_flag
    is retained as metadata so analysis can distinguish performance on observed vs
    MICE-imputed counties as a sensitivity check — but it is no longer a training
    exclusion criterion.

    Benefits:
      - 3,108 training counties instead of 1,311
      - ~781 high-risk counties instead of 74 (natural 25/75 class balance)
      - eps sweep now statistically meaningful (~195 high-risk counties per quartile)
      - class_weight less critical given improved balance

Extends model.py with:
  - Parameterized training functions (class_weight, eps, constrained_C, c_grid)
  - Hyperparameter sweep across eps, class_weight, and constrained C
  - Fairness-aware threshold tuning
  - Final model run using best discovered parameters
  - Sensitivity analysis: observed vs MICE-imputed county performance

All outputs are CSVs only. Load into analysis notebook for visualizations.

Inputs:
    - ../data/heat_risk_dataframe.csv

Outputs (all in TunedOutputs/):
    Sweep CSVs:
      sweep_eps.csv
      sweep_class_weight.csv
      sweep_constrained_C.csv
    Threshold tuning:
      optimal_threshold.txt
      threshold_curve.csv
      threshold_group_tpr.csv
    Final model outputs (best params):
      cv_split_indices.pkl
      predictions_unconstrained.csv
      predictions_constrained.csv
      predictions_cdc_baseline.csv
      coefficients_unconstrained.csv
      coefficients_constrained.csv
      cdc_implied_weights.csv
    Sensitivity analysis:
      sensitivity_observed_vs_mice.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

# ── Column name schema ────────────────────────────────────────────────────────

FIPS_COL         = "county_fips"
LABEL_COL        = "label"
SENSITIVE_COL    = "poverty_quartile"
HHI_RANKING_COL  = "OVERALL_RANK"
DATA_QUALITY_COL = "n_zctas_in_county"
SUPPRESSION_COL  = "suppression_flag"   # 0 = observed, 1 = MICE-imputed
                                         # retained as metadata, NOT used for exclusion

HHI_INDICATORS = [
    "PR_NEHD", "PR_HRI",
    "PR_CHD", "PR_OBS", "PR_DIABETES", "PR_COPD", "PR_ASTHMA", "PR_MNTHL",
    "PR_UNINSUR", "PR_POV", "PR_UNEMP", "PR_NOHSDP", "PR_ISO", "PR_ELP",
    "PR_DISABL", "PR_ODW", "PR_AGE65", "PR_AGE5",
    "PR_IMPERV", "PR_TREEC", "PR_NOVEH", "PR_MOBILE", "PR_RENT",
    "PR_OZONE", "PR_PM25",
]

CDC_IMPLIED_WEIGHTS = {
    "PR_NEHD": 0.125,  "PR_HRI": 0.125,
    "PR_CHD": 0.0417,  "PR_OBS": 0.0417,  "PR_DIABETES": 0.0417,
    "PR_COPD": 0.0417, "PR_ASTHMA": 0.0417, "PR_MNTHL": 0.0417,
    "PR_UNINSUR": 0.025, "PR_POV": 0.025, "PR_UNEMP": 0.025,
    "PR_NOHSDP": 0.025,  "PR_ISO": 0.025, "PR_ELP": 0.025,
    "PR_DISABL": 0.025,  "PR_ODW": 0.025, "PR_AGE65": 0.025, "PR_AGE5": 0.025,
    "PR_IMPERV": 0.0357, "PR_TREEC": 0.0357, "PR_NOVEH": 0.0357,
    "PR_MOBILE": 0.0357, "PR_RENT": 0.0357, "PR_OZONE": 0.0357, "PR_PM25": 0.0357,
}

# ── Configuration ─────────────────────────────────────────────────────────────

N_FOLDS           = 5
RANDOM_STATE      = 42
CDC_THRESHOLD_PCT = 0.75
OUTPUT_DIR        = "TunedOutputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(path: str = "../data/heat_risk_dataframe.csv") -> pd.DataFrame:
    """Load county-level dataframe. Suppressed counties are excluded from training."""
    df = pd.read_csv(path, dtype={FIPS_COL: str})
 
    before = len(df)
    df = df.dropna(subset=[LABEL_COL])
    if before != len(df):
        print(f"  Dropped {before - len(df)} counties with missing labels")
 
    print(f"Loaded {len(df)} total counties from {path}")
    print(f"  Observed counties    : {(df[SUPPRESSION_COL] == 0).sum()}")
    print(f"  Suppressed counties  : {(df[SUPPRESSION_COL] == 1).sum()}")
    print(f"  High-risk (observed) : "
          f"{df[df[SUPPRESSION_COL]==0][LABEL_COL].sum()} "
          f"({df[df[SUPPRESSION_COL]==0][LABEL_COL].mean():.1%})")
 
    before = len(df)
    df = df.dropna(subset=HHI_INDICATORS)
    if before != len(df):
        print(f"  Dropped {before - len(df)} rows with missing HHI indicators")
 
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df



def validate_schema(df: pd.DataFrame) -> None:
    required = ([FIPS_COL, LABEL_COL, SENSITIVE_COL,
                 HHI_RANKING_COL, DATA_QUALITY_COL, SUPPRESSION_COL]
                + HHI_INDICATORS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print("Schema validation passed.")

def split_observed_suppressed(df: pd.DataFrame):
    """Split into observed (training) and suppressed (prediction-only) sets."""
    train_df   = df[df[SUPPRESSION_COL] == 0].copy().reset_index(drop=True)
    predict_df = df[df[SUPPRESSION_COL] == 1].copy().reset_index(drop=True)
    print(f"\nTraining set (observed)    : {len(train_df)} counties")
    print(f"Prediction set (suppressed): {len(predict_df)} counties")
    print(f"  High-risk in training    : "
          f"{train_df[LABEL_COL].sum()} ({train_df[LABEL_COL].mean():.1%})")
    return train_df, predict_df



def extract_arrays(df: pd.DataFrame):
    """
    Extract feature matrix X, label vector y, sensitive attribute A,
    and metadata from the full dataframe (all counties, observed + MICE).
    suppression_flag is included in metadata for sensitivity analysis.
    """
    X    = df[HHI_INDICATORS].values
    y    = df[LABEL_COL].values
    A    = df[SENSITIVE_COL].values
    meta = df[[FIPS_COL, SENSITIVE_COL, DATA_QUALITY_COL, SUPPRESSION_COL]].copy()
    return X, y, A, meta


# ── CV Splits ─────────────────────────────────────────────────────────────────

def generate_cv_splits(y: np.ndarray) -> list:
    """
    Stratified 5-fold CV on the full dataset (all 3,108 counties).
    Stratification preserves the ~25/75 class balance from MICE-imputed labels.
    """
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(np.zeros(len(y)), y))
    save_path = os.path.join(OUTPUT_DIR, "cv_split_indices.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Saved {N_FOLDS}-fold CV split indices to {save_path}")
    return splits


# ── Helpers ───────────────────────────────────────────────────────────────────

def tune_regularization(X_train: np.ndarray,
                        y_train: np.ndarray,
                        c_grid: list = None,
                        class_weight = None) -> float:
    """Inner 3-fold CV to tune L2 regularization C. Never touches test fold."""
    if c_grid is None:
        c_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = {"C": c_grid}
    base_model = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=1000,
        random_state=RANDOM_STATE, class_weight=class_weight
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search   = GridSearchCV(base_model, param_grid, cv=inner_cv,
                            scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    best_C = search.best_params_["C"]
    if best_C == max(c_grid):
        print(f"  WARNING: Best C hit ceiling ({best_C}).")
    return best_C


def ensemble_predict_proba(mitigator: ExponentiatedGradient,
                           X: np.ndarray) -> np.ndarray:
    """Weighted average probability across ExponentiatedGradient ensemble."""
    proba = np.zeros(len(X))
    for est, weight in zip(mitigator.predictors_, mitigator.weights_):
        proba += weight * est.predict_proba(X)[:, 1]
    return proba


def get_best_estimator(mitigator: ExponentiatedGradient):
    """Highest-weighted estimator from ensemble for coefficient extraction."""
    try:
        weights  = np.array(mitigator.weights_)
        best_idx = np.argmax(weights)
        return mitigator.predictors_[best_idx]
    except Exception:
        return None


def compute_tpr_disparity(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           A: np.ndarray) -> float:
    """Max TPR - Min TPR across poverty quartile groups."""
    tprs = []
    for g in np.unique(A):
        mask = (A == g)
        y_g  = y_true[mask]
        p_g  = y_pred[mask]
        if y_g.sum() == 0:
            continue
        tprs.append(p_g[y_g == 1].mean())
    return float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0


def fold_summary(preds_df: pd.DataFrame):
    """Average AUC, F1, and TPR disparity across folds."""
    aucs, f1s, disps = [], [], []
 
    for fold in preds_df["fold"].unique():
        fd  = preds_df[preds_df["fold"] == fold]
        y_t = fd["true_label"].values.astype(int)
        p   = fd["prob_high_risk"].values
        b   = fd["pred_high_risk"].values.astype(int)
        A_f = fd[SENSITIVE_COL].values
 
        if len(np.unique(y_t)) < 2:
            continue
 
        aucs.append(roc_auc_score(y_t, p))
        f1s.append(f1_score(y_t, b))
        disps.append(compute_tpr_disparity(y_t, b, A_f))
        
    return (np.mean(aucs), np.std(aucs),
        np.mean(f1s),  np.std(f1s),
        np.mean(disps))



# ── Parameterized Training Functions ─────────────────────────────────────────

def train_unconstrained(
    X, y, meta, splits,
    X_suppressed=None, meta_suppressed=None,
    class_weight="balanced",
    c_grid=None,
    tag="",
):
    """
    Unconstrained logistic regression on observed counties only.
    class_weight='balanced' compensates for 4.5% positive rate.
    Suppressed counties predicted by each fold's model for external validation.
    """
    all_predictions  = []
    all_coefficients = []
    all_suppressed   = []
 
    for fold, (train_idx, test_idx) in enumerate(splits):
        if not tag:
            print(f"\n── Unconstrained | Fold {fold + 1}/{N_FOLDS} ──")
 
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
 
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
 
        best_C = tune_regularization(X_train, y_train,
                                     c_grid=c_grid,
                                     class_weight=class_weight)
        if not tag:
            print(f"  Best C: {best_C}")
 
        model = LogisticRegression(
            penalty="l2", C=best_C, solver="lbfgs",
            max_iter=1000, random_state=RANDOM_STATE,
            class_weight=class_weight
        )
        model.fit(X_train, y_train)
 
        proba  = model.predict_proba(X_test)[:, 1]
        binary = model.predict(X_test)
 
        auc = roc_auc_score(y_test, proba)
        f1  = f1_score(y_test, binary)
        if not tag:
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
 
    suffix = f"_{tag}" if tag else ""
    predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, f"predictions_unconstrained{suffix}.csv"), index=False)
    coefficients_df.to_csv(
        os.path.join(OUTPUT_DIR, f"coefficients_unconstrained{suffix}.csv"), index=False)
 
    if all_suppressed and not tag:
        sup_df = pd.concat(all_suppressed, ignore_index=True)
        sup_df.to_csv(
            os.path.join(OUTPUT_DIR, "predictions_unconstrained_suppressed.csv"), index=False)
        print(f"  Suppressed predictions: {len(meta_suppressed)} counties × {N_FOLDS} folds")
 
    if not tag:
        print("\nUnconstrained model outputs saved.")
    return predictions_df, coefficients_df
 
 
def train_constrained(
    X, y, A, meta, splits,
    X_suppressed=None, meta_suppressed=None,
    eps=0.05,
    constrained_C=1.0,
    class_weight="balanced",
    tag="",
):
    """
    Fairness-constrained logistic regression on observed counties.
    Sensitive attribute: poverty_quartile (1-4).
    EqualizedOdds enforces TPR/FPR parity across quartile groups.
    """
    all_predictions  = []
    all_coefficients = []
    all_suppressed   = []
 
    for fold, (train_idx, test_idx) in enumerate(splits):
        if not tag:
            print(f"\n── Constrained | Fold {fold + 1}/{N_FOLDS} ──")
 
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        A_train         = A[train_idx]
 
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
 
        constraint     = EqualizedOdds()
        base_estimator = LogisticRegression(
            penalty="l2", C=constrained_C, solver="lbfgs",
            max_iter=1000, random_state=RANDOM_STATE,
            class_weight=class_weight
        )
        mitigator = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraint,
            eps=eps,
            max_iter=50,
            nu=1e-6
        )
        mitigator.fit(X_train, y_train, sensitive_features=A_train)
 
        proba  = ensemble_predict_proba(mitigator, X_test)
        binary = (proba >= 0.5).astype(int)
 
        auc = roc_auc_score(y_test, proba)
        f1  = f1_score(y_test, binary)
        if not tag:
            print(f"  AUC-ROC: {auc:.4f} | F1: {f1:.4f}")
 
        fold_meta = meta.iloc[test_idx].copy()
        fold_meta["fold"]           = fold
        fold_meta["prob_high_risk"] = proba
        fold_meta["pred_high_risk"] = binary
        fold_meta["true_label"]     = y_test
        all_predictions.append(fold_meta)
 
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
 
    suffix = f"_{tag}" if tag else ""
    predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, f"predictions_constrained{suffix}.csv"), index=False)
    coefficients_df.to_csv(
        os.path.join(OUTPUT_DIR, f"coefficients_constrained{suffix}.csv"), index=False)
 
    if all_suppressed and not tag:
        sup_df = pd.concat(all_suppressed, ignore_index=True)
        sup_df.to_csv(
            os.path.join(OUTPUT_DIR, "predictions_constrained_suppressed.csv"), index=False)
        print(f"  Suppressed predictions: {len(meta_suppressed)} counties × {N_FOLDS} folds")
 
    if not tag:
        print("\nConstrained model outputs saved.")
    return predictions_df, coefficients_df



# ── CDC Baseline ──────────────────────────────────────────────────────────────

def generate_cdc_baseline(train_df: pd.DataFrame,
                           predict_df: pd.DataFrame) -> pd.DataFrame:
    """
    CDC equal-weights baseline. Threshold from observed counties only.
    Evaluated on observed counties only since those are ground-truth labels.
    """
    threshold = train_df[HHI_RANKING_COL].quantile(CDC_THRESHOLD_PCT)
    print(f"\nCDC baseline threshold (75th pct of observed): {threshold:.4f}")
 
    all_df = pd.concat([train_df, predict_df], ignore_index=True)
    cdc_df = all_df[[FIPS_COL, SENSITIVE_COL, DATA_QUALITY_COL,
                     SUPPRESSION_COL, LABEL_COL, HHI_RANKING_COL]].copy()
    cdc_df["prob_high_risk"] = all_df[HHI_RANKING_COL].values
    cdc_df["pred_high_risk"] = (all_df[HHI_RANKING_COL] >= threshold).astype(int).values
    cdc_df["true_label"]     = all_df[LABEL_COL].values
 
    obs = cdc_df[cdc_df[SUPPRESSION_COL] == 0]
    auc = roc_auc_score(obs["true_label"], obs["prob_high_risk"])
    f1  = f1_score(obs["true_label"], obs["pred_high_risk"])
    print(f"CDC baseline (observed only) — AUC-ROC: {auc:.4f} | F1: {f1:.4f}")
 
    cdc_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_cdc_baseline.csv"), index=False)
    return cdc_df
 
 
def save_cdc_implied_weights():
    pd.DataFrame([{"indicator": k, "cdc_implied_weight": v}
                  for k, v in CDC_IMPLIED_WEIGHTS.items()]).to_csv(
        os.path.join(OUTPUT_DIR, "cdc_implied_weights.csv"), index=False)
    print("CDC implied weights saved.")


# ── Hyperparameter Sweep ──────────────────────────────────────────────────────

def run_hyperparameter_sweep(X, y, A, meta, splits,
                              X_suppressed=None, meta_suppressed=None):
    """
    Sweep three hyperparameters on observed counties only.
    Experiment 1 — eps: fairness tolerance in ExponentiatedGradient
    Experiment 2 — class_weight: compensating for 4.5% positive rate
    Experiment 3 — C in constrained model base estimator
    """
 
    # ── Experiment 1: eps sweep ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("SWEEP 1: eps — fairness-accuracy tradeoff")
    print("="*60)
 
    eps_values  = [0.01, 0.05, 0.10, 0.20]
    eps_results = []
 
    for eps in eps_values:
        print(f"  eps={eps} ...", end=" ", flush=True)
        preds, _ = train_constrained(
            X, y, A, meta, splits,
            X_suppressed, meta_suppressed,
            eps=eps, tag=f"sweep_eps_{eps}"
        )
        auc, auc_std, f1, f1_std, disp = fold_summary(preds)
        eps_results.append({
            "eps": eps, "mean_auc": auc, "std_auc": auc_std,
            "mean_f1": f1, "std_f1": f1_std,
            "mean_tpr_disparity": disp,
        })
        print(f"AUC={auc:.4f} | F1={f1:.4f} | disp={disp:.4f}")
 
    eps_df = pd.DataFrame(eps_results)
    eps_df.to_csv(os.path.join(OUTPUT_DIR, "sweep_eps.csv"), index=False)
 
    fair_mask = eps_df["mean_tpr_disparity"] <= 0.10
    if fair_mask.any():
        best_eps_row = eps_df[fair_mask].loc[eps_df[fair_mask]["mean_f1"].idxmax()]
    else:
        best_eps_row = eps_df.iloc[eps_df["mean_tpr_disparity"].argmin()]
    best_eps = float(best_eps_row["eps"])
    print(f"  → sweep_eps.csv saved | recommended eps={best_eps}")
 
    # ── Experiment 2: class_weight sweep ─────────────────────────────────────
    print("\n" + "="*60)
    print("SWEEP 2: class_weight — compensating for 4.5% positive rate")
    print("="*60)
 
    # Natural imbalance ratio: ~1253/59 ≈ 21x — testing up to 17x
    cw_options = [
        ("balanced",    "balanced"),
        ("{0:1,1:5}",   {0: 1, 1: 5}),
        ("{0:1,1:10}",  {0: 1, 1: 10}),
        ("{0:1,1:17}",  {0: 1, 1: 17}),
    ]
    cw_results = []
 
    for lbl, cw in cw_options:
        print(f"  class_weight={lbl} ...", end=" ", flush=True)
        preds, _ = train_unconstrained(
            X, y, meta, splits,
            X_suppressed, meta_suppressed,
            class_weight=cw,
            tag=f"sweep_cw_{lbl.replace(':', '').replace(',', '_').replace('{', '').replace('}', '')}"
        )
        auc, auc_std, f1, f1_std, disp = fold_summary(preds)
        cw_results.append({
            "class_weight": lbl, "mean_auc": auc, "std_auc": auc_std,
            "mean_f1": f1, "std_f1": f1_std,
            "mean_tpr_disparity": disp,
        })
        print(f"AUC={auc:.4f} | F1={f1:.4f} | disp={disp:.4f}")
 
    cw_df = pd.DataFrame(cw_results)
    cw_df.to_csv(os.path.join(OUTPUT_DIR, "sweep_class_weight.csv"), index=False)
 
    best_cw_idx   = cw_df["mean_f1"].idxmax()
    best_cw       = cw_options[best_cw_idx][1]
    best_cw_label = cw_options[best_cw_idx][0]
    print(f"  → sweep_class_weight.csv saved | best class_weight={best_cw_label}")
 
    # ── Experiment 3: constrained C sweep ────────────────────────────────────
    print("\n" + "="*60)
    print("SWEEP 3: C in constrained model base estimator")
    print("="*60)
 
    c_values  = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    c_results = []
 
    for c_val in c_values:
        print(f"  C={c_val} ...", end=" ", flush=True)
        preds, _ = train_constrained(
            X, y, A, meta, splits,
            X_suppressed, meta_suppressed,
            constrained_C=c_val,
            eps=best_eps,
            tag=f"sweep_constC_{c_val}"
        )
        auc, auc_std, f1, f1_std, disp = fold_summary(preds)
        c_results.append({
            "constrained_C": c_val, "mean_auc": auc, "std_auc": auc_std,
            "mean_f1": f1, "std_f1": f1_std,
            "mean_tpr_disparity": disp,
        })
        print(f"AUC={auc:.4f} | F1={f1:.4f} | disp={disp:.4f}")
 
    c_df = pd.DataFrame(c_results)
    c_df.to_csv(os.path.join(OUTPUT_DIR, "sweep_constrained_C.csv"), index=False)
 
    fair_mask_c = c_df["mean_tpr_disparity"] <= 0.15
    if fair_mask_c.any():
        best_c_row = c_df[fair_mask_c].loc[c_df[fair_mask_c]["mean_f1"].idxmax()]
    else:
        best_c_row = c_df.iloc[c_df["mean_f1"].idxmax()]
    best_constrained_C = float(best_c_row["constrained_C"])
    print(f"  → sweep_constrained_C.csv saved | best C={best_constrained_C}")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SWEEP COMPLETE — BEST PARAMETERS FOUND")
    print("="*60)
    print(f"  best_eps           = {best_eps}")
    print(f"  best_class_weight  = {best_cw_label}")
    print(f"  best_constrained_C = {best_constrained_C}")
 
    return best_eps, best_cw, best_cw_label, best_constrained_C
 
 
# ── Threshold Tuning ──────────────────────────────────────────────────────────
 
def tune_threshold(predictions_df: pd.DataFrame,
                   max_disparity: float = 0.10) -> float:
    """
    Find the classification threshold that maximizes F1 subject to
    TPR disparity <= max_disparity across poverty quartile groups.
    Saves optimal_threshold.txt, threshold_curve.csv, threshold_group_tpr.csv.
    All plotting done in analysis notebook.
    """
    y_true = predictions_df["true_label"].values.astype(int)
    proba  = predictions_df["prob_high_risk"].values
    A      = predictions_df[SENSITIVE_COL].values
 
    thresholds  = np.linspace(0.01, 0.99, 200)
    f1_scores, disparities, precisions, recalls = [], [], [], []
 
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        disparities.append(compute_tpr_disparity(y_true, y_pred, A))
        precisions.append(prec)
        recalls.append(rec)
 
    f1_scores   = np.array(f1_scores)
    disparities = np.array(disparities)
    precisions  = np.array(precisions)
    recalls     = np.array(recalls)
 
    naive_idx       = np.argmax(f1_scores)
    naive_threshold = thresholds[naive_idx]
 
    fair_mask = disparities <= max_disparity
    if fair_mask.any():
        fair_idx       = np.argmax(np.where(fair_mask, f1_scores, -1))
        fair_threshold = thresholds[fair_idx]
    else:
        fair_idx       = np.argmin(disparities)
        fair_threshold = thresholds[fair_idx]
        print(f"  WARNING: No threshold achieves disparity <= {max_disparity:.2f}. "
              f"Using minimum disparity threshold.")
 
    fair_f1    = f1_scores[fair_idx]
    fair_disp  = disparities[fair_idx]
    naive_f1   = f1_scores[naive_idx]
    naive_disp = disparities[naive_idx]
 
    y_pred_fair = (proba >= fair_threshold).astype(int)
    groups      = sorted(np.unique(A))
    group_tprs  = {}
    for g in groups:
        mask = (A == g)
        y_g  = y_true[mask]
        p_g  = y_pred_fair[mask]
        group_tprs[g] = p_g[y_g == 1].mean() if y_g.sum() > 0 else 0.0
 
    print(f"\n{'─'*50}")
    print("THRESHOLD TUNING RESULTS")
    print(f"{'─'*50}")
    print(f"  Naive optimal : threshold={naive_threshold:.3f} | "
          f"F1={naive_f1:.4f} | disp={naive_disp:.4f}")
    print(f"  Fair optimal  : threshold={fair_threshold:.3f} | "
          f"F1={fair_f1:.4f} | disp={fair_disp:.4f}")
    print(f"  TPR by poverty quartile at fair threshold:")
    for g, tpr in group_tprs.items():
        lbl = "lowest poverty" if g == 1 else "highest poverty" if g == 4 else f"Q{g}"
        print(f"    Q{g} ({lbl}): {tpr:.4f}")
 
    with open(os.path.join(OUTPUT_DIR, "optimal_threshold.txt"), "w") as f:
        f.write(f"fairness_aware_threshold={fair_threshold:.6f}\n")
        f.write(f"naive_threshold={naive_threshold:.6f}\n")
        f.write(f"max_disparity_constraint={max_disparity:.4f}\n")
        f.write(f"fair_f1={fair_f1:.6f}\n")
        f.write(f"fair_tpr_disparity={fair_disp:.6f}\n")
        f.write(f"naive_f1={naive_f1:.6f}\n")
        f.write(f"naive_tpr_disparity={naive_disp:.6f}\n")
 
    pd.DataFrame({
        "threshold": thresholds,
        "f1":        f1_scores,
        "disparity": disparities,
        "precision": precisions,
        "recall":    recalls,
    }).to_csv(os.path.join(OUTPUT_DIR, "threshold_curve.csv"), index=False)
 
    pd.DataFrame([
        {"poverty_quartile": g, "tpr_at_fair_threshold": tpr}
        for g, tpr in group_tprs.items()
    ]).to_csv(os.path.join(OUTPUT_DIR, "threshold_group_tpr.csv"), index=False)
 
    print(f"  → optimal_threshold.txt + threshold_curve.csv + "
          f"threshold_group_tpr.csv saved")
    return float(fair_threshold)



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load and validate
    df = load_data("../data/heat_risk_dataframe.csv")
    validate_schema(df)
 
    # 2. Split observed (training) from suppressed (prediction-only)
    train_df, predict_df = split_observed_suppressed(df)
 
    # 3. Extract arrays from observed counties only
    X, y, A, meta         = extract_arrays(train_df)
    X_sup, _, _, meta_sup = extract_arrays(predict_df)
    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Class balance        : {y.mean():.1%} high-risk in training set")
 
    # 4. CV splits on observed counties only
    splits = generate_cv_splits(y)
 
    # 5. Hyperparameter sweep
    print("\n" + "="*60)
    print("HYPERPARAMETER SWEEP")
    print("="*60)
    best_eps, best_cw, best_cw_label, best_constrained_C = run_hyperparameter_sweep(
        X, y, A, meta, splits, X_sup, meta_sup
    )
 
    # 6. Final unconstrained model
    print("\n" + "="*60)
    print(f"FINAL UNCONSTRAINED MODEL (class_weight={best_cw_label})")
    print("="*60)
    preds_unc, _ = train_unconstrained(
        X, y, meta, splits, X_sup, meta_sup,
        class_weight=best_cw
    )
 
    # 7. Threshold tuning
    print("\n" + "="*60)
    print("THRESHOLD TUNING")
    print("="*60)
    optimal_threshold = tune_threshold(preds_unc, max_disparity=0.10)
 
    # 8. Final constrained model
    print("\n" + "="*60)
    print(f"FINAL CONSTRAINED MODEL (eps={best_eps}, C={best_constrained_C})")
    print("="*60)
    train_constrained(
        X, y, A, meta, splits, X_sup, meta_sup,
        eps=best_eps,
        constrained_C=best_constrained_C,
        class_weight=best_cw
    )
 
    # 9. CDC baseline
    print("\n" + "="*60)
    print("CDC EQUAL-WEIGHTS BASELINE")
    print("="*60)
    generate_cdc_baseline(train_df, predict_df)
    save_cdc_implied_weights()
 
    # 10. Summary
    print("\n" + "="*60)
    print("ALL OUTPUTS SAVED TO: TunedOutputs_Case1/")
    print("="*60)
    print(f"  Best eps              : {best_eps}")
    print(f"  Best class_weight     : {best_cw_label}")
    print(f"  Best constrained C    : {best_constrained_C}")
    print(f"  Optimal threshold     : {optimal_threshold:.4f}")
    print("\n  Sweep CSVs:")
    for f in ["sweep_eps.csv", "sweep_class_weight.csv", "sweep_constrained_C.csv"]:
        print(f"    {f}")
    print("\n  Threshold tuning:")
    for f in ["optimal_threshold.txt", "threshold_curve.csv", "threshold_group_tpr.csv"]:
        print(f"    {f}")
    print("\n  Final model outputs:")
    for f in ["predictions_unconstrained.csv", "predictions_unconstrained_suppressed.csv",
              "predictions_constrained.csv", "predictions_constrained_suppressed.csv",
              "predictions_cdc_baseline.csv", "coefficients_unconstrained.csv",
              "coefficients_constrained.csv", "cdc_implied_weights.csv",
              "cv_split_indices.pkl"]:
        print(f"    {f}")
    print("\nLoad all CSVs into your analysis notebook for plotting.")
 
 
if __name__ == "__main__":
    main()
