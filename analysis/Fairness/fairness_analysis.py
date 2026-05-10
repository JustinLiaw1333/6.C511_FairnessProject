"""
Fairness comparison — TPR/FPR table + confusion matrices by poverty quartile.
 
Usage:
    python fairness_comparison.py
 
Outputs:
    fairness_table.csv          — long-form TPR/FPR results
    fairness_summary.csv        — per-model disparity summary
    confusion_matrices.png      — 3x4 grid (model x quartile)
"""
 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
 
# -- 1. Configure -------------------------------------------------------------
 
MODELS = {
    "Unconstrained": "predictions_unconstrained.csv",
    "Constrained":   "predictions_constrained.csv",
    "CDC Baseline":  "predictions_cdc_baseline.csv",
}
 
QUARTILE_COL = "poverty_quartile"
PRED_COL     = "pred_high_risk"
LABEL_COL    = "true_label"
 
QUARTILE_LABELS = {
    1: "Q1 (lowest poverty)",
    2: "Q2",
    3: "Q3",
    4: "Q4 (highest poverty)",
}
 
# -- 2. Compute TPR / FPR -----------------------------------------------------
 
def compute_rates(df):
    rows = []
    for q in sorted(df[QUARTILE_COL].unique()):
        sub = df[df[QUARTILE_COL] == q]
        tp = int(((sub[PRED_COL] == 1) & (sub[LABEL_COL] == 1)).sum())
        fn = int(((sub[PRED_COL] == 0) & (sub[LABEL_COL] == 1)).sum())
        fp = int(((sub[PRED_COL] == 1) & (sub[LABEL_COL] == 0)).sum())
        tn = int(((sub[PRED_COL] == 0) & (sub[LABEL_COL] == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        rows.append({
            "quartile": q, "n": len(sub),
            "positives": tp + fn, "negatives": fp + tn,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "tpr": round(tpr, 4), "fpr": round(fpr, 4),
        })
    return pd.DataFrame(rows)
 
 
dfs, records = {}, []
for model_name, filepath in MODELS.items():
    dfs[model_name] = pd.read_csv(filepath)
    rates = compute_rates(dfs[model_name])
    rates.insert(0, "model", model_name)
    rates["quartile_label"] = rates["quartile"].map(QUARTILE_LABELS)
    records.append(rates)
 
results = pd.concat(records, ignore_index=True)
 
# -- 3. Print TPR/FPR table ---------------------------------------------------
 
display_cols = ["model", "quartile_label", "n", "positives", "negatives",
                "tp", "fn", "fp", "tn", "tpr", "fpr"]
 
print("\n=== Fairness Comparison: TPR and FPR by Poverty Quartile ===\n")
print(results[display_cols].to_string(index=False))
 
# -- 4. Disparity summary -----------------------------------------------------
 
summary_rows = []
for model_name, grp in results.groupby("model", sort=False):
    summary_rows.append({
        "model":     model_name,
        "tpr_q1":    grp.loc[grp["quartile"] == 1, "tpr"].values[0],
        "tpr_q4":    grp.loc[grp["quartile"] == 4, "tpr"].values[0],
        "tpr_range": round(grp["tpr"].max() - grp["tpr"].min(), 4),
        "fpr_q1":    grp.loc[grp["quartile"] == 1, "fpr"].values[0],
        "fpr_q4":    grp.loc[grp["quartile"] == 4, "fpr"].values[0],
        "fpr_range": round(grp["fpr"].max() - grp["fpr"].min(), 4),
    })
 
summary = pd.DataFrame(summary_rows)
# print("\n=== Per-model Disparity Summary ===\n")
# print(summary.to_string(index=False))
 
# -- 5. Confusion matrices: 3 models x 4 quartiles ---------------------------
 
QUARTILES   = sorted(QUARTILE_LABELS.keys())
model_names = list(MODELS.keys())
 
fig, axes = plt.subplots(
    nrows=len(model_names),
    ncols=len(QUARTILES),
    figsize=(14, 9),
)
 
fig.suptitle("Confusion matrices by model and poverty quartile", fontsize=13, y=1.01)
 
for row_idx, model_name in enumerate(model_names):
    df = dfs[model_name]
    for col_idx, q in enumerate(QUARTILES):
        ax = axes[row_idx][col_idx]
        sub = df[df[QUARTILE_COL] == q]
 
        cm = confusion_matrix(sub[LABEL_COL], sub[PRED_COL], labels=[0, 1])
 
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Low risk", "High risk"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
 
        if col_idx == 0:
            ax.set_ylabel(f"{model_name}\n\nActual", fontsize=9)
        else:
            ax.set_ylabel("")
 
        if row_idx == 0:
            ax.set_title(QUARTILE_LABELS[q], fontsize=9)
        else:
            ax.set_title("")
 
        if row_idx < len(model_names) - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Predicted", fontsize=9)
 
        ax.tick_params(labelsize=8)
 
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
print("\nSaved: confusion_matrices.png")
 
# -- 6. Save CSVs -------------------------------------------------------------
 
results[display_cols].to_csv("fairness_table.csv", index=False)
summary.to_csv("fairness_summary.csv", index=False)
print("Saved: fairness_table.csv  |  fairness_summary.csv")
