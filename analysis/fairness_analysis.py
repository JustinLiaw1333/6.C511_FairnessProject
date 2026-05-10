
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
 
 
MODELS = {
    "Unconstrained": "/Users/shreeya/Downloads/6.C511_FairnessProject/model/TunedOutputs_SMOTE/predictions_unconstrained.csv",
    "Constrained":   "/Users/shreeya/Downloads/6.C511_FairnessProject/model/TunedOutputs_SMOTE/predictions_constrained.csv",
    "CDC Baseline":  "/Users/shreeya/Downloads/6.C511_FairnessProject/model/TunedOutputs_SMOTE/predictions_cdc_baseline.csv",
}
 
INCOME_COL = "income_group"
PRED_COL     = "pred_high_risk"
LABEL_COL    = "true_label"
 
GROUP_LABELS = {
    0: "Non-Persistent Poverty Counties",
    1: "Persistent Poverty Counties",
}
 
# -- 2. Compute TPR / FPR -----------------------------------------------------
 
def compute_rates(df):
    rows = []
    for q in sorted(df[INCOME_COL].unique()):
        sub = df[df[INCOME_COL] == q]
        tp = int(((sub[PRED_COL] == 1) & (sub[LABEL_COL] == 1)).sum())
        fn = int(((sub[PRED_COL] == 0) & (sub[LABEL_COL] == 1)).sum())
        fp = int(((sub[PRED_COL] == 1) & (sub[LABEL_COL] == 0)).sum())
        tn = int(((sub[PRED_COL] == 0) & (sub[LABEL_COL] == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        rows.append({
            "group": q, "n": len(sub),
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
    rates["group_label"] = rates["group"].map(GROUP_LABELS)
    records.append(rates)
 
results = pd.concat(records, ignore_index=True)
 
# -- 3. Print TPR/FPR table ---------------------------------------------------
 
display_cols = ["model", "group_label", "n", "positives", "negatives",
                "tp", "fn", "fp", "tn", "tpr", "fpr"]
 
print("\n=== Fairness Comparison: TPR and FPR by Poverty Quartile ===\n")
print(results[display_cols].to_string(index=False))
 
# -- 4. Disparity summary -----------------------------------------------------
 
summary_rows = []
for model_name, grp in results.groupby("model", sort=False):
    summary_rows.append({
        "model":     model_name,
        "tpr_g0":    grp.loc[grp["group"] == 0, "tpr"].values[0],
        "tpr_g1":    grp.loc[grp["group"] == 1, "tpr"].values[0],
        "tpr_range": round(grp["tpr"].max() - grp["tpr"].min(), 1),
        "fpr_g0":    grp.loc[grp["group"] == 0, "fpr"].values[0],
        "fpr_g1":    grp.loc[grp["group"] == 1, "fpr"].values[0],
        "fpr_range": round(grp["fpr"].max() - grp["fpr"].min(), 1),
    })
 
summary = pd.DataFrame(summary_rows)
# print("\n=== Per-model Disparity Summary ===\n")
# print(summary.to_string(index=False))
 
# -- 5. Confusion matrices: 3 models x 4 quartiles ---------------------------
 
GROUPS   = sorted(GROUP_LABELS.keys())
model_names = list(MODELS.keys())
 
fig, axes = plt.subplots(
    nrows=len(GROUPS),
    ncols=len(model_names),
    figsize=(14, 9),
)
 
fig.suptitle("Confusion matrices by model and poverty groups", fontsize=13, y=1.01)
 
for row_idx, model_name in enumerate(model_names):
    df = dfs[model_name]
    for row_idx, q in enumerate(GROUPS):                # outer = rows = groups
        for col_idx, model_name in enumerate(model_names):  # inner = cols = models
            ax = axes[row_idx][col_idx]
            df = dfs[model_name]
            sub = df[df[INCOME_COL] == q]

            cm = confusion_matrix(sub[LABEL_COL], sub[PRED_COL], labels=[0, 1])

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Low risk", "High risk"],
            )
            disp.plot(ax=ax, colorbar=False, cmap="Blues")

            # Column headers = model names (top row only)
            if row_idx == 0:
                ax.set_title(model_name, fontsize=9)
            else:
                ax.set_title("")

            # Row labels = income group (left column only)
            if col_idx == 0:
                ax.set_ylabel(f"{GROUP_LABELS[q]}\n\nActual", fontsize=9)
            else:
                ax.set_ylabel("")

            # X-axis label on bottom row only
            if row_idx == len(GROUPS) - 1:
                ax.set_xlabel("Predicted", fontsize=9)
            else:
                ax.set_xlabel("")

            ax.tick_params(labelsize=8)
    # for col_idx, q in enumerate(GROUPS):
    #     ax = axes[col_idx][row_idx]
    #     sub = df[df[INCOME_COL] == q]
 
    #     cm = confusion_matrix(sub[LABEL_COL], sub[PRED_COL], labels=[0, 1])
 
    #     disp = ConfusionMatrixDisplay(
    #         confusion_matrix=cm,
    #         display_labels=["Low risk", "High risk"],
    #     )
    #     disp.plot(ax=ax, colorbar=False, cmap="Blues")
 
    #     if row_idx == 0:
    #         ax.set_title(model_name, fontsize=9)
    #     else:
    #         ax.set_title("")

    #     # Row labels = income group (left column only)
    #     if col_idx == 0:
    #         ax.set_ylabel(f"{GROUP_LABELS[q]}\n\nActual", fontsize=9)
    #     else:
    #         ax.set_ylabel("")

    #     # X-axis label on bottom row only
    #     if row_idx < len(GROUPS) - 1:
    #         ax.set_xlabel("")
    #     else:
    #         ax.set_xlabel("Predicted", fontsize=9)
 
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
print("\nSaved: confusion_matrices.png")
 
# -- 6. Save CSVs -------------------------------------------------------------
 
results[display_cols].to_csv("fairness_table.csv", index=False)
summary.to_csv("fairness_summary.csv", index=False)
print("Saved: fairness_table.csv  |  fairness_summary.csv")
