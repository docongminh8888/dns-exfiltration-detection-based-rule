import pandas as pd

df = pd.read_csv("results/detection_results.csv")

print("========== BASIC INFO ==========")
print("Total rows:", len(df))
print(df[["y_true", "y_pred"]].value_counts().sort_index())

tp = df[(df["y_true"] == 1) & (df["y_pred"] == 1)]
fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]
fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)]
tn = df[(df["y_true"] == 0) & (df["y_pred"] == 0)]

print("\n========== GROUP COUNTS ==========")
print("TP:", len(tp))
print("FN:", len(fn))
print("FP:", len(fp))
print("TN:", len(tn))

print("\n========== FN DESCRIPTION ==========")
cols = [
    "dns_query_count",
    "unique_subdomain_count",
    "unique_ratio",
    "max_query_length",
    "max_label_length",
    "median_label_length",
    "max_entropy",
    "avg_entropy",
    "encoded_label_count",
    "nxdomain_ratio",
    "risk_score"
]

print(fn[cols].describe())

print("\n========== FP DESCRIPTION ==========")
print(fp[cols].describe())

print("\n========== TP TRIGGERED RULES ==========")
print(tp["triggered_rules"].value_counts().head(20))

print("\n========== FP TRIGGERED RULES ==========")
print(fp["triggered_rules"].value_counts().head(20))

print("\n========== FN BY FILE ==========")
print(fn["source_file"].value_counts().head(30))

print("\n========== TP BY FILE ==========")
print(tp["source_file"].value_counts().head(30))

fn.to_csv("results/fn_analysis.csv", index=False, encoding="utf-8-sig")
fp.to_csv("results/fp_analysis.csv", index=False, encoding="utf-8-sig")
tp.to_csv("results/tp_analysis.csv", index=False, encoding="utf-8-sig")

print("\nSaved:")
print("results/fn_analysis.csv")
print("results/fp_analysis.csv")
print("results/tp_analysis.csv")
