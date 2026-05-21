import os
from pathlib import Path

import pandas as pd

from detect_dns_exfiltration import process_file, load_baseline


NORMAL_DIR = "normal_DNS_traffic"
ATTACK_DIR = "attack_DNS_traffic"

BASELINE_FILE = "baseline/dns_normal_baseline.csv"

RESULTS_DIR = "results"
DETECTION_RESULTS_FILE = "results/detection_results.csv"
METRICS_FILE = "results/metrics.csv"


def calculate_metrics(results_df):
    y_true = results_df["y_true"]
    y_pred = results_df["y_pred"]

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total
    }


def process_dataset_folder(files, y_true, baseline_dict, label):
    all_results = []

    for file_path in files:
        print(f"[{label}] Processing: {file_path}")

        try:
            file_results_df = process_file(
                file_path=str(file_path),
                baseline_dict=baseline_dict
            )

            if file_results_df.empty:
                print(f"[{label}] No valid result: {file_path}")
                continue

            file_results_df["y_true"] = int(y_true)

            all_results.append(file_results_df)

        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")

    return all_results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    baseline_dict = load_baseline(BASELINE_FILE)

    normal_files = sorted(Path(NORMAL_DIR).glob("*.tsv"))
    attack_files = sorted(Path(ATTACK_DIR).glob("*.tsv"))

    print("========== DNS EXFILTRATION EVALUATION ==========")
    print("Normal files:", len(normal_files))
    print("attack_DNS_traffic files:", len(attack_files))
    print("Baseline file:", BASELINE_FILE)

    all_results = []

    all_results.extend(
        process_dataset_folder(
            files=normal_files,
            y_true=0,
            baseline_dict=baseline_dict,
            label="NORMAL"
        )
    )

    all_results.extend(
        process_dataset_folder(
            files=attack_files,
            y_true=1,
            baseline_dict=baseline_dict,
            label="ATTACK"
        )
    )

    if not all_results:
        raise ValueError("Không có kết quả nào để đánh giá.")

    results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    preferred_cols = [
        "source_file",
        "src_ip",
        "etld1",
        "hour_window",

        "y_true",
        "y_pred",
        "risk_score",

        "triggered_groups",
        "triggered_rules",

        "dns_query_count",
        "unique_subdomain_count",
        "unique_ratio",

        "nxdomain_count",
        "nxdomain_ratio",

        "max_query_length",
        "max_num_labels",
        "max_label_length",
        "avg_label_length",
        "median_label_length",
        "min_label_length",

        "max_entropy",
        "avg_entropy",
        "encoded_label_count",

        "query_length_threshold",
        "dns_query_count_threshold",
        "unique_subdomain_count_threshold",
        "unique_ratio_threshold",
        "entropy_threshold",

        "baseline_status",
        "source_path"
    ]

    existing_cols = [
        col for col in preferred_cols
        if col in results_df.columns
    ]

    results_df = results_df[existing_cols]

    results_df.to_csv(
        DETECTION_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    metrics = calculate_metrics(results_df)

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== METRICS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved detection results:", DETECTION_RESULTS_FILE)
    print("Saved metrics:", METRICS_FILE)


if __name__ == "__main__":
    main()
