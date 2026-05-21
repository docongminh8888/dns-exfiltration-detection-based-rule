import os
import math
import glob
from collections import Counter

import pandas as pd
import numpy as np
import tldextract


NORMAL_DIR = "normal_DNS_traffic/*.tsv"
OUTPUT_FILE = "baseline/dns_normal_baseline.csv"

BASELINE_DAYS = None
MIN_SAMPLES = 5

# =========================
# WINDOW CONFIGURATION
# =========================
# Thay đổi WINDOW_MINUTES để chạy benchmark với kích thước cửa sổ khác nhau.
# Giá trị này PHẢI đồng nhất giữa dns_baseline.py và detect_dns_exfiltration.py.
# Khi đổi window, các ngưỡng count trong detect_dns_exfiltration.py sẽ tự
# scale tương ứng để bảo đảm so sánh công bằng.
WINDOW_MINUTES = 5
TIME_WINDOW = f"{WINDOW_MINUTES}min"


def shannon_entropy(text):
    if not text:
        return 0.0

    counter = Counter(text)
    length = len(text)
    entropy = 0.0

    for count in counter.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def extract_domain_parts(query):
    query = str(query).strip(".").lower()
    ext = tldextract.extract(query)

    if ext.domain and ext.suffix:
        etld1 = f"{ext.domain}.{ext.suffix}"
    else:
        etld1 = query

    subdomain_labels = ext.subdomain.split(".") if ext.subdomain else []

    return etld1, subdomain_labels


def longest_label_entropy(labels):
    if not labels:
        return 0.0

    longest_label = max(labels, key=len)
    return shannon_entropy(longest_label)


def build_baseline_from_group(df, group_cols, baseline_level):
    print(f"[+] Building {baseline_level} baseline - R01 query length...", flush=True)

    r01_group = (
        df
        .groupby(group_cols, sort=False)
        .agg(
            query_length_mean=("query_length", "mean"),
            query_length_std=("query_length", "std"),
            query_length_count=("query_length", "count")
        )
        .reset_index()
    )

    r01_group["query_length_std"] = r01_group["query_length_std"].fillna(0)

    r01_group["query_length_threshold"] = (
        r01_group["query_length_mean"] + 3 * r01_group["query_length_std"]
    )

    r01_group["query_length_threshold"] = (
        r01_group["query_length_threshold"].apply(lambda x: max(180, x))
    )

    print(f"[+] {baseline_level} R01 done. Groups: {len(r01_group)}", flush=True)
    print(f"[+] Filtering groups with MIN_SAMPLES >= {MIN_SAMPLES}...", flush=True)

    valid_groups = r01_group[
        r01_group["query_length_count"] >= MIN_SAMPLES
    ][group_cols]

    df = df.merge(valid_groups, on=group_cols, how="inner")

    r01_group = r01_group[
        r01_group["query_length_count"] >= MIN_SAMPLES
    ].reset_index(drop=True)

    print(f"[+] Rows after MIN_SAMPLES filter: {len(df)}", flush=True)

    if df.empty:
        baseline = r01_group.copy()
        baseline["baseline_level"] = baseline_level
        return baseline

    print(f"[+] Building {baseline_level} window features...", flush=True)

    window_df = (
        df
        .groupby(group_cols + ["hour_window"], sort=False)
        .agg(
            dns_query_count=("query_clean", "count"),
            unique_subdomain_count=("subdomain", "nunique"),
            avg_entropy=("subdomain_entropy", "mean"),
            max_entropy=("subdomain_entropy", "max"),
            avg_label_length=("avg_label_length", "mean"),
            median_label_length=("median_label_length", "median"),
            min_label_length=("min_label_length", "min")
        )
        .reset_index()
    )

    window_df["unique_ratio"] = (
        window_df["unique_subdomain_count"] / window_df["dns_query_count"]
    )

    window_df["unique_ratio"] = (
        window_df["unique_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    print(f"[+] {baseline_level} window rows: {len(window_df)}", flush=True)
    print(f"[+] Building {baseline_level} threshold statistics...", flush=True)

    window_group = (
        window_df
        .groupby(group_cols, sort=False)
        .agg(
            dns_query_count_mean=("dns_query_count", "mean"),
            dns_query_count_std=("dns_query_count", "std"),
            unique_subdomain_count_mean=("unique_subdomain_count", "mean"),
            unique_subdomain_count_std=("unique_subdomain_count", "std"),
            unique_ratio_mean=("unique_ratio", "mean"),
            unique_ratio_std=("unique_ratio", "std"),
            entropy_mean=("avg_entropy", "mean"),
            entropy_std=("avg_entropy", "std"),
            avg_label_length_mean=("avg_label_length", "mean"),
            avg_label_length_std=("avg_label_length", "std"),
            median_label_length_mean=("median_label_length", "mean"),
            median_label_length_std=("median_label_length", "std"),
            min_label_length_mean=("min_label_length", "mean"),
            min_label_length_std=("min_label_length", "std"),
            hourly_sample_count=("dns_query_count", "count")
        )
        .reset_index()
    )

    std_cols = [
        "dns_query_count_std",
        "unique_subdomain_count_std",
        "unique_ratio_std",
        "entropy_std",
        "avg_label_length_std",
        "median_label_length_std",
        "min_label_length_std",
    ]

    window_group[std_cols] = window_group[std_cols].fillna(0)

    window_group["dns_query_count_threshold"] = (
        window_group["dns_query_count_mean"] + 3 * window_group["dns_query_count_std"]
    )

    window_group["unique_subdomain_count_threshold"] = (
        window_group["unique_subdomain_count_mean"]
        + 3 * window_group["unique_subdomain_count_std"]
    )

    window_group["unique_ratio_threshold"] = (
        window_group["unique_ratio_mean"] + 3 * window_group["unique_ratio_std"]
    )

    window_group["entropy_threshold"] = (
        window_group["entropy_mean"] + 3 * window_group["entropy_std"]
    )

    window_group["avg_label_length_threshold"] = (
        window_group["avg_label_length_mean"] + 3 * window_group["avg_label_length_std"]
    )

    window_group["median_label_length_threshold"] = (
        window_group["median_label_length_mean"] + 3 * window_group["median_label_length_std"]
    )

    window_group["min_label_length_threshold"] = (
        window_group["min_label_length_mean"] + 3 * window_group["min_label_length_std"]
    )

    baseline = pd.merge(
        r01_group,
        window_group,
        on=group_cols,
        how="outer"
    )

    baseline = (
        baseline
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    baseline["baseline_level"] = baseline_level

    cols = ["baseline_level"] + [col for col in baseline.columns if col != "baseline_level"]
    baseline = baseline[cols]

    print(f"[+] {baseline_level} baseline done. Rows: {len(baseline)}", flush=True)

    return baseline


def main():
    all_dfs = []
    normal_files = glob.glob(NORMAL_DIR)

    if not normal_files:
        raise FileNotFoundError(f"No normal_DNS_traffic TSV files found in: {NORMAL_DIR}")

    for file in normal_files:
        print(f"[+] Processing: {os.path.basename(file)}", flush=True)

        try:
            df = pd.read_csv(
                file,
                sep="\t",
                comment="#",
                usecols=lambda c: c in ["ts", "id.orig_h", "query"],
                low_memory=False
            )
        except Exception as e:
            print(f"[!] Failed to read {file}: {e}", flush=True)
            continue

        required_cols = {"ts", "id.orig_h", "query"}

        if not required_cols.issubset(df.columns):
            print(f"[!] Missing required columns in {file}", flush=True)
            continue

        df = df.dropna(subset=["ts", "id.orig_h", "query"])

        if df.empty:
            continue

        df["datetime"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
        df = df.dropna(subset=["datetime"])

        if df.empty:
            continue

        if BASELINE_DAYS is not None:
            cutoff = df["datetime"].max() - pd.Timedelta(days=BASELINE_DAYS)
            df = df[df["datetime"] >= cutoff]

            if df.empty:
                continue

        df = df.rename(columns={"id.orig_h": "src_ip"})

        df["query_clean"] = (
            df["query"]
            .astype(str)
            .str.strip(".")
            .str.lower()
        )

        df = df[df["query_clean"] != ""]

        if df.empty:
            continue

        domain_parts = df["query_clean"].apply(extract_domain_parts)

        df["etld1"] = domain_parts.apply(lambda x: x[0])
        df["subdomain_labels"] = domain_parts.apply(lambda x: x[1])
        df["subdomain"] = df["subdomain_labels"].apply(".".join)

        df["labels"] = df["query_clean"].str.split(".")
        df["query_length"] = df["query_clean"].str.len()

        df["label_lengths"] = df["labels"].apply(
            lambda labels: [len(label) for label in labels]
        )

        df["avg_label_length"] = df["label_lengths"].apply(
            lambda lens: sum(lens) / len(lens) if lens else 0
        )

        df["median_label_length"] = df["label_lengths"].apply(
            lambda lens: np.median(lens) if lens else 0
        )

        df["min_label_length"] = df["label_lengths"].apply(min)

        df["subdomain_entropy"] = df["subdomain_labels"].apply(
            longest_label_entropy
        )

        df["hour_window"] = df["datetime"].dt.floor(TIME_WINDOW)

        needed_cols = [
            "src_ip",
            "query_clean",
            "etld1",
            "subdomain",
            "query_length",
            "subdomain_entropy",
            "avg_label_length",
            "median_label_length",
            "min_label_length",
            "hour_window"
        ]

        df = df[needed_cols]
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid DNS normal_DNS_traffic data.")

    print("[+] Concatenating normal_DNS_traffic data...", flush=True)
    normal_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[+] Total normal_DNS_traffic rows: {len(normal_df)}", flush=True)

    print("[+] Start IP_DOMAIN baseline...", flush=True)
    ip_domain_baseline = build_baseline_from_group(
        df=normal_df,
        group_cols=["src_ip", "etld1"],
        baseline_level="IP_DOMAIN"
    )

    print("[+] Start DOMAIN baseline...", flush=True)
    domain_baseline = build_baseline_from_group(
        df=normal_df,
        group_cols=["etld1"],
        baseline_level="DOMAIN"
    )

    print("[+] Start GLOBAL baseline...", flush=True)
    global_df = normal_df.copy()
    global_df["etld1"] = "__GLOBAL__"

    global_baseline = build_baseline_from_group(
        df=global_df,
        group_cols=["etld1"],
        baseline_level="GLOBAL"
    )

    print("[+] Combining IP_DOMAIN, DOMAIN and GLOBAL baseline...", flush=True)
    dns_baseline = pd.concat(
        [ip_domain_baseline, domain_baseline, global_baseline],
        ignore_index=True,
        sort=False
    )

    numeric_cols = dns_baseline.select_dtypes(include=[np.number]).columns

    dns_baseline[numeric_cols] = (
        dns_baseline[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"[+] Saving baseline to: {OUTPUT_FILE}", flush=True)

    dns_baseline.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== DNS NORMAL BASELINE RESULT ==========")
    print(f"[+] Saved baseline: {OUTPUT_FILE}")
    print(f"[+] Total baseline rows: {len(dns_baseline)}")
    print("\nBaseline level counts:")
    print(dns_baseline["baseline_level"].value_counts())
    print("\nSample baseline:")
    print(dns_baseline.head())


if __name__ == "__main__":
    main()
