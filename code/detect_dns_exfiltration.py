import os
import math
import re
from collections import Counter

import pandas as pd
import numpy as np
import tldextract


# =========================
# WINDOW CONFIGURATION
# =========================
# Thay đổi WINDOW_MINUTES để chạy benchmark với kích thước cửa sổ khác nhau.
# Giá trị này PHẢI đồng nhất với WINDOW_MINUTES trong dns_baseline.py.
WINDOW_MINUTES = 1
TIME_WINDOW = f"{WINDOW_MINUTES}min"


def _scale_count(base_value_at_10min):
    """Scale ngưỡng count theo tỷ lệ window so với cấu hình chuẩn 10 phút.

    Các ngưỡng số lượng truy vấn trong rules được calibrate cho window 10p.
    Khi đổi WINDOW_MINUTES, mật độ ngưỡng (count/phút) cần giữ cố định để
    so sánh giữa các kích thước cửa sổ là công bằng.

    Ví dụ: ngưỡng 8 ở window 10p → 4 ở window 5p, 24 ở window 30p, 1 ở 1p.
    Tối thiểu trả về 1 để tránh ngưỡng = 0 ở window quá nhỏ.
    """
    return max(1, int(round(base_value_at_10min * WINDOW_MINUTES / 10)))


# =========================
# Default thresholds
# =========================
# Các ngưỡng độ dài, tỷ lệ, entropy KHÔNG phụ thuộc window → giữ nguyên.
# Các ngưỡng count phụ thuộc window → dùng _scale_count() để tự scale.

DEFAULT_QUERY_LENGTH_THRESHOLD = 150
DEFAULT_DNS_QUERY_COUNT_THRESHOLD = _scale_count(8)
DEFAULT_UNIQUE_SUBDOMAIN_COUNT_THRESHOLD = _scale_count(8)
DEFAULT_UNIQUE_RATIO_THRESHOLD = 0.80
DEFAULT_ENTROPY_THRESHOLD = 3.8

ALERT_SCORE_THRESHOLD = 3
ALERT_MIN_GROUPS = 1


BASE32_RE = re.compile(r"^[A-Z2-7]{16,}={0,6}$", re.IGNORECASE)

BASE64_RE = re.compile(
    r"^(?:[A-Za-z0-9+/_-]{4}){4,}(?:==|=)?$"
)

HEX_RE = re.compile(r"^[0-9a-f]{20,}$", re.IGNORECASE)

# FIX: R07 protocol-level signatures cho DNS tunnel.
# Lý do thêm: dns2tcp-key dùng qtype KEY (RFC 2535, legacy DNSSEC, gần như
# zero use trong production), và dns2tcp-txt dùng delimiter `=auth/=connect`
# trong subdomain. Các signal này có specificity cực cao → strong-single-rule.
RARE_QTYPES = {"KEY", "TKEY"}

# RFC 1035: subdomain labels hợp lệ là [a-zA-Z0-9-].
# `_` được dùng hợp lệ ở service records (SRV, DKIM, DMARC) nên KHÔNG tính.
# `=`, `+`, `/` là signature của dns2tcp / một số DNSStager.
SUSPICIOUS_CHAR_RE = re.compile(r"[=+/]")


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


def get_etld1(query):
    ext = tldextract.extract(str(query).strip("."))

    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"

    return str(query).strip(".")


def get_subdomain_labels(query):
    ext = tldextract.extract(str(query).strip("."))

    if ext.subdomain:
        return ext.subdomain.split(".")

    return []


def longest_label_entropy(labels):
    if not labels:
        return 0.0

    longest_label = max(labels, key=len)
    return shannon_entropy(longest_label)


def detect_encoding(labels):
    for label in labels:
        label = str(label)

        if len(label) < 16:
            continue

        entropy = shannon_entropy(label)

        if HEX_RE.match(label) and entropy >= 3.2:
            return "hex"

        if BASE32_RE.match(label) and entropy >= 3.2:
            return "base32"

        if BASE64_RE.match(label):
            has_upper = any(c.isupper() for c in label)
            has_lower = any(c.islower() for c in label)
            has_digit = any(c.isdigit() for c in label)

            has_symbol = (
                "-" in label
                or "_" in label
                or "+" in label
                or "/" in label
            )

            if (
                len(label) >= 20
                and entropy >= 3.6
                and (
                    has_symbol
                    or (
                        has_upper
                        and has_lower
                        and has_digit
                    )
                )
            ):
                return "base64"

    return None


def safe_get(row, col, default_value):
    if row is None:
        return default_value

    if col not in row.index:
        return default_value

    value = row[col]

    if pd.isna(value):
        return default_value

    return value


# FIX #1: effective_threshold — ưu tiên baseline nếu hợp lệ.
# Logic cũ: min(default, baseline) → luôn chọn giá trị nhỏ hơn,
# bỏ qua baseline cao hơn default → FP tăng với domain traffic lớn.
# Logic mới: dùng baseline nếu hợp lệ, floor là min_value, fallback về default.
def effective_threshold(default_value, baseline_value, min_value):
    try:
        baseline_value = float(baseline_value)
    except Exception:
        return default_value

    if pd.isna(baseline_value) or baseline_value <= 0:
        return default_value

    return max(min_value, baseline_value)


def load_baseline(baseline_file):
    """
    FIX: Tách baseline_dict theo baseline_level.
    Bug cũ: dict cũ build bằng row["etld1"] làm key, mà cả 3 cấp
    (IP_DOMAIN, DOMAIN, GLOBAL) đều dùng cùng giá trị etld1 →
    row sau ghi đè row trước, gần như chỉ giữ lại level cuối cùng.
    Cấp IP_DOMAIN còn bị flatten do key không chứa src_ip.

    Cấu trúc mới:
      {
          "IP_DOMAIN": {(src_ip, etld1): row, ...},
          "DOMAIN":    {etld1: row, ...},
          "GLOBAL":    {"__GLOBAL__": row}
      }
    """
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(
            f"Baseline file not found: {baseline_file}"
        )

    baseline = pd.read_csv(baseline_file)

    if "baseline_level" not in baseline.columns:
        raise ValueError(
            "Baseline file thiếu cột 'baseline_level'. "
            "Hãy rebuild baseline bằng dns_baseline.py phiên bản mới."
        )

    ip_domain_dict = {}
    domain_dict = {}
    global_dict = {}

    for _, row in baseline.iterrows():
        level = row["baseline_level"]

        if level == "IP_DOMAIN":
            src_ip = row.get("src_ip")

            if pd.isna(src_ip) or src_ip == "":
                continue

            ip_domain_dict[(src_ip, row["etld1"])] = row

        elif level == "DOMAIN":
            domain_dict[row["etld1"]] = row

        elif level == "GLOBAL":
            global_dict["__GLOBAL__"] = row

    return {
        "IP_DOMAIN": ip_domain_dict,
        "DOMAIN": domain_dict,
        "GLOBAL": global_dict
    }


def get_baseline_for_domain(baseline_dict, src_ip, etld1):
    """
    FIX: Thêm src_ip vào lookup để tận dụng cấp IP_DOMAIN.
    Priority: IP_DOMAIN > DOMAIN > GLOBAL > DEFAULT.
    """
    ip_domain_key = (src_ip, etld1)

    if ip_domain_key in baseline_dict["IP_DOMAIN"]:
        return (
            baseline_dict["IP_DOMAIN"][ip_domain_key],
            "IP_DOMAIN_BASELINE"
        )

    if etld1 in baseline_dict["DOMAIN"]:
        return (
            baseline_dict["DOMAIN"][etld1],
            "DOMAIN_BASELINE"
        )

    if "__GLOBAL__" in baseline_dict["GLOBAL"]:
        return (
            baseline_dict["GLOBAL"]["__GLOBAL__"],
            "GLOBAL_BASELINE"
        )

    return None, "NO_BASELINE_USE_DEFAULT"


def add_rule(triggered_rules, triggered_groups, rule_name, group_name, score):
    triggered_rules.append(rule_name)
    triggered_groups.add(group_name)
    return score


def process_file(file_path, baseline_dict):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"TSV file not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep="\t",
        comment="#",
        low_memory=False
    )

    required_cols = {"ts", "id.orig_h", "query"}

    if not required_cols.issubset(df.columns):
        raise ValueError(
            "TSV file thiếu cột ts, id.orig_h hoặc query"
        )

    keep_cols = ["ts", "id.orig_h", "query"]

    if "rcode_name" in df.columns:
        keep_cols.append("rcode_name")

    if "qtype_name" in df.columns:
        keep_cols.append("qtype_name")

    df = df[keep_cols].dropna(subset=["ts", "id.orig_h", "query"])

    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"id.orig_h": "src_ip"})

    df["datetime"] = pd.to_datetime(
        df["ts"],
        unit="s",
        errors="coerce"
    )

    df = df.dropna(subset=["datetime"])

    df["query_clean"] = (
        df["query"]
        .astype(str)
        .str.strip(".")
        .str.lower()
    )

    df = df[df["query_clean"] != ""]

    if df.empty:
        return pd.DataFrame()

    df["etld1"] = df["query_clean"].apply(get_etld1)
    df["labels"] = df["query_clean"].apply(lambda q: q.split("."))
    df["subdomain_labels"] = df["query_clean"].apply(get_subdomain_labels)
    df["subdomain"] = df["subdomain_labels"].apply(".".join)

    df["num_labels"] = df["labels"].str.len()
    df["query_length"] = df["query_clean"].str.len()

    df["label_lengths"] = df["labels"].apply(
        lambda labels: [len(label) for label in labels]
    )

    df["max_label_length"] = df["label_lengths"].apply(max)

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

    df["encoding_type"] = df["subdomain_labels"].apply(detect_encoding)

    if "rcode_name" in df.columns:
        df["is_nxdomain"] = (
            df["rcode_name"]
            .astype(str)
            .str.upper()
            .eq("NXDOMAIN")
        )
    else:
        df["is_nxdomain"] = False

    if "qtype_name" in df.columns:
        df["qtype_upper"] = (
            df["qtype_name"]
            .astype(str)
            .str.upper()
        )
        df["is_txt_query"] = df["qtype_upper"].eq("TXT")
        df["is_null_query"] = df["qtype_upper"].eq("NULL")
        df["is_rare_qtype"] = df["qtype_upper"].isin(RARE_QTYPES)
    else:
        df["is_txt_query"] = False
        df["is_null_query"] = False
        df["is_rare_qtype"] = False

    # FIX: detect ký tự bất thường trong subdomain - signature của DNS tunnel
    # (dns2tcp dùng `=auth`/`=connect` làm delimiter). Bắt được single-query
    # window mà volume-based rule câm.
    df["has_suspicious_char"] = (
        df["subdomain"]
        .astype(str)
        .str.contains(SUSPICIOUS_CHAR_RE, regex=True)
    )

    df["hour_window"] = df["datetime"].dt.floor(TIME_WINDOW)

    df = df.sort_values(["src_ip", "etld1", "datetime"])

    df["interarrival_seconds"] = (
        df
        .groupby(["src_ip", "etld1"])["datetime"]
        .diff()
        .dt.total_seconds()
    )

    window_df = (
        df
        .groupby(["src_ip", "etld1", "hour_window"], sort=False)
        .agg(
            dns_query_count=("query_clean", "count"),
            unique_subdomain_count=("subdomain", "nunique"),

            first_seen=("datetime", "min"),
            last_seen=("datetime", "max"),
            interarrival_mean=("interarrival_seconds", "mean"),
            interarrival_std=("interarrival_seconds", "std"),
            interarrival_min=("interarrival_seconds", "min"),
            interarrival_max=("interarrival_seconds", "max"),

            max_query_length=("query_length", "max"),
            max_num_labels=("num_labels", "max"),
            max_label_length=("max_label_length", "max"),

            avg_label_length=("avg_label_length", "mean"),
            median_label_length=("median_label_length", "median"),
            min_label_length=("min_label_length", "min"),

            max_entropy=("subdomain_entropy", "max"),
            avg_entropy=("subdomain_entropy", "mean"),

            encoded_label_count=("encoding_type", lambda x: x.notna().sum()),
            nxdomain_count=("is_nxdomain", "sum"),
            txt_query_count=("is_txt_query", "sum"),
            null_query_count=("is_null_query", "sum"),
            rare_qtype_count=("is_rare_qtype", "sum"),
            suspicious_char_count=("has_suspicious_char", "sum")
        )
        .reset_index()
    )

    window_df["unique_ratio"] = (
        window_df["unique_subdomain_count"]
        / window_df["dns_query_count"]
    )

    window_df["nxdomain_ratio"] = (
        window_df["nxdomain_count"]
        / window_df["dns_query_count"]
    )

    window_df["txt_query_ratio"] = (
        window_df["txt_query_count"]
        / window_df["dns_query_count"]
    )

    window_df["null_query_ratio"] = (
        window_df["null_query_count"]
        / window_df["dns_query_count"]
    )

    window_df["rare_qtype_ratio"] = (
        window_df["rare_qtype_count"]
        / window_df["dns_query_count"]
    )

    window_df["suspicious_char_ratio"] = (
        window_df["suspicious_char_count"]
        / window_df["dns_query_count"]
    )

    window_df["duration_seconds"] = (
        window_df["last_seen"] - window_df["first_seen"]
    ).dt.total_seconds()

    window_df["duration_seconds"] = (
        window_df["duration_seconds"]
        .replace(0, 1)
    )

    window_df["query_rate"] = (
        window_df["dns_query_count"] / window_df["duration_seconds"]
    )

    window_df = (
        window_df
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    results = []

    for _, row in window_df.iterrows():
        src_ip = row["src_ip"]
        etld1 = row["etld1"]

        base, baseline_status = get_baseline_for_domain(
            baseline_dict,
            src_ip,
            etld1
        )

        baseline_query_length_threshold = safe_get(
            base,
            "query_length_threshold",
            DEFAULT_QUERY_LENGTH_THRESHOLD
        )

        baseline_dns_query_count_threshold = safe_get(
            base,
            "dns_query_count_threshold",
            DEFAULT_DNS_QUERY_COUNT_THRESHOLD
        )

        baseline_unique_subdomain_count_threshold = safe_get(
            base,
            "unique_subdomain_count_threshold",
            DEFAULT_UNIQUE_SUBDOMAIN_COUNT_THRESHOLD
        )

        baseline_unique_ratio_threshold = safe_get(
            base,
            "unique_ratio_threshold",
            DEFAULT_UNIQUE_RATIO_THRESHOLD
        )

        baseline_entropy_threshold = safe_get(
            base,
            "entropy_threshold",
            DEFAULT_ENTROPY_THRESHOLD
        )

        # FIX #1 applied: effective_threshold giờ ưu tiên baseline
        query_length_threshold = effective_threshold(
            DEFAULT_QUERY_LENGTH_THRESHOLD,
            baseline_query_length_threshold,
            min_value=100
        )

        dns_query_count_threshold = effective_threshold(
            DEFAULT_DNS_QUERY_COUNT_THRESHOLD,
            baseline_dns_query_count_threshold,
            min_value=_scale_count(5)
        )

        unique_subdomain_count_threshold = effective_threshold(
            DEFAULT_UNIQUE_SUBDOMAIN_COUNT_THRESHOLD,
            baseline_unique_subdomain_count_threshold,
            min_value=_scale_count(5)
        )

        unique_ratio_threshold = effective_threshold(
            DEFAULT_UNIQUE_RATIO_THRESHOLD,
            baseline_unique_ratio_threshold,
            min_value=0.70
        )

        entropy_threshold = effective_threshold(
            DEFAULT_ENTROPY_THRESHOLD,
            baseline_entropy_threshold,
            min_value=3.3
        )

        triggered_rules = []
        triggered_groups = set()
        risk_score = 0

        # =========================
        # R01 - Query length / label structure
        #
        # FIX #2: Chuyển sang if/elif để tránh double-count.
        # Trước đây R01_LONG_QUERY_STRUCTURAL (+2) và R01_VERY_LONG_QUERY (+3)
        # có thể cùng fire cho một event → cộng 5 điểm chỉ từ R01,
        # vượt ALERT_SCORE_THRESHOLD mà không cần tín hiệu khác.
        # Chỉ lấy điểm cao nhất phù hợp theo thứ tự nghiêm trọng giảm dần.
        # =========================

        if (
            row["max_query_length"] >= 180
            and (
                row["max_label_length"] >= 30
                or row["max_num_labels"] >= 7
            )
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R01_VERY_LONG_QUERY",
                "R01",
                3
            )

        elif (
            row["max_query_length"] >= query_length_threshold
            and (
                row["max_label_length"] >= 30
                or row["max_num_labels"] >= 7
                or row["median_label_length"] >= 12
            )
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R01_LONG_QUERY_STRUCTURAL",
                "R01",
                2
            )

        elif (
            row["max_num_labels"] >= 9
            and row["max_query_length"] >= 120
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R01_MANY_LABELS",
                "R01",
                1
            )

        # =========================
        # R02 - Entropy / encoded payload
        # =========================

        if (
            row["encoded_label_count"] >= 1
            and row["max_label_length"] >= 20
            and row["max_entropy"] >= 3.8
            and (
                row["encoded_label_count"] >= 2
                or row["unique_ratio"] >= 0.80
                or row["dns_query_count"] >= _scale_count(8)
            )
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R02_ENCODED_LABEL_PATTERN",
                "R02",
                3
            )

        if (
            row["max_entropy"] >= entropy_threshold
            and (
                row["dns_query_count"] >= dns_query_count_threshold
                or row["max_label_length"] >= 24
            )
            and row["median_label_length"] >= 8
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R02_HIGH_ENTROPY",
                "R02",
                2
            )

        if (
            row["max_entropy"] >= 3.6
            and row["median_label_length"] >= 12
            and row["dns_query_count"] >= _scale_count(8)
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R02_SOFT_ENTROPY_LONG_LABEL",
                "R02",
                1
            )

        # =========================
        # R03 - Cardinality / unique subdomain behavior
        # =========================

        if (
            row["unique_subdomain_count"] >= unique_subdomain_count_threshold
            and row["dns_query_count"] >= dns_query_count_threshold
            and (
                row["max_label_length"] >= 16
                or row["median_label_length"] >= 10
                or row["max_entropy"] >= 3.6
            )
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R03_UNIQUE_SUBDOMAIN_COUNT",
                "R03",
                2
            )

        if (
            row["unique_ratio"] >= max(0.95, unique_ratio_threshold)
            and row["dns_query_count"] >= _scale_count(10)
            and (
                row["max_label_length"] >= 20
                or row["median_label_length"] >= 12
                or row["max_entropy"] >= 3.6
            )
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R03_HIGH_UNIQUE_RATIO",
                "R03",
                2
            )

        # =========================
        # R04 - NXDOMAIN behavior
        # =========================

        if (
            row["nxdomain_ratio"] >= 0.80
            and row["dns_query_count"] >= _scale_count(8)
            and (
                row["unique_ratio"] >= 0.80
                or row["max_label_length"] >= 16
            )
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R04_VERY_HIGH_NXDOMAIN_RATIO",
                "R04",
                2
            )
        elif (
            row["nxdomain_ratio"] >= 0.50
            and row["dns_query_count"] >= _scale_count(10)
            and row["unique_ratio"] >= 0.90
            and row["max_label_length"] >= 20
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R04_HIGH_NXDOMAIN_RATIO",
                "R04",
                1
            )

        # =========================
        # R05 - Suspicious qtype behavior
        # =========================

        if (
            row["txt_query_ratio"] >= 0.80
            and row["dns_query_count"] >= _scale_count(5)
            and (
                row["max_entropy"] >= 3.8
                or row["max_label_length"] >= 24
                or row["unique_ratio"] >= 0.90
            )
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R05_HIGH_TXT_QUERY_RATIO",
                "R05",
                2
            )

        if (
            row["null_query_ratio"] >= 0.30
            and row["dns_query_count"] >= _scale_count(5)
            and (
                row["unique_ratio"] >= 0.80
                or row["max_label_length"] >= 16
            )
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R05_NULL_QUERY_PATTERN",
                "R05",
                2
            )

        # =========================
        # R06 - Behavioral temporal pattern
        # =========================

        if (
            row["dns_query_count"] >= _scale_count(15)
            and row["unique_ratio"] >= 0.85
            and row["query_rate"] >= 0.20
            and (
                row["max_label_length"] >= 16
                or row["median_label_length"] >= 10
                or row["max_entropy"] >= 3.6
            )
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R06_BURST_UNIQUE_QUERIES",
                "R06",
                2
            )

        if (
            row["dns_query_count"] >= _scale_count(10)
            and row["unique_ratio"] >= 0.85
            and row["interarrival_mean"] > 0
            and row["interarrival_std"] > 0
            and row["interarrival_std"] <= 2
            and (
                row["max_label_length"] >= 16
                or row["max_entropy"] >= 3.6
            )
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R06_REGULAR_INTERVAL_QUERIES",
                "R06",
                2
            )

        # =========================
        # R07 - Protocol-level tunnel signatures
        #
        # 2 signal trong nhóm này có specificity cực cao và độc lập với
        # volume, nên đặt là strong-single-rule để bắt được window 1 query.
        #
        # - RARE_QTYPE: KEY / TKEY gần như zero base-rate trong DNS hợp pháp,
        #   nhưng được dns2tcp-key dùng làm transport.
        # - SUSPICIOUS_CHAR: `=`, `+`, `/` không hợp lệ trong subdomain theo
        #   RFC 1035, nhưng dns2tcp dùng `=auth` / `=connect` làm delimiter.
        # =========================

        if row["rare_qtype_count"] >= 1:
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R07_RARE_QTYPE",
                "R07",
                3
            )

        if (
            row["suspicious_char_count"] >= 1
            and row["max_label_length"] >= 8
        ):
            risk_score += add_rule(
                triggered_rules,
                triggered_groups,
                "R07_SUSPICIOUS_CHAR_IN_SUBDOMAIN",
                "R07",
                3
            )

        # =========================
        # Final decision
        #
        # FIX #2 side-effect: R01_VERY_LONG_QUERY đã được bỏ khỏi
        # strong_single_rules vì double-count đã được xử lý bằng if/elif.
        # R01 một mình tối đa 3 điểm, cần kết hợp thêm rule khác để alert.
        # Giữ R05_NULL_QUERY_PATTERN và thêm R02_ENCODED_LABEL_PATTERN
        # vì hai rule này có độ chính xác cao khi fire độc lập.
        # Thêm R07_* vì specificity ngang/cao hơn (xem comment khối R07).
        # =========================

        strong_single_rules = {
            "R05_NULL_QUERY_PATTERN",
            "R02_ENCODED_LABEL_PATTERN",
            "R07_RARE_QTYPE",
            "R07_SUSPICIOUS_CHAR_IN_SUBDOMAIN",
        }

        has_strong_single_rule = any(
            rule in strong_single_rules
            for rule in triggered_rules
        )

        if (
            (
                risk_score >= ALERT_SCORE_THRESHOLD
                and len(triggered_groups) >= ALERT_MIN_GROUPS
            )
            or has_strong_single_rule
        ):
            y_pred = 1
        else:
            y_pred = 0

        results.append({
            "source_file": os.path.basename(file_path),
            "source_path": file_path,

            "src_ip": src_ip,
            "etld1": etld1,
            "hour_window": row["hour_window"],

            "y_pred": int(y_pred),
            "risk_score": int(risk_score),

            "dns_query_count": int(row["dns_query_count"]),
            "unique_subdomain_count": int(row["unique_subdomain_count"]),
            "unique_ratio": round(row["unique_ratio"], 3),

            "duration_seconds": round(row["duration_seconds"], 3),
            "query_rate": round(row["query_rate"], 6),
            "interarrival_mean": round(row["interarrival_mean"], 3),
            "interarrival_std": round(row["interarrival_std"], 3),
            "interarrival_min": round(row["interarrival_min"], 3),
            "interarrival_max": round(row["interarrival_max"], 3),

            "nxdomain_count": int(row["nxdomain_count"]),
            "nxdomain_ratio": round(row["nxdomain_ratio"], 3),

            "txt_query_count": int(row["txt_query_count"]),
            "txt_query_ratio": round(row["txt_query_ratio"], 3),
            "null_query_count": int(row["null_query_count"]),
            "null_query_ratio": round(row["null_query_ratio"], 3),

            "rare_qtype_count": int(row["rare_qtype_count"]),
            "rare_qtype_ratio": round(row["rare_qtype_ratio"], 3),
            "suspicious_char_count": int(row["suspicious_char_count"]),
            "suspicious_char_ratio": round(row["suspicious_char_ratio"], 3),

            "max_query_length": int(row["max_query_length"]),
            "max_num_labels": int(row["max_num_labels"]),
            "max_label_length": int(row["max_label_length"]),
            "avg_label_length": round(row["avg_label_length"], 3),
            "median_label_length": round(row["median_label_length"], 3),
            "min_label_length": int(row["min_label_length"]),

            "max_entropy": round(row["max_entropy"], 3),
            "avg_entropy": round(row["avg_entropy"], 3),
            "encoded_label_count": int(row["encoded_label_count"]),

            "query_length_threshold": round(query_length_threshold, 3),
            "dns_query_count_threshold": round(dns_query_count_threshold, 3),
            "unique_subdomain_count_threshold": round(
                unique_subdomain_count_threshold,
                3
            ),
            "unique_ratio_threshold": round(unique_ratio_threshold, 3),
            "entropy_threshold": round(entropy_threshold, 3),

            "triggered_rules": "|".join(triggered_rules),
            "triggered_groups": "|".join(sorted(triggered_groups)),
            "baseline_status": baseline_status
        })

    return pd.DataFrame(results)
