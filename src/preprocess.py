"""数据获取与预处理。

流程（对 Pima 与 FHGD 两个数据集一致）：
  1. 去重
  2. 识别 Glucose/BloodPressure/SkinThickness/Insulin/BMI 中以 0 编码的缺失值并删除对应样本
  3. IQR(1.5) 异常值检测与边界截断
  4. StandardScaler 标准化
  5. SMOTE 类别不平衡处理

输出：
  - <name>_cleaned.csv      清理后、原始尺度数据（供模型训练在 Pipeline 内重新标准化）
  - <name>_preprocessed.csv 标准化 + SMOTE 平衡后的最终数据（供演示与讲解）
  - preprocess_report.json  各步骤统计信息
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from config import (FHGD_PATH, PIMA_DIR, DATA_OUT, ZERO_COLS, FEATURES,
                    TARGET, IQR_MULT, RANDOM_STATE)


def load_fhgd():
    return pd.read_csv(FHGD_PATH)


def load_pima():
    rows = []
    for split in ("train", "validation", "test"):
        with open(f"{PIMA_DIR}/{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


def clean(df, name):
    """去重 + 删除 0 编码缺失样本 + IQR 截断，返回清理后数据与统计。"""
    report = {"name": name, "raw_shape": list(df.shape)}
    report["raw_class"] = df[TARGET].value_counts().sort_index().to_dict()

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = n_dup
    report["after_dedup"] = len(df)

    missing = {c: int((df[c] == 0).sum()) for c in ZERO_COLS}
    report["zero_missing"] = missing
    mask = (df[ZERO_COLS] == 0).any(axis=1)
    df = df[~mask].reset_index(drop=True)
    report["missing_samples_removed"] = int(mask.sum())
    report["after_missing"] = len(df)

    outlier_stats = {}
    total = 0
    for c in FEATURES:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - IQR_MULT * iqr, q3 + IQR_MULT * iqr
        n_low, n_high = int((df[c] < lo).sum()), int((df[c] > hi).sum())
        total += n_low + n_high
        outlier_stats[c] = {"low": n_low, "high": n_high, "bound": [round(lo, 2), round(hi, 2)]}
        df[c] = df[c].clip(lo, hi)
    report["outliers_clipped"] = total
    report["outlier_detail"] = outlier_stats
    report["cleaned_shape"] = list(df.shape)
    report["cleaned_class"] = df[TARGET].value_counts().sort_index().to_dict()
    return df, report


def standardize(df):
    scaler = StandardScaler()
    df = df.copy()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    return df, scaler


def balance_smote(df):
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(df[FEATURES].values, df[TARGET].values)
    out = pd.DataFrame(X_res, columns=FEATURES)
    out[TARGET] = y_res.astype(int)
    return out


def preprocess(df, name):
    """完整预处理：返回 (cleaned_df, final_df, report)。"""
    cleaned, report = clean(df, name)
    scaled, _ = standardize(cleaned)
    final = balance_smote(scaled)
    report["final_shape"] = list(final.shape)
    report["final_class"] = final[TARGET].value_counts().sort_index().to_dict()
    return cleaned, final, report


def save(df, name):
    path = f"{DATA_OUT}/{name}.csv"
    df.to_csv(path, index=False)
    return path


def main():
    datasets = [("FHGD", load_fhgd()), ("pima", load_pima())]
    reports = []
    for tag, raw in datasets:
        cleaned, final, report = preprocess(raw, tag)
        reports.append(report)
        print(f"\n[{tag}] 原始 {report['raw_shape']} -> 去重 {report['after_dedup']}"
              f" -> 删缺失 {report['after_missing']} -> 清理 {report['cleaned_shape']}"
              f" -> 平衡 {report['final_shape']}")
        print(f"  类别分布: 原始 {report['raw_class']} -> 清理 {report['cleaned_class']}"
              f" -> 平衡 {report['final_class']}")
        print(f"  0 编码缺失: {report['zero_missing']}")
        print(f"  异常值截断: {report['outliers_clipped']}")
        print(f"  已保存: {save(cleaned, tag + '_cleaned')}")
        print(f"  已保存: {save(final, tag + '_preprocessed')}")

    with open(f"{DATA_OUT}/preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print("\n报告已保存: preprocess_report.json")


if __name__ == "__main__":
    main()
