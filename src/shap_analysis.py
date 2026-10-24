"""模型可解释性分析（SHAP）。

对每个数据集的最优模型：
  1. 提取特征重要性排序（mean|SHAP|）
  2. 全局解释：SHAP 摘要图（beeswarm）与条形图
  3. 个体解释：正例 / 负例 waterfall 图
"""
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_OUT, MODEL_OUT, FIG_OUT, FEATURES, TARGET
from plotting import setup_plot

setup_plot()


def load_best(tag):
    pipe = joblib.load(f"{MODEL_OUT}/{tag}_best_model.joblib")
    df = pd.read_csv(f"{DATA_OUT}/{tag}_cleaned.csv")
    return pipe, df


def analyze(tag):
    pipe, df = load_best(tag)
    X = df[FEATURES]
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    X_scaled = scaler.transform(X)

    explainer = shap.Explainer(clf, X_scaled)
    shap_values = explainer(X_scaled)
    if shap_values.values.ndim == 3:
        shap_values = shap_values[..., 1]

    imp = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(imp)[::-1]
    imp_df = pd.DataFrame({"feature": [FEATURES[i] for i in order],
                           "mean_abs_shap": imp[order].round(4)})
    imp_df.to_csv(f"{DATA_OUT}/{tag}_shap_importance.csv", index=False)
    print(f"[{tag}] 特征重要性排序 (mean|SHAP|):")
    print(imp_df.to_string(index=False))

    shap.plots.bar(shap_values, max_display=8, show=False)
    plt.title(f"{tag} 最优模型 SHAP 特征重要性")
    plt.savefig(f"{FIG_OUT}/shap_bar_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(shap_values, max_display=8, show=False)
    plt.title(f"{tag} 最优模型 SHAP 摘要图（全局解释）")
    plt.savefig(f"{FIG_OUT}/shap_summary_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

    proba = clf.predict_proba(X_scaled)[:, 1]
    idx_pos = int(np.argmax(proba))
    idx_neg = int(np.argmin(proba))
    for kind, idx in [("pos", idx_pos), ("neg", idx_neg)]:
        shap.plots.waterfall(shap_values[idx], max_display=8, show=False)
        plt.title(f"{tag} 个体解释（{'糖尿病' if kind == 'pos' else '非糖尿病'}样本）")
        plt.savefig(f"{FIG_OUT}/shap_waterfall_{tag}_{kind}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[{tag}] 已保存 shap_bar / shap_summary / shap_waterfall 图表")


def main():
    for tag in ["FHGD", "pima"]:
        print(f"\n===== {tag} =====")
        analyze(tag)
    print("\nSHAP 可解释性分析完成。")


if __name__ == "__main__":
    main()
