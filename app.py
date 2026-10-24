"""糖尿病风险预测与可视化系统（Streamlit 应用）。

运行：streamlit run app.py
模块：数据概览 / 探索分析 / 模型性能 / 特征重要性 / 风险预测 / 模型解释
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import streamlit as st

from config import (DATA_OUT, MODEL_OUT, FIG_OUT, DATA_DIR, FEATURES, TARGET,
                    ZERO_COLS, RANDOM_STATE)
from plotting import setup_plot

setup_plot()

st.set_page_config(page_title="糖尿病风险预测与可视化系统", layout="wide")

FEATURE_LABELS = {
    "Pregnancies": "怀孕次数",
    "Glucose": "血糖 (mg/dL)",
    "BloodPressure": "血压 (mm Hg)",
    "SkinThickness": "皮肤厚度 (mm)",
    "Insulin": "胰岛素 (μU/mL)",
    "BMI": "体质指数 BMI",
    "DiabetesPedigreeFunction": "糖尿病谱系函数",
    "Age": "年龄",
}
MODEL_NAMES = {"逻辑回归": "逻辑回归", "随机森林": "随机森林",
               "XGBoost": "XGBoost", "SVM": "SVM"}
DATASETS = {"FHGD": "FHGD", "Pima": "pima"}


@st.cache_data
def load_cleaned(tag):
    return pd.read_csv(f"{DATA_OUT}/{tag}_cleaned.csv")


@st.cache_data
def load_raw(tag):
    if tag == "FHGD":
        return pd.read_csv(os.path.join(DATA_DIR, "FHGD.csv"))
    rows = []
    for split in ("train", "validation", "test"):
        path = os.path.join(DATA_DIR, "pima", f"{split}.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


@st.cache_data
def load_report():
    with open(f"{DATA_OUT}/preprocess_report.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics(tag):
    tuned = pd.read_csv(f"{MODEL_OUT}/{tag}_tuned.csv", index_col=0)
    baseline = pd.read_csv(f"{MODEL_OUT}/{tag}_baseline.csv", index_col=0)
    with open(f"{MODEL_OUT}/{tag}_best_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return tuned, baseline, meta


@st.cache_resource
def load_model(tag):
    return joblib.load(f"{MODEL_OUT}/{tag}_best_model.joblib")


@st.cache_resource
def get_explainer(tag):
    pipe = load_model(tag)
    df = load_cleaned(tag)
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    X_scaled = scaler.transform(df[FEATURES])
    explainer = shap.Explainer(clf, X_scaled)
    return pipe, explainer, df


def feature_dist_figure(df, cols):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    for i, c in enumerate(cols):
        for outcome, color, label in [(0, "#4C72B0", "非糖尿病"), (1, "#DD8452", "糖尿病")]:
            axes[i].hist(df[df[TARGET] == outcome][c], bins=20, alpha=0.55,
                         color=color, label=label, density=True)
        axes[i].set_title(FEATURE_LABELS.get(c, c))
        axes[i].legend(fontsize=7)
    axes[-1].axis("off")
    fig.tight_layout()
    return fig


def main():
    st.title("🩺 糖尿病风险预测与可视化系统")
    st.caption("数据集：Pima 印第安人糖尿病数据集 & FHGD 糖尿病数据集（2768 条记录，8 个特征）")

    tag = st.sidebar.selectbox("选择数据集", list(DATASETS.keys()))
    tag = DATASETS[tag]
    report = load_report()
    rpt = next(r for r in report if r["name"].lower() == tag.lower())
    df = load_cleaned(tag)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📋 数据概览", "🔍 探索分析", "📊 模型性能", "⭐ 特征重要性",
         "🩹 风险预测", "🔬 模型解释"])

    with tab1:
        st.subheader("数据与预处理概览")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("原始样本", rpt["raw_shape"][0])
        c2.metric("去重后", rpt["after_dedup"])
        c3.metric("清理后", rpt["cleaned_shape"][0])
        c4.metric("SMOTE 平衡后", rpt["final_shape"][0])

        st.write("**0 编码缺失值统计**（识别并删除对应样本）")
        st.dataframe(pd.DataFrame([rpt["zero_missing"]]).T.rename(
            columns={0: "缺失个数"}))

        st.write("**类别分布变化**")
        dist = pd.DataFrame({
            "原始": rpt["raw_class"], "清理后": rpt["cleaned_class"],
            "平衡后": rpt["final_class"]}).T
        dist.columns = ["非糖尿病(0)", "糖尿病(1)"]
        st.dataframe(dist)

        st.write("**清理后数据预览**")
        st.dataframe(df.head(10))
        st.write("**原始数据预览**")
        st.dataframe(load_raw(tag).head(10))

    with tab2:
        st.subheader("探索性数据分析")
        st.write(f"**描述性统计**（{tag}，清理后 {len(df)} 样本）")
        st.dataframe(df[FEATURES + [TARGET]].describe().T.round(3))

        c1, c2 = st.columns(2)
        with c1:
            st.write("**类别分布**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            counts = df[TARGET].value_counts().sort_index()
            ax.bar(["非糖尿病 (0)", "糖尿病 (1)"], counts.values,
                   color=["#4C72B0", "#DD8452"])
            for i, v in enumerate(counts.values):
                ax.text(i, v + 1, str(v), ha="center")
            ax.set_ylabel("样本数")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            st.write("**特征相关性热力图**")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f",
                        cmap="RdBu_r", center=0, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        st.write("**关键特征分布（按糖尿病与否）**")
        st.pyplot(feature_dist_figure(df, ["Glucose", "BMI", "Age", "Insulin", "BloodPressure"]))

        st.write("**Glucose 与 BMI 散点分布**")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for outcome, color, label in [(0, "#4C72B0", "非糖尿病"), (1, "#DD8452", "糖尿病")]:
            sub = df[df[TARGET] == outcome]
            ax.scatter(sub["Glucose"], sub["BMI"], s=15, alpha=0.6, color=color, label=label)
        ax.set_xlabel("Glucose (血糖)")
        ax.set_ylabel("BMI (体质指数)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("机器学习模型对比")
        tuned, baseline, meta = load_metrics(tag)
        st.write(f"**最优模型：{meta['name']}**（ROC-AUC = {meta['auc']:.4f}），"
                 f"最优参数：{meta['params']}")
        st.write("**调优后五项指标**（5 折分层交叉验证均值）")
        st.dataframe(tuned.round(4))

        st.write("**模型对比图**")
        fig, ax = plt.subplots(figsize=(8, 5))
        tuned.T.plot(kind="bar", ax=ax, rot=0)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{tag} 调优后指标对比")
        ax.set_xlabel("指标")
        ax.legend(title="模型")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.write("**调优前后 ROC-AUC 对比**")
        compare = pd.DataFrame({
            "基线(默认参数)": baseline["ROC-AUC"],
            "调优后": tuned["ROC-AUC"]})
        st.dataframe(compare.round(4))

        c1, c2 = st.columns(2)
        with c1:
            st.write("**ROC 曲线对比（测试集）**")
            st.image(f"{FIG_OUT}/roc_curve_{tag}.png")
        with c2:
            st.write("**最优模型混淆矩阵**")
            st.image(f"{FIG_OUT}/confusion_matrix_{tag}.png")

    with tab4:
        st.subheader("特征重要性")
        imp = pd.read_csv(f"{DATA_OUT}/{tag}_shap_importance.csv")
        st.write("**SHAP 特征重要性排序（mean|SHAP|）**")
        st.dataframe(imp)
        st.write("**SHAP 条形图**")
        st.image(f"{FIG_OUT}/shap_bar_{tag}.png")
        st.write("**SHAP 摘要图（全局解释）**")
        st.image(f"{FIG_OUT}/shap_summary_{tag}.png")

    with tab5:
        st.subheader("糖尿病风险预测")
        st.write("输入 8 项生理指标，使用最优模型预测糖尿病风险。")
        pipe = load_model(tag)
        defaults = df[FEATURES].median()
        c1, c2 = st.columns(2)
        inputs = {}
        for i, c in enumerate(FEATURES):
            col = c1 if i % 2 == 0 else c2
            lo, hi = float(df[c].min()), float(df[c].max())
            inputs[c] = col.number_input(
                FEATURE_LABELS[c], min_value=lo, max_value=hi * 1.5,
                value=float(defaults[c]), step=0.1 if c in
                ["BMI", "DiabetesPedigreeFunction"] else 1.0)
        if st.button("预测", type="primary"):
            x = pd.DataFrame([inputs], columns=FEATURES)
            proba = pipe.predict_proba(x)[0]
            pred = int(pipe.predict(x)[0])
            c1, c2 = st.columns(2)
            c1.metric("患病概率", f"{proba[1] * 100:.1f}%")
            c2.metric("预测结果", "糖尿病" if pred == 1 else "非糖尿病")
            if pred == 1:
                st.warning("模型预测为糖尿病高风险，建议就医进一步检查。")
            else:
                st.success("模型预测为低风险，保持健康生活方式。")

    with tab6:
        st.subheader("模型解释（SHAP 个体预测解释）")
        st.write("调整下方 8 项指标，查看各特征对预测结果的贡献。")
        pipe, explainer, _ = get_explainer(tag)
        scaler = pipe.named_steps["scaler"]
        defaults = df[FEATURES].median()
        c1, c2 = st.columns(2)
        inputs = {}
        for i, c in enumerate(FEATURES):
            col = c1 if i % 2 == 0 else c2
            lo, hi = float(df[c].min()), float(df[c].max())
            inputs[c] = col.number_input(
                FEATURE_LABELS[c], min_value=lo, max_value=hi * 1.5,
                value=float(defaults[c]), step=0.1 if c in
                ["BMI", "DiabetesPedigreeFunction"] else 1.0,
                key=f"exp_{c}")
        x = pd.DataFrame([inputs], columns=FEATURES)
        x_scaled = scaler.transform(x)
        sv = explainer(x_scaled)
        if sv.values.ndim == 3:
            sv = sv[..., 1]
        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(sv[0], max_display=8, show=False)
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
