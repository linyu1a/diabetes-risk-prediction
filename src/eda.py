"""探索性数据分析：描述统计、类别分布、相关性、关键特征关联，输出图表。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_OUT, FIG_OUT, FEATURES, TARGET
from plotting import setup_plot

setup_plot()

KEY_FEATURES = ["Glucose", "BMI", "Age", "Insulin", "BloodPressure"]


def load_cleaned(tag):
    return pd.read_csv(f"{DATA_OUT}/{tag}_cleaned.csv")


def class_distribution(dfs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (tag, df) in zip(axes, dfs.items()):
        counts = df[TARGET].value_counts().sort_index()
        ax.bar(["非糖尿病 (0)", "糖尿病 (1)"], counts.values,
               color=["#4C72B0", "#DD8452"])
        for i, v in enumerate(counts.values):
            ax.text(i, v + 1, str(v), ha="center", fontsize=11)
        ax.set_title(f"{tag} 目标变量分布")
        ax.set_ylabel("样本数")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def correlation_heatmap(df, tag, path):
    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, cbar_kws={"label": "相关系数"})
    ax.set_title(f"{tag} 特征相关性热力图")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return corr


def feature_distribution(df, tag, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for i, c in enumerate(KEY_FEATURES):
        for outcome, color, label in [(0, "#4C72B0", "非糖尿病"), (1, "#DD8452", "糖尿病")]:
            sub = df[df[TARGET] == outcome][c]
            axes[i].hist(sub, bins=20, alpha=0.55, color=color, label=label, density=True)
        axes[i].set_title(c)
        axes[i].legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle(f"{tag} 关键特征分布（按糖尿病与否）", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def feature_boxplot(df, tag, path):
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for i, c in enumerate(KEY_FEATURES):
        sns.boxplot(x=TARGET, y=c, data=df, ax=axes[i], hue=TARGET,
                    palette=["#4C72B0", "#DD8452"], legend=False)
        axes[i].set_title(c)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["非糖尿病", "糖尿病"])
        axes[i].set_xlabel("")
    fig.suptitle(f"{tag} 关键特征箱线图对比", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def glucose_bmi_scatter(df, tag, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for outcome, color, label in [(0, "#4C72B0", "非糖尿病"), (1, "#DD8452", "糖尿病")]:
        sub = df[df[TARGET] == outcome]
        ax.scatter(sub["Glucose"], sub["BMI"], s=20, alpha=0.6, color=color, label=label)
    ax.set_xlabel("Glucose (血糖)")
    ax.set_ylabel("BMI (体质指数)")
    ax.set_title(f"{tag} Glucose 与 BMI 散点分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def association_analysis(df):
    """关键特征与 Outcome 的点二列相关 + 分组均值。"""
    rows = []
    for c in KEY_FEATURES:
        r = df[c].corr(df[TARGET])
        m0, m1 = df.loc[df[TARGET] == 0, c].mean(), df.loc[df[TARGET] == 1, c].mean()
        rows.append({"feature": c, "point_biserial_r": round(r, 4),
                     "mean_non_diabetic": round(m0, 2), "mean_diabetic": round(m1, 2),
                     "mean_diff": round(m1 - m0, 2)})
    return pd.DataFrame(rows).sort_values("point_biserial_r", key=abs, ascending=False)


def main():
    dfs = {"FHGD": load_cleaned("FHGD"), "pima": load_cleaned("pima")}

    print("\n===== 描述性统计 =====")
    describe = {}
    for tag, df in dfs.items():
        d = df[FEATURES + [TARGET]].describe().T
        describe[tag] = d
        d.to_csv(f"{DATA_OUT}/{tag}_describe.csv")
        print(f"\n[{tag}] 样本数 {len(df)}, 类别分布 {df[TARGET].value_counts().sort_index().to_dict()}")
        print(d[["mean", "std", "min", "50%", "max"]].round(3))

    class_distribution(dfs, f"{FIG_OUT}/class_distribution.png")
    print("\n已保存: class_distribution.png")

    print("\n===== 特征与 Outcome 关联分析 =====")
    for tag, df in dfs.items():
        corr = correlation_heatmap(df, tag, f"{FIG_OUT}/correlation_{tag}.png")
        assoc = association_analysis(df)
        assoc.to_csv(f"{DATA_OUT}/{tag}_association.csv", index=False)
        print(f"\n[{tag}] 点二列相关排序:")
        print(assoc.to_string(index=False))
        print(f"  已保存: correlation_{tag}.png, {tag}_association.csv")

        feature_distribution(df, tag, f"{FIG_OUT}/dist_{tag}.png")
        feature_boxplot(df, tag, f"{FIG_OUT}/boxplot_{tag}.png")
        glucose_bmi_scatter(df, tag, f"{FIG_OUT}/scatter_{tag}.png")
        print(f"  已保存: dist_{tag}.png, boxplot_{tag}.png, scatter_{tag}.png")

    print("\nEDA 完成。")


if __name__ == "__main__":
    main()
