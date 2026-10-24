"""机器学习模型构建与对比。

四个模型：逻辑回归、随机森林、XGBoost、SVM。
流程：SMOTE + StandardScaler + 分类器 组成 Pipeline，
      5 折分层交叉验证评估，GridSearchCV 超参数调优，
      报告 Accuracy/Precision/Recall/F1/ROC-AUC 五项指标。
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import (StratifiedKFold, GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from config import DATA_OUT, MODEL_OUT, FIG_OUT, FEATURES, TARGET, RANDOM_STATE
from plotting import setup_plot

setup_plot()

SCORING = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC-AUC": "roc_auc",
}
N_SPLITS = 5


def build_models():
    return {
        "逻辑回归": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "随机森林": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, verbosity=0),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }


PARAM_GRIDS = {
    "逻辑回归": {"clf__C": [0.01, 0.1, 1, 10]},
    "随机森林": {"clf__n_estimators": [100, 300],
                 "clf__max_depth": [None, 5, 10]},
    "XGBoost": {"clf__n_estimators": [100, 300],
                "clf__max_depth": [3, 6],
                "clf__learning_rate": [0.01, 0.1]},
    "SVM": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"]},
}


def make_pipeline(model):
    return Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
        ("clf", model),
    ])


def load_cleaned(tag):
    return pd.read_csv(f"{DATA_OUT}/{tag}_cleaned.csv")


def evaluate(model_name, model, X, y, tuned):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipe = make_pipeline(model)
    best_params = None
    if tuned:
        grid = GridSearchCV(pipe, PARAM_GRIDS[model_name], cv=cv,
                            scoring="roc_auc", n_jobs=-1)
        grid.fit(X, y)
        pipe = grid.best_estimator_
        best_params = {k.split("__")[-1]: v for k, v in grid.best_params_.items()}
    else:
        pipe.fit(X, y)
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    scores = {m: {"mean": float(res[f"test_{m}"].mean()),
                  "std": float(res[f"test_{m}"].std())}
              for m in SCORING}
    return scores, best_params, pipe


def train(tag, X, y):
    models = build_models()
    base_rows, tuned_rows = [], []
    tuned_pipes = {}
    best_model = {"name": None, "auc": -1, "pipe": None, "params": None}
    for name, model in models.items():
        base, _, _ = evaluate(name, model, X, y, tuned=False)
        tuned, params, pipe = evaluate(name, model, X, y, tuned=True)
        tuned_pipes[name] = pipe
        base_rows.append({"模型": name, **{k: v["mean"] for k, v in base.items()}})
        tuned_rows.append({"模型": name, **{k: v["mean"] for k, v in tuned.items()}})
        print(f"  [{tag}] {name}: 调优后 AUC={tuned['ROC-AUC']['mean']:.4f} "
              f"(±{tuned['ROC-AUC']['std']:.4f})  参数={params}")
        if tuned["ROC-AUC"]["mean"] > best_model["auc"]:
            best_model = {"name": name, "auc": tuned["ROC-AUC"]["mean"],
                          "pipe": pipe, "params": params,
                          "scores": {k: v for k, v in tuned.items()}}

    base_df = pd.DataFrame(base_rows).set_index("模型")
    tuned_df = pd.DataFrame(tuned_rows).set_index("模型")
    base_df.to_csv(f"{MODEL_OUT}/{tag}_baseline.csv")
    tuned_df.to_csv(f"{MODEL_OUT}/{tag}_tuned.csv")

    joblib.dump(best_model["pipe"], f"{MODEL_OUT}/{tag}_best_model.joblib")
    with open(f"{MODEL_OUT}/{tag}_best_meta.json", "w", encoding="utf-8") as f:
        json.dump({"name": best_model["name"], "params": best_model["params"],
                   "auc": best_model["auc"],
                   "scores": {k: {"mean": v["mean"], "std": v["std"]}
                              for k, v in best_model["scores"].items()}},
                  f, ensure_ascii=False, indent=2)
    return tuned_df, best_model, tuned_pipes


def comparison_plot(results, path):
    tags = list(results.keys())
    metrics = list(SCORING.keys())
    models = list(results[tags[0]].index)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    for i, tag in enumerate(tags):
        df = results[tag].T
        df.columns = models
        df.index = metrics
        df.T.plot(kind="bar", ax=axes[i], rot=0)
        axes[i].set_title(f"{tag} 调优后指标对比")
        axes[i].set_ylim(0, 1.05)
        axes[i].set_xlabel("模型")
        axes[i].legend(title="指标", fontsize=8, ncol=2)
        axes[i].grid(axis="y", alpha=0.3)
    axes[2].axis("off")
    axes[3].axis("off")
    fig.suptitle("四模型五项指标对比（5 折分层交叉验证，调优后）", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def roc_curve_plot(tag, X, y, pipes, path):
    """在 holdout 测试集上绘制四模型 ROC 曲线对比（使用调优后模型）。"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in pipes.items():
        pipe = clone(pipe).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机分类 (AUC=0.500)")
    ax.set_xlabel("假阳性率 (FPR)")
    ax.set_ylabel("真阳性率 (TPR)")
    ax.set_title(f"{tag} 四模型 ROC 曲线对比（测试集）")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def confusion_matrix_plot(tag, model_name, pipe, X, y, path):
    """最优模型在 holdout 测试集上的混淆矩阵。"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    pipe = clone(pipe).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    cm = confusion_matrix(y_te, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["非糖尿病", "糖尿病"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{tag} 最优模型（{model_name}）混淆矩阵")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    results = {}
    for tag in ["FHGD", "pima"]:
        df = load_cleaned(tag)
        X, y = df[FEATURES], df[TARGET]
        print(f"\n===== {tag}: {len(df)} 样本, 类别 {y.value_counts().sort_index().to_dict()} =====")
        tuned_df, best, tuned_pipes = train(tag, X, y)
        results[tag] = tuned_df
        print(f"  最优模型: {best['name']} (ROC-AUC={best['auc']:.4f}), "
              f"已保存 {tag}_best_model.joblib")

        roc_curve_plot(tag, X, y, tuned_pipes, f"{FIG_OUT}/roc_curve_{tag}.png")
        confusion_matrix_plot(tag, best["name"], best["pipe"], X, y,
                              f"{FIG_OUT}/confusion_matrix_{tag}.png")
        print(f"  已保存: roc_curve_{tag}.png, confusion_matrix_{tag}.png")

    comparison_plot(results, f"{FIG_OUT}/model_comparison.png")
    print("\n已保存: model_comparison.png")
    print("模型构建与对比完成。")


if __name__ == "__main__":
    main()
