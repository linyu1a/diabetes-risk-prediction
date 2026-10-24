# 糖尿病风险预测与可视化系统

基于 Pima 与 FHGD 糖尿病数据集（2768 条记录、8 个特征）的风险预测与可视化系统，
包含数据预处理、探索性分析、四模型对比（逻辑回归/随机森林/XGBoost/SVM）、
SHAP 可解释性分析，以及 Streamlit 交互式 Web 应用。

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 复现完整流程

```bash
python src/preprocess.py       # 数据预处理
python src/eda.py              # 探索性数据分析
python src/train.py            # 模型训练与对比
python src/shap_analysis.py    # SHAP 可解释性
```

## 部署到 Streamlit Cloud

1. 将本项目推送到 GitHub 仓库；
2. 打开 <https://share.streamlit.io>，点击 "New app"；
3. 选择仓库、分支与主文件 `app.py`，点击 "Deploy"；
4. 部署完成后即可获得公开访问链接。

## 目录结构

```
├── data/              原始数据
├── output/            预处理结果、图表、模型（已生成）
├── src/               预处理/EDA/训练/SHAP 脚本
├── app.py             Streamlit 应用
├── 讲解文档.md        详细讲解文档
└── requirements.txt   依赖清单
```
