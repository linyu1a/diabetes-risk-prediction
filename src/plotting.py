"""绘图公共配置：中文字体、样式。所有绘图脚本与 Streamlit 应用共用。"""
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot():
    sns.set_style("whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK JP"]
    plt.rcParams["axes.unicode_minus"] = False
