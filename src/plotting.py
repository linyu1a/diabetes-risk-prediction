"""绘图公共配置：中文字体、样式。所有绘图脚本与 Streamlit 应用共用。

为保证在无中文字体的部署环境（如 Streamlit Cloud）中图表正常显示中文，
将字体文件打包在项目 fonts/ 目录，并在运行时动态注册。
"""
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")


def setup_plot():
    sns.set_style("whitegrid")
    candidates = ["Microsoft-YaHei.ttf", "SimHei.ttf"]
    installed = []
    for fname in candidates:
        path = os.path.join(_FONT_DIR, fname)
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                installed.append(name)
            except Exception:
                pass
    installed += ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = installed
    plt.rcParams["axes.unicode_minus"] = False
