"""全局配置：路径、字段常量、随机种子。"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_OUT = os.path.join(OUTPUT_DIR, "data")
FIG_OUT = os.path.join(OUTPUT_DIR, "figures")
MODEL_OUT = os.path.join(OUTPUT_DIR, "models")

FHGD_PATH = os.path.join(DATA_DIR, "FHGD.csv")
PIMA_DIR = os.path.join(DATA_DIR, "pima")

ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
IQR_MULT = 1.5
RANDOM_STATE = 42

for _d in (DATA_OUT, FIG_OUT, MODEL_OUT):
    os.makedirs(_d, exist_ok=True)
