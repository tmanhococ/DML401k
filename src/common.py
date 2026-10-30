"""Shared config, data loading, learners and plotting style for the 401(k) DML experiments.

Notation (matches the theory slides): Y = net_tfa, D = e401 (treatment), Z = e401 (instrument
for participation D = p401), X = 9 controls. Nuisances: l(X) = E[Y|X], m(X) = E[D|X],
g(D, X) = E[Y|D, X].

Set the environment variable DML_FAST=1 for a quick smoke run (100 trees, fewer repetitions).
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "PVW_data.dta"
FAST = os.environ.get("DML_FAST", "0") == "1"
OUT = ROOT / ("outputs_fast" if FAST else "outputs")
TAB = OUT / "tables"
FIG = OUT / "figs"
for p in (TAB, FIG):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20241111
N_TREES = 100 if FAST else 1000
N_FOLDS = 10
TRIM = 0.001  # propensity clipping for IRM / IIVM, as in the guide

Y, D, P = "net_tfa", "e401", "p401"
X = ["age", "tw", "inc", "fsize", "db", "marr", "twoearn", "pira", "hown"]
X_LABELS = {
    "age": "Tuổi chủ hộ",
    "tw": "Tổng tài sản",
    "inc": "Thu nhập",
    "fsize": "Quy mô hộ",
    "db": "Có lương hưu DB",
    "marr": "Đã kết hôn",
    "twoearn": "Hai người có thu nhập",
    "pira": "Có tài khoản IRA",
    "hown": "Sở hữu nhà",
}

# Palette (validated reference instance from the dataviz skill, light mode)
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "violet": "#6250d6",
    "red": "#e34948",
    "gray": "#8a8984",
    "grid": "#e6e5e1",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "surface": "#ffffff",
}


def load_data():
    df = pd.read_stata(DATA)
    for c in df.columns:
        df[c] = df[c].astype(float)
    return df


def rf_reg(depth, seed=SEED):
    return RandomForestRegressor(n_estimators=N_TREES, max_depth=depth, random_state=seed, n_jobs=-1)


def rf_clf(depth, seed=SEED):
    return RandomForestClassifier(n_estimators=N_TREES, max_depth=depth, random_state=seed, n_jobs=-1)


def summarize(name, estimand, coef, se, n, extra=None):
    from scipy.stats import norm

    z = coef / se
    row = {
        "model": name,
        "estimand": estimand,
        "coef": coef,
        "se": se,
        "ci_lo": coef - 1.959964 * se,
        "ci_hi": coef + 1.959964 * se,
        "pval": 2 * (1 - norm.cdf(abs(z))),
        "n_obs": n,
    }
    if extra:
        row.update(extra)
    return row


def dml_row(name, estimand, model, extra=None):
    coef, se = float(model.coef[0]), float(model.se[0])
    ci = model.confint(level=0.95).iloc[0]
    row = summarize(name, estimand, coef, se, model.n_obs, extra)
    row["ci_lo"], row["ci_hi"] = float(ci.iloc[0]), float(ci.iloc[1])
    row["pval"] = float(model.pval[0])
    return row


def setup_style():
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 13,
            "axes.labelcolor": C["ink2"],
            "axes.edgecolor": C["grid"],
            "axes.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": C["grid"],
            "grid.linewidth": 1,
            "xtick.color": C["ink2"],
            "ytick.color": C["ink2"],
            "text.color": C["ink"],
            "legend.frameon": False,
            "figure.facecolor": C["surface"],
            "axes.facecolor": C["surface"],
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def money(x, pos=None):
    sign = "−" if x < 0 else ""
    x = abs(x)
    if x < 1000:
        return f"{sign}${x:,.0f}"
    k = f"{x / 1000:,.1f}".rstrip("0").rstrip(".")
    return f"{sign}${k}K"


def save_fig(fig, name):
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def save_table(df, name):
    path = TAB / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  saved {path.relative_to(ROOT)}")
    return path


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.perf_counter()
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} ...", flush=True)
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} done in {self.elapsed:,.1f}s", flush=True)


def write_env():
    import doubleml
    import scipy
    import sklearn

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "doubleml": doubleml.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "n_trees": N_TREES,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "fast_mode": FAST,
        "cpu_count": os.cpu_count(),
    }
    (OUT / "environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
