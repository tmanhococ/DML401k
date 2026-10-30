"""E2: DML for the PLR model "by hand" (residual-on-residual with cross-fitting), compared with
DoubleMLPLR on the same folds, plus two broken variants that show why each ingredient matters:
  - no cross-fitting (nuisances fitted and predicted on the same data)
  - no orthogonalization (only Y is residualized, D is used raw)
"""
import doubleml as dml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter
from sklearn.base import clone
from sklearn.model_selection import KFold

import plots
from common import (C, D, N_FOLDS, SEED, TAB, X, Y, Timer, load_data, money, rf_clf, rf_reg, save_fig, save_table,
                    setup_style, summarize)

setup_style()
df = load_data()
n = len(df)
y, d, x = df[Y].values, df[D].values, df[X].values
ml_l, ml_m = rf_reg(8), rf_clf(4)

folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(x))


def theta_se(y_res, d_res):
    theta = np.sum(d_res * y_res) / np.sum(d_res ** 2)
    psi = (y_res - theta * d_res) * d_res
    J = -np.mean(d_res ** 2)
    se = np.sqrt(np.mean(psi ** 2) / J ** 2 / len(y_res))
    return theta, se


# (a) cross-fitted residuals
l_hat, m_hat = np.empty(n), np.empty(n)
with Timer("E2 cross-fitting by hand"):
    for k, (tr, te) in enumerate(folds):
        l_hat[te] = clone(ml_l).fit(x[tr], y[tr]).predict(x[te])
        m_hat[te] = clone(ml_m).fit(x[tr], d[tr]).predict_proba(x[te])[:, 1]
y_res, d_res = y - l_hat, d - m_hat
th_a, se_a = theta_se(y_res, d_res)

# (b) no cross-fitting: in-sample predictions
with Timer("E2 in-sample nuisances"):
    l_in = clone(ml_l).fit(x, y).predict(x)
    m_in = clone(ml_m).fit(x, d).predict_proba(x)[:, 1]
th_b, se_b = theta_se(y - l_in, d - m_in)

# (c) not orthogonal: residualize Y only, regress on raw D
mc = sm.OLS(y_res, sm.add_constant(d)).fit(cov_type="HC1")
th_c, se_c = mc.params[1], mc.bse[1]

# (d) DoubleML on the very same folds
with Timer("E2 DoubleMLPLR same folds"):
    data = dml.DoubleMLData(df, y_col=Y, d_cols=D, x_cols=X)
    plr = dml.DoubleMLPLR(data, ml_l=clone(ml_l), ml_m=clone(ml_m), n_folds=N_FOLDS, draw_sample_splitting=False)
    plr.set_sample_splitting([[(tr, te) for tr, te in folds]])
    plr.fit()
th_d, se_d = float(plr.coef[0]), float(plr.se[0])

rows = [
    summarize("DML thủ công: cross-fitting + trực giao", "e401", th_a, se_a, n, {"variant": "manual"}),
    summarize("DoubleMLPLR (cùng folds)", "e401", th_d, se_d, n, {"variant": "library"}),
    summarize("Bỏ cross-fitting (dự báo in-sample)", "e401", th_b, se_b, n, {"variant": "no_crossfit"}),
    summarize("Bỏ trực giao (chỉ khử X khỏi Y)", "e401", th_c, se_c, n, {"variant": "no_orthogonal"}),
]
out = pd.DataFrame(rows)
save_table(out, "e2_dml_by_hand")
print(out[["model", "coef", "se"]].to_string(index=False))
print(f"  |manual - library| = {abs(th_a - th_d):.6f}")


def r2(t, p):
    return 1 - np.mean((t - p) ** 2) / np.var(t)


fit_tab = pd.DataFrame(
    [
        {"nuisance": "l(X)=E[Y|X]", "r2_out_of_fold": r2(y, l_hat), "r2_in_sample": r2(y, l_in)},
        {"nuisance": "m(X)=E[D|X]", "r2_out_of_fold": r2(d, m_hat), "r2_in_sample": r2(d, m_in)},
    ]
)
save_table(fit_tab, "e2_overfit_check")
print(fit_tab.to_string(index=False))

# residuals for later figures / slides
pd.DataFrame({"y": y, "d": d, "l_hat": l_hat, "m_hat": m_hat, "y_res": y_res, "d_res": d_res}).to_csv(
    TAB / "e2_residuals.csv", index=False
)

# ---------- figure: before vs after partialling out ----------
lo, hi = np.quantile(y, [0.02, 0.98])
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

ax = axes[0]
jit = d + rng.uniform(-0.12, 0.12, n)
keep = (y >= lo) & (y <= hi)
ax.scatter(jit[keep], y[keep], s=6, alpha=0.15, color=C["gray"], lw=0)
m0, m1 = y[d == 0].mean(), y[d == 1].mean()
ax.plot([0, 1], [m0, m1], color=C["orange"], lw=2.5)
ax.scatter([0, 1], [m0, m1], s=80, color=C["orange"], edgecolor="white", linewidth=2, zorder=3)
ax.text(0.5, (m0 + m1) / 2 + (hi - lo) * 0.06, f"độ dốc = {m1 - m0:,.0f}", color=C["ink"], fontsize=12,
        ha="center", va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85))
ax.set_xticks([0, 1], ["e401 = 0", "e401 = 1"])
ax.set_ylim(lo, hi)
ax.yaxis.set_major_formatter(FuncFormatter(money))
ax.set_ylabel("Y = net_tfa")
ax.set_title("Trước: Y theo D (so sánh thô)")

ax = axes[1]
rlo, rhi = np.quantile(y_res, [0.02, 0.98])
keep = (y_res >= rlo) & (y_res <= rhi)
ax.scatter(d_res[keep], y_res[keep], s=6, alpha=0.12, color=C["gray"], lw=0)
q = pd.qcut(d_res, 20, labels=False, duplicates="drop")
bs = pd.DataFrame({"x": d_res, "y": y_res, "q": q}).groupby("q").mean()
ax.scatter(bs.x, bs.y, s=60, color=C["blue"], edgecolor="white", linewidth=2, zorder=3, label="Trung bình theo 20 nhóm")
xx = np.linspace(d_res.min(), d_res.max(), 2)
ax.plot(xx, th_a * xx, color=C["blue"], lw=2.5, label=f"Độ dốc = θ ước lượng = {th_a:,.0f}")
ax.set_ylim(rlo, rhi)
ax.yaxis.set_major_formatter(FuncFormatter(money))
ax.set_xlabel("D̃ = D − m̂(X)   (phần 'bất ngờ' của việc đủ điều kiện)")
ax.set_ylabel("Ỹ = Y − l̂(X)")
ax.set_title("Sau: hồi quy phần dư Ỹ lên D̃ (DML-PLR)")
ax.legend(loc="upper left")
fig.text(0.01, -0.02, "Điểm xám: từng hộ (chỉ vẽ phân vị 2%–98% của trục dọc; ước lượng dùng toàn bộ mẫu). Phần dư tính out-of-fold, 10 folds, Random Forest.",
         fontsize=10, color=C["ink2"])
save_fig(fig, "e2_residual_on_residual")

# ---------- figure: variants ----------
plots.e2_variants(out)
