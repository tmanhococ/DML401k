"""E0 + E1: covariate balance between eligible / non-eligible households, and naive baselines."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter

from common import C, D, P, X, X_LABELS, Y, load_data, money, save_fig, save_table, setup_style, summarize, write_env

setup_style()
df = load_data()
write_env()
n = len(df)

# ---------- E0: descriptives & balance ----------
desc = df[[Y, D, P] + X].describe().T.reset_index().rename(columns={"index": "variable"})
save_table(desc, "e0_descriptives")

rows = []
for v in [Y] + X:
    a, b = df.loc[df[D] == 1, v], df.loc[df[D] == 0, v]
    smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    rows.append(
        {
            "variable": v,
            "label": X_LABELS.get(v, "Tài sản tài chính ròng (Y)"),
            "mean_eligible": a.mean(),
            "mean_not_eligible": b.mean(),
            "diff": a.mean() - b.mean(),
            "smd": smd,
        }
    )
bal = pd.DataFrame(rows)
save_table(bal, "e0_balance_table")
counts = df[D].value_counts().rename({1.0: "eligible", 0.0: "not_eligible"})
save_table(
    pd.DataFrame(
        {
            "group": counts.index,
            "n": counts.values,
            "share": counts.values / n,
            "participation_rate": [df.loc[df[D] == (1.0 if g == "eligible" else 0.0), P].mean() for g in counts.index],
        }
    ),
    "e0_group_sizes",
)

# Love plot: standardized mean differences of X
b = bal[bal.variable != Y].sort_values("smd")
fig, ax = plt.subplots(figsize=(8, 5))
ax.axvspan(-0.1, 0.1, color=C["grid"], alpha=0.6, lw=0)
ax.axvline(0, color=C["ink2"], lw=1)
ax.hlines(range(len(b)), 0, b.smd, color=C["blue"], lw=2)
ax.scatter(b.smd, range(len(b)), s=70, color=C["blue"], edgecolor="white", linewidth=2, zorder=3)
for i, s in enumerate(b.smd):
    ax.text(s + (0.02 if s >= 0 else -0.02), i, f"{s:+.2f}", va="center", ha="left" if s >= 0 else "right",
            fontsize=11, color=C["ink2"])
ax.set_yticks(range(len(b)), b.label)
ax.set_xlabel("Chênh lệch chuẩn hóa (SMD): đủ điều kiện − không đủ điều kiện")
ax.set_title("Hai nhóm khác nhau ngay từ đầu: confounding")
ax.text(0, len(b) - 0.4, "|SMD| < 0.1\n= cân bằng", ha="center", va="bottom", fontsize=10, color=C["ink2"])
ax.set_ylim(-0.6, len(b) + 0.5)
ax.grid(axis="y", visible=False)
xmax = max(abs(b.smd).max() + 0.15, 0.4)
ax.set_xlim(-xmax, xmax)
save_fig(fig, "e0_love_plot")

# Outcome distribution by group (heavy tails -> why SEs are large)
fig, ax = plt.subplots(figsize=(8, 4.5))
clip = df[Y].clip(df[Y].quantile(0.01), df[Y].quantile(0.99))
bins = np.linspace(clip.min(), clip.max(), 60)
for g, col, lab in [(0, C["gray"], "Không đủ điều kiện (e401=0)"), (1, C["blue"], "Đủ điều kiện (e401=1)")]:
    ax.hist(clip[df[D] == g], bins=bins, density=True, alpha=0.55, color=col, label=lab, edgecolor="white", lw=0.5)
    ax.axvline(df.loc[df[D] == g, Y].mean(), color=col, lw=2, ls="-")
ax.xaxis.set_major_formatter(FuncFormatter(money))
ax.set_yticks([])
ax.set_xlabel("net_tfa (cắt ở phân vị 1%–99% để hiển thị; vạch dọc = trung bình)")
ax.set_title("Phân phối tài sản tài chính ròng theo nhóm")
ax.legend(loc="upper right")
ax.grid(False)
save_fig(fig, "e0_outcome_dist")

# ---------- E1: naive baselines ----------
res = []
diff = df.loc[df[D] == 1, Y].mean() - df.loc[df[D] == 0, Y].mean()
m0 = sm.OLS(df[Y], sm.add_constant(df[[D]])).fit(cov_type="HC1")
res.append(summarize("Chênh lệch trung bình thô", "e401", m0.params[D], m0.bse[D], n, {"family": "baseline"}))
m1 = sm.OLS(df[Y], sm.add_constant(df[[D] + X])).fit(cov_type="HC1")
res.append(summarize("OLS + 9 biến kiểm soát (tuyến tính)", "e401", m1.params[D], m1.bse[D], n, {"family": "baseline"}))

# Flexible OLS: quadratic terms + interactions of continuous vars, a common "manual" fix
Xf = df[X].copy()
for v in ["age", "inc", "tw", "fsize"]:
    Xf[f"{v}^2"] = df[v] ** 2
for i, a in enumerate(["age", "inc", "tw"]):
    for bb in ["age", "inc", "tw"][i + 1:]:
        Xf[f"{a}*{bb}"] = df[a] * df[bb]
m2 = sm.OLS(df[Y], sm.add_constant(pd.concat([df[[D]], Xf], axis=1))).fit(cov_type="HC1")
res.append(summarize("OLS + bậc 2 & tương tác", "e401", m2.params[D], m2.bse[D], n, {"family": "baseline"}))

# Participation baselines
m3 = sm.OLS(df[Y], sm.add_constant(df[[P]])).fit(cov_type="HC1")
res.append(summarize("Chênh lệch thô (p401)", "p401", m3.params[P], m3.bse[P], n, {"family": "baseline"}))
m4 = sm.OLS(df[Y], sm.add_constant(df[[P] + X])).fit(cov_type="HC1")
res.append(summarize("OLS + 9 biến (p401)", "p401", m4.params[P], m4.bse[P], n, {"family": "baseline"}))
# Linear 2SLS: p401 instrumented by e401
first = sm.OLS(df[P], sm.add_constant(df[[D] + X])).fit()
phat = first.fittedvalues
Xs = sm.add_constant(pd.concat([phat.rename(P), df[X]], axis=1))
ss = sm.OLS(df[Y], Xs).fit()
# correct 2SLS residuals use actual p401
resid = df[Y] - sm.add_constant(df[[P] + X]).values @ ss.params.values
XtX_inv = np.linalg.inv(Xs.T.values @ Xs.values)
meat = (Xs.values * resid.values[:, None] ** 2).T @ Xs.values
V = XtX_inv @ meat @ XtX_inv * n / (n - Xs.shape[1])
res.append(summarize("2SLS tuyến tính (p401 ~ e401)", "p401", ss.params[P], float(np.sqrt(V[1, 1])), n,
                     {"family": "baseline"}))

out = pd.DataFrame(res)
save_table(out, "e1_baselines")
print(out[["model", "estimand", "coef", "se"]].to_string(index=False))
