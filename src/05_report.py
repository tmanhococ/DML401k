"""Collect all tables into the headline forest plot and a markdown summary for review."""
import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from common import C, N_FOLDS, N_TREES, OUT, TAB, money, save_fig, save_table, setup_style

setup_style()


def read(name):
    p = TAB / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


base, main = read("e1_baselines"), read("e3_dml_main")
allres = pd.concat([base, main], ignore_index=True)
save_table(allres, "results_main")

# ---------- headline forest plot ----------
panels = [
    ("e401", "Đủ điều kiện 401(k) (e401) → net_tfa",
     ["Chênh lệch trung bình thô", "OLS + 9 biến kiểm soát (tuyến tính)", "OLS + bậc 2 & tương tác", "DML-PLR",
      "DML-IRM (ATE)", "DML-IRM (ATTE)"]),
    ("p401", "Tham gia 401(k) (p401) → net_tfa",
     ["Chênh lệch thô (p401)", "OLS + 9 biến (p401)", "2SLS tuyến tính (p401 ~ e401)", "DML-PLIV", "DML-IIVM (LATE)"]),
]
fig, axes = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw={"hspace": 0.45})
for ax, (est, title, order) in zip(axes, panels):
    sub = allres[allres.estimand == est].set_index("model")
    order = [o for o in order if o in sub.index][::-1]
    for i, mname in enumerate(order):
        r = sub.loc[mname]
        col = C["blue"] if r.family == "dml" else C["gray"]
        ax.hlines(i, r.ci_lo, r.ci_hi, color=col, lw=2.5)
        ax.scatter(r.coef, i, s=90, color=col, edgecolor="white", linewidth=2, zorder=3)
        ax.text(r.ci_hi + 300, i, f"{r.coef:,.0f}  (SE {r.se:,.0f})", va="center", fontsize=11, color=C["ink"])
    ax.set_yticks(range(len(order)), order)
    ax.axvline(0, color=C["ink2"], lw=1)
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.set_title(title)
    ax.set_xlabel("Ước lượng (CI 95%)")
    ax.grid(axis="y", visible=False)
    lo, hi = sub.loc[order, "ci_lo"].min(), sub.loc[order, "ci_hi"].max()
    ax.set_xlim(min(0, lo) - 1000, hi + (hi - lo) * 0.45)
fig.text(0.01, 0.0, f"Xám: baseline không dùng ML.  Xanh: DML với Random Forest ({N_TREES} cây, {N_FOLDS} folds).", fontsize=11,
         color=C["ink2"])
save_fig(fig, "e3_forest_plot")

# ---------- markdown summary ----------
env = json.loads((OUT / "environment.json").read_text(encoding="utf-8"))
fmt = lambda d: d.to_markdown(index=False, floatfmt=",.2f") if not d.empty else "_(missing)_"
cols = ["model", "estimand", "coef", "se", "ci_lo", "ci_hi", "pval"]
sections = [
    ("Environment", "```json\n" + json.dumps(env, indent=2) + "\n```"),
    ("E1 + E3 + E6: main results", fmt(allres[cols])),
    ("E3: vs published guide values", fmt(read("e3_vs_guide"))),
    ("E0: covariate balance", fmt(read("e0_balance_table"))),
    ("E2: DML by hand", fmt(read("e2_dml_by_hand")[["model", "coef", "se", "ci_lo", "ci_hi"]] if not read("e2_dml_by_hand").empty else pd.DataFrame())),
    ("E2: overfitting check (R²)", fmt(read("e2_overfit_check"))),
    ("E4: nuisance metrics", fmt(read("e4_nuisance_metrics"))),
    ("E4: overlap summary", fmt(read("e4_overlap_summary"))),
    ("E5a: repeated cross-fitting", fmt(read("e5a_repeated_crossfit_summary"))),
    ("E5b: learners", fmt(read("e5b_learners").drop(columns=["estimand", "n_obs"], errors="ignore"))),
    ("E5c: folds", fmt(read("e5c_folds")[cols + ["n_folds"]] if not read("e5c_folds").empty else pd.DataFrame())),
    ("E7: sensitivity", fmt(read("e7_sensitivity_summary"))),
    ("E7: benchmarks", fmt(read("e7_sensitivity_benchmarks"))),
    ("E8: GATE by income quintile", fmt(read("e8_gate_income"))),
]
md = "# 401(k) DML experiments: results\n\n" + "\n\n".join(f"## {h}\n\n{b}" for h, b in sections) + "\n"
(OUT / "results_summary.md").write_text(md, encoding="utf-8")
print(f"  saved {OUT / 'results_summary.md'}")
