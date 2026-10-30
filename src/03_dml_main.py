"""E3 + E4 + E6 + E7 + E8: replicate the dmlguide configuration and diagnose it.

E3  PLR and IRM(ATE) for eligibility e401           (RF 1000 trees, depth 8 / 4, 10 folds, trim 0.001)
E6  PLIV and IIVM(LATE) for participation p401, e401 as instrument
    (PLIV in DoubleML calls .predict() on ml_m / ml_r, so they must be regressors even for binary targets)
E4  overlap of the propensity score, nuisance fit quality
E7  sensitivity analysis to unobserved confounding (Chernozhukov et al. 2022) + benchmarks
E8  group ATEs by income quintile
"""
import pickle

import doubleml as dml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from doubleml.utils import PSProcessorConfig
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import log_loss, roc_auc_score

import plots
from common import (C, D, N_FOLDS, OUT, P, SEED, TAB, TRIM, X, Y, Timer, dml_row, load_data, money, rf_clf,
                    rf_reg, save_fig, save_table, setup_style)

setup_style()
df = load_data()
n = len(df)
ps_cfg = PSProcessorConfig(clipping_threshold=TRIM)

data_d = dml.DoubleMLData(df, y_col=Y, d_cols=D, x_cols=X)
data_iv = dml.DoubleMLData(df, y_col=Y, d_cols=P, x_cols=X, z_cols=D)
assert df.loc[df[D] == 0, P].sum() == 0, "one-sided non-compliance expected"

models, rows, timing = {}, [], []

specs = [
    ("PLR", "e401", lambda: dml.DoubleMLPLR(data_d, ml_l=rf_reg(8), ml_m=rf_clf(4), n_folds=N_FOLDS)),
    ("IRM (ATE)", "e401", lambda: dml.DoubleMLIRM(data_d, ml_g=rf_reg(8), ml_m=rf_clf(4), n_folds=N_FOLDS,
                                                 score="ATE", ps_processor_config=ps_cfg)),
    ("IRM (ATTE)", "e401", lambda: dml.DoubleMLIRM(data_d, ml_g=rf_reg(8), ml_m=rf_clf(4), n_folds=N_FOLDS,
                                                  score="ATTE", ps_processor_config=ps_cfg)),
    ("PLIV", "p401", lambda: dml.DoubleMLPLIV(data_iv, ml_l=rf_reg(8), ml_m=rf_reg(4), ml_r=rf_reg(4),
                                             n_folds=N_FOLDS)),
    ("IIVM (LATE)", "p401", lambda: dml.DoubleMLIIVM(data_iv, ml_g=rf_reg(8), ml_m=rf_clf(4), ml_r=rf_clf(4),
                                                    n_folds=N_FOLDS, score="LATE", ps_processor_config=ps_cfg,
                                                    subgroups={"always_takers": False, "never_takers": True})),
]
for name, est, make in specs:
    np.random.seed(SEED)
    with Timer(f"E3/E6 {name}") as t:
        m = make()
        m.fit()
    models[name] = m
    rows.append(dml_row(f"DML-{name}", est, m, {"family": "dml", "learner": "Random Forest", "fit_seconds": t.elapsed}))
    print(m.summary.round(2).to_string())

main = pd.DataFrame(rows)
save_table(main, "e3_dml_main")

# Reference values published on dmlguide.github.io (Python, DoubleML)
ref = pd.DataFrame(
    [
        ("DML-PLR", 6737.64, 909.32, 7061.52, 857.95, 7826, 907),
        ("DML-IRM (ATE)", 6344.02, 719.96, 6099.18, 780.94, 6901, 771),
        ("DML-PLIV", 10375.66, 1290.76, 9896.56, 1250.05, 10746, 1323),
        ("DML-IIVM (LATE)", 9343.24, 1045.90, 9313.53, 1148.57, 9611, 1205),
    ],
    columns=["model", "guide_python_coef", "guide_python_se", "guide_stata_coef", "guide_stata_se",
             "guide_r_coef", "guide_r_se"],
)
cmp = main[["model", "coef", "se"]].merge(ref, on="model", how="inner")
cmp["diff_vs_guide_python"] = cmp.coef - cmp.guide_python_coef
cmp["diff_in_se_units"] = cmp.diff_vs_guide_python / cmp.guide_python_se
save_table(cmp, "e3_vs_guide")
print(cmp.round(2).to_string(index=False))

# ---------- E4: nuisance quality & overlap ----------
y, d, z = df[Y].values, df[D].values, df[D].values
p = df[P].values


def pred(m, k):
    return m.predictions[k][:, 0, 0]


def reg_metrics(t, pr):
    rmse = np.sqrt(np.mean((t - pr) ** 2))
    return {"rmse": rmse, "r2_out_of_fold": 1 - rmse ** 2 / np.var(t)}


def clf_metrics(t, pr):
    pr = np.clip(pr, 1e-6, 1 - 1e-6)
    base = np.full_like(pr, t.mean())
    return {"rmse": np.sqrt(np.mean((t - pr) ** 2)), "log_loss": log_loss(t, pr), "log_loss_constant": log_loss(t, base),
            "auc": roc_auc_score(t, pr)}


nm = []
plr, irm = models["PLR"], models["IRM (ATE)"]
nm.append({"model": "PLR", "nuisance": "l(X) = E[Y|X]", **reg_metrics(y, pred(plr, "ml_l"))})
nm.append({"model": "PLR", "nuisance": "m(X) = E[D|X]", **clf_metrics(d, pred(plr, "ml_m"))})
nm.append({"model": "IRM", "nuisance": "g(0,X) = E[Y|D=0,X]", **reg_metrics(y[d == 0], pred(irm, "ml_g0")[d == 0])})
nm.append({"model": "IRM", "nuisance": "g(1,X) = E[Y|D=1,X]", **reg_metrics(y[d == 1], pred(irm, "ml_g1")[d == 1])})
nm.append({"model": "IRM", "nuisance": "m(X) = P(D=1|X)", **clf_metrics(d, pred(irm, "ml_m"))})
pliv = models["PLIV"]
nm.append({"model": "PLIV", "nuisance": "r(X) = E[p401|X]", **clf_metrics(p, pred(pliv, "ml_r"))})
nuis = pd.DataFrame(nm)
save_table(nuis, "e4_nuisance_metrics")
print(nuis.round(3).to_string(index=False))

ps = pred(irm, "ml_m")
ov = pd.DataFrame({"m_hat": ps, "e401": d})
q = ov.groupby("e401").m_hat.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).reset_index()
q["share_below_0.05"] = ov.groupby("e401").m_hat.apply(lambda s: (s < 0.05).mean()).values
q["share_above_0.95"] = ov.groupby("e401").m_hat.apply(lambda s: (s > 0.95).mean()).values
q["share_clipped_at_trim"] = ov.groupby("e401").m_hat.apply(lambda s: ((s <= TRIM) | (s >= 1 - TRIM)).mean()).values
save_table(q, "e4_overlap_summary")
ov.to_csv(TAB / "e4_propensity_scores.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4.8))
bins = np.linspace(0, 1, 41)
ax.hist(ps[d == 0], bins=bins, color=C["gray"], alpha=0.6, label=f"e401 = 0 (n={int((d == 0).sum()):,})",
        edgecolor="white", lw=0.5)
ax.hist(ps[d == 1], bins=bins, color=C["blue"], alpha=0.6, label=f"e401 = 1 (n={int((d == 1).sum()):,})",
        edgecolor="white", lw=0.5)
ax.set_xlabel("Propensity score m̂(X) = P̂(e401 = 1 | X)  (out-of-fold, Random Forest)")
ax.set_ylabel("Số hộ")
ax.set_title("Overlap: hai nhóm có cùng vùng m̂(X) không?")
ax.set_xlim(0, 1)
lo, hi = np.min(ps), np.max(ps)
ax.plot([], [], " ", label=f"m̂(X) nằm trong [{lo:.3f}, {hi:.3f}]")
ax.legend(loc="upper right")
save_fig(fig, "e4_overlap")

# ---------- E7: sensitivity to unobserved confounding ----------
sens_rows, bench_rows = [], []
for name in ["PLR", "IRM (ATE)"]:
    m = models[name]
    m.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    sp = m.sensitivity_params
    sens_rows.append(
        {
            "model": name,
            "scenario": "cf_y=0.03, cf_d=0.03, rho=1",
            "theta": float(m.coef[0]),
            "theta_lower": float(np.ravel(sp["theta"]["lower"])[0]),
            "theta_upper": float(np.ravel(sp["theta"]["upper"])[0]),
            "ci_lower": float(np.ravel(sp["ci"]["lower"])[0]),
            "ci_upper": float(np.ravel(sp["ci"]["upper"])[0]),
            "RV": float(np.ravel(sp["rv"])[0]),
            "RVa": float(np.ravel(sp["rva"])[0]),
        }
    )
    print(m.sensitivity_summary)
    for bset in (["inc"], ["pira"], ["twoearn"], ["tw"]):
        with Timer(f"E7 benchmark {name} without {bset}"):
            try:
                b = m.sensitivity_benchmark(benchmarking_set=bset)
                bench_rows.append({"model": name, "benchmark": "+".join(bset), **{k: float(v) for k, v in b.iloc[0].items()}})
            except Exception as e:  # keep the overnight run going
                print(f"  benchmark failed: {e!r}")
sens = pd.DataFrame(sens_rows)
bench = pd.DataFrame(bench_rows)
save_table(sens, "e7_sensitivity_summary")
save_table(bench, "e7_sensitivity_benchmarks")

grid = np.linspace(0.0, 0.15, 31)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, name in zip(axes, ["PLR", "IRM (ATE)"]):
    m = models[name]
    Z = np.empty((len(grid), len(grid)))
    for i, cy in enumerate(grid):
        for j, cd in enumerate(grid):
            m.sensitivity_analysis(cf_y=max(cy, 1e-6), cf_d=max(cd, 1e-6), rho=1.0)
            Z[i, j] = float(np.ravel(m.sensitivity_params["theta"]["lower"])[0])
    pd.DataFrame(Z, index=grid, columns=grid).to_csv(TAB / f"e7_grid_lower_{name.split()[0]}.csv")
    cs = ax.contourf(grid, grid, Z, levels=12, cmap="Blues_r")
    c0 = ax.contour(grid, grid, Z, levels=[0], colors=[C["red"]], linewidths=2.5)
    ax.plot([], [], color=C["red"], lw=2.5, label="Cận dưới của θ = 0")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88))
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(grid[0], grid[-1])
    if not bench.empty:
        bb = bench[bench.model == name]
        outside = bb[(bb.cf_y > grid[-1]) | (bb.cf_d > grid[-1])]
        if len(outside):
            ax.text(0.98, 0.97, "Ngoài khung: " + ", ".join(f"{r.benchmark} (cf_y={r.cf_y:.2f}, cf_d={r.cf_d:.2f})"
                                                          for _, r in outside.iterrows()),
                    transform=ax.transAxes, ha="right", va="top", fontsize=10, color=C["ink2"],
                    bbox=dict(fc="white", ec="none", alpha=0.85))
        bb = bb.drop(outside.index)
        ax.scatter(bb.cf_d, bb.cf_y, s=80, color=C["orange"], edgecolor="white", linewidth=2, zorder=4)
        for _, r in bb.iterrows():
            ax.annotate(r.benchmark, (r.cf_d, r.cf_y), xytext=(6, 4), textcoords="offset points", fontsize=11)
    ax.set_title(f"DML-{name}")
    ax.set_xlabel("cf_d: phần biến thiên của D do confounder ẩn")
    ax.grid(False)
    fig.colorbar(cs, ax=ax, format=FuncFormatter(money), label="Cận dưới của θ")
axes[0].set_ylabel("cf_y: phần biến thiên của Y do confounder ẩn")
fig.suptitle("Confounder ẩn phải mạnh đến đâu thì tác động biến mất?", x=0.01, ha="left", fontsize=15,
             fontweight="bold")
fig.text(0.01, -0.03, "Điểm cam: confounder ẩn mạnh ngang biến quan sát được (benchmark = bỏ biến đó ra rồi đo). rho = 1 (kịch bản xấu nhất).",
         fontsize=10, color=C["ink2"])
save_fig(fig, "e7_sensitivity_contour")

# ---------- E8: GATE by income quintile ----------
inc_q = pd.qcut(df["inc"], 5, labels=[f"Q{i}" for i in range(1, 6)])
groups = pd.DataFrame({"inc_quintile": inc_q.astype(str)})
gate = irm.gate(groups=groups)
gci = gate.confint(level=0.95).reset_index().rename(columns={"index": "group"})
bounds = df.groupby(inc_q, observed=True)["inc"].agg(["min", "max", "count"]).reset_index(drop=True)
gci = pd.concat([gci, bounds], axis=1)
save_table(gci, "e8_gate_income")
print(gci.round(1).to_string(index=False))

plots.e8_gate(gci, float(irm.coef[0]))

with open(OUT / "models_main.pkl", "wb") as f:
    pickle.dump({k: {"coef": float(v.coef[0]), "se": float(v.se[0])} for k, v in models.items()}, f)
