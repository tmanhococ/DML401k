"""Figures drawn from the saved result tables, so they can be redrawn without refitting (see replot.py)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from common import C, SEED, money, save_fig

GUIDE = {"PLR": 6737.64, "IRM (ATE)": 6344.02}
LEARNER_ORDER = ["Tuyến tính (OLS / Logit)", "Lasso + đa thức bậc 2", "Random Forest (cấu hình guide)", "Gradient Boosting"]


def e2_variants(out):
    o = out.iloc[::-1].reset_index(drop=True)
    th = out.loc[out.variant == "manual", "coef"].iloc[0]
    cols = {"manual": C["blue"], "library": C["blue"], "no_crossfit": C["red"], "no_orthogonal": C["red"]}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, r in o.iterrows():
        ax.hlines(i, r.ci_lo, r.ci_hi, color=cols[r.variant], lw=2)
        ax.scatter(r.coef, i, s=80, color=cols[r.variant], edgecolor="white", linewidth=2, zorder=3)
        ax.text(r.ci_hi + 120, i, f"{r.coef:,.0f}  (SE {r.se:,.0f})", va="center", fontsize=12, color=C["ink"])
    ax.set_yticks(range(len(o)), o.model)
    ax.axvline(th, color=C["blue"], lw=1, alpha=0.4)
    ax.set_xlim(o.ci_lo.min() - 300, o.ci_hi.max() + 2200)
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlabel("Ước lượng tác động của e401 lên net_tfa (CI 95%)")
    ax.set_title("Bỏ một thành phần của DML thì điều gì xảy ra?")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "e2_variants")


def e5a_repeated(reps):
    n_rep = reps.rep.nunique()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    rng = np.random.default_rng(SEED)
    for i, name in enumerate(["PLR", "IRM (ATE)"]):
        r = reps[reps.model == name]
        yy = i + rng.uniform(-0.12, 0.12, len(r))
        ax.scatter(r.coef, yy, s=50, color=C["blue"], alpha=0.6, edgecolor="white", linewidth=1.5,
                   label="Một lần chia folds ngẫu nhiên" if i == 0 else None)
        ax.vlines(np.median(r.coef), i - 0.28, i + 0.28, color=C["ink"], lw=2.5,
                  label="Trung vị (DoubleML gộp)" if i == 0 else None)
        ax.scatter(GUIDE[name], i, marker="D", s=70, facecolor="white", edgecolor=C["orange"], linewidth=2, zorder=4,
                   label="Số công bố trên dmlguide (Python)" if i == 0 else None)
        ax.text(r.coef.min(), i + 0.34, f"từ {r.coef.min():,.0f} đến {r.coef.max():,.0f}", va="center", fontsize=11,
                color=C["ink2"])
    ax.set_yticks([0, 1], ["DML-PLR", "DML-IRM (ATE)"])
    ax.set_ylim(-0.6, 1.6)
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlabel("Ước lượng θ")
    ax.set_title(f"Cùng dữ liệu, cùng mô hình, {n_rep} cách chia folds khác nhau")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=11)
    ax.grid(axis="y", visible=False)
    save_fig(fig, "e5a_repeated_crossfit")


def e5b_learners(lt):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    order = [l for l in LEARNER_ORDER if l in set(lt.learner)][::-1]
    for name, col, off in [("DML-PLR", C["blue"], 0.15), ("DML-IRM (ATE)", C["orange"], -0.15)]:
        sub = lt[lt.model == name].set_index("learner")
        for i, l in enumerate(order):
            if l not in sub.index:
                continue
            r = sub.loc[l]
            ax.hlines(i + off, r.ci_lo, r.ci_hi, color=col, lw=2)
            ax.scatter(r.coef, i + off, s=70, color=col, edgecolor="white", linewidth=2, zorder=3,
                       label=name if i == 0 else None)
            ax.text(r.ci_hi + 150, i + off, f"{r.coef:,.0f}", va="center", fontsize=10, color=C["ink2"])
    ax.set_yticks(range(len(order)), order)
    ax.axvline(0, color=C["ink2"], lw=1)
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlabel("Ước lượng tác động của e401 (CI 95%)")
    ax.set_title("Đổi learner cho nuisance: kết quả thay đổi bao nhiêu?")
    ax.legend(loc="lower left")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "e5b_learners")


def e5c_folds(ft):
    ks = sorted(ft.n_folds.unique())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, col, off in [("DML-PLR", C["blue"], -0.08), ("DML-IRM (ATE)", C["orange"], 0.08)]:
        sub = ft[ft.model == name].sort_values("n_folds").reset_index(drop=True)
        xs = np.arange(len(sub)) + off
        ax.vlines(xs, sub.ci_lo, sub.ci_hi, color=col, lw=2)
        ax.plot(xs, sub.coef, color=col, lw=2, marker="o", ms=8, mec="white", mew=2, label=name)
    ax.set_xticks(range(len(ks)), [f"K = {k}" for k in ks])
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_ylabel("θ ước lượng (CI 95%)")
    ax.set_title("Số folds trong cross-fitting")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save_fig(fig, "e5c_folds")


def e8_gate(gci, ate):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    xs = np.arange(len(gci))
    ax.axhline(ate, color=C["gray"], lw=1.5, label=f"ATE chung = {ate:,.0f}")
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.vlines(xs, gci["2.5 %"], gci["97.5 %"], color=C["blue"], lw=2)
    ax.scatter(xs, gci["effect"], s=80, color=C["blue"], edgecolor="white", linewidth=2, zorder=3, label="GATE (CI 95%)")
    for i, r in gci.iterrows():
        ax.text(i + 0.08, r["effect"], f"{r['effect']:,.0f}", va="center", fontsize=11)
    labels = [f"{g.replace('Group_', '')}\n{money(a)} – {money(b)}" for g, a, b in zip(gci.group, gci["min"], gci["max"])]
    ax.set_xticks(xs, labels)
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlabel("Nhóm ngũ phân vị thu nhập (thu nhập năm)")
    ax.set_ylabel("Tác động lên net_tfa")
    ax.set_title("Tác động không đồng nhất: GATE theo thu nhập (DML-IRM)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save_fig(fig, "e8_gate_income")
