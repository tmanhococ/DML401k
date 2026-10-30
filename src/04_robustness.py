"""E5: how stable are the DML estimates?
  (a) repeated cross-fitting with different random splits (n_rep)
  (b) different nuisance learners
  (c) different numbers of folds K
"""
import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.utils import PSProcessorConfig
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import plots
from common import (D, FAST, N_FOLDS, SEED, TRIM, X, Y, Timer, dml_row, load_data, rf_clf, rf_reg, save_table,
                    setup_style)

setup_style()
df = load_data()
data = dml.DoubleMLData(df, y_col=Y, d_cols=D, x_cols=X)
ps_cfg = PSProcessorConfig(clipping_threshold=TRIM)
N_REP = 5 if FAST else 20


def plr(ml_l, ml_m, k=N_FOLDS, n_rep=1):
    return dml.DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=k, n_rep=n_rep)


def irm(ml_g, ml_m, k=N_FOLDS, n_rep=1):
    return dml.DoubleMLIRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=k, n_rep=n_rep, score="ATE", ps_processor_config=ps_cfg)


def fit(make, label):
    np.random.seed(SEED)
    with Timer(label) as t:
        m = make()
        m.fit()
    return m, t.elapsed


# ---------- (a) repeated cross-fitting ----------
rep_rows, rep_summary = [], []
for name, make in [("PLR", lambda: plr(rf_reg(8), rf_clf(4), n_rep=N_REP)),
                   ("IRM (ATE)", lambda: irm(rf_reg(8), rf_clf(4), n_rep=N_REP))]:
    m, sec = fit(make, f"E5a {name} n_rep={N_REP}")
    coefs, ses = m.all_coef[0], m.all_se[0]
    for r, (c, s) in enumerate(zip(coefs, ses)):
        rep_rows.append({"model": name, "rep": r + 1, "coef": c, "se": s})
    rep_summary.append(dml_row(f"DML-{name}", "e401", m, {"n_rep": N_REP, "min_coef": coefs.min(),
                                                           "max_coef": coefs.max(), "sd_coef": coefs.std(ddof=1),
                                                           "fit_seconds": sec}))
reps = pd.DataFrame(rep_rows)
save_table(reps, "e5a_repeated_crossfit_draws")
save_table(pd.DataFrame(rep_summary), "e5a_repeated_crossfit_summary")

plots.e5a_repeated(reps)

# ---------- (b) learners ----------
poly = lambda: PolynomialFeatures(degree=2, include_bias=False)
learners = {
    "Tuyến tính (OLS / Logit)": (
        lambda: make_pipeline(StandardScaler(), LinearRegression()),
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    ),
    "Lasso + đa thức bậc 2": (
        lambda: make_pipeline(poly(), StandardScaler(), LassoCV(cv=5, alphas=50, max_iter=20000, random_state=SEED)),
        lambda: make_pipeline(poly(), StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, max_iter=5000)),
    ),
    "Random Forest (cấu hình guide)": (lambda: rf_reg(8), lambda: rf_clf(4)),
    "Gradient Boosting": (
        lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=SEED),
        lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=SEED),
    ),
}
lrows = []
for lname, (mk_reg, mk_clf) in learners.items():
    for name, make in [("PLR", lambda: plr(mk_reg(), mk_clf())), ("IRM (ATE)", lambda: irm(mk_reg(), mk_clf()))]:
        try:
            m, sec = fit(make, f"E5b {name} / {lname}")
        except Exception as e:
            print(f"  failed: {e!r}")
            continue
        ev = m.evaluate_learners()
        extra = {"learner": lname, "fit_seconds": sec}
        extra.update({f"rmse_{k}": float(np.ravel(v)[0]) for k, v in ev.items()})
        lrows.append(dml_row(f"DML-{name}", "e401", m, extra))
lt = pd.DataFrame(lrows)
save_table(lt, "e5b_learners")
print(lt[["model", "learner", "coef", "se"]].round(1).to_string(index=False))

plots.e5b_learners(lt)

# ---------- (c) number of folds ----------
frows = []
for k in [2, 5, 10, 20]:
    for name, make in [("PLR", lambda: plr(rf_reg(8), rf_clf(4), k=k)), ("IRM (ATE)", lambda: irm(rf_reg(8), rf_clf(4), k=k))]:
        m, sec = fit(make, f"E5c {name} K={k}")
        frows.append(dml_row(f"DML-{name}", "e401", m, {"n_folds": k, "fit_seconds": sec}))
ft = pd.DataFrame(frows)
save_table(ft, "e5c_folds")

plots.e5c_folds(ft)
