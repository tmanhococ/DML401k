"""Redraw the table-based figures from saved CSVs without refitting any model."""
import pandas as pd

import plots
from common import TAB, setup_style

setup_style()
read = lambda n: pd.read_csv(TAB / f"{n}.csv")
plots.e2_variants(read("e2_dml_by_hand"))
plots.e5a_repeated(read("e5a_repeated_crossfit_draws"))
plots.e5b_learners(read("e5b_learners"))
plots.e5c_folds(read("e5c_folds"))
main = read("e3_dml_main")
plots.e8_gate(read("e8_gate_income"), float(main.loc[main.model == "DML-IRM (ATE)", "coef"].iloc[0]))
