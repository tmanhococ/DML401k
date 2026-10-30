# DML 401(k): thực nghiệm

Tái lập và mở rộng ví dụ [401(k) của dmlguide](https://dmlguide.github.io/examples/401k) (dữ liệu SIPP 1991, n = 9.915) bằng Python và DoubleML.

**Câu hỏi:** việc đủ điều kiện tham gia 401(k) (`e401`) tác động thế nào đến tài sản tài chính ròng (`net_tfa`)?

Ký hiệu: Y = `net_tfa`, D = `e401`, X = 9 biến kiểm soát (`age, tw, inc, fsize, db, marr, twoearn, pira, hown`).
Nuisance: l(X) = E[Y|X], m(X) = E[D|X], g(D,X) = E[Y|D,X].

## Chạy

```bash
python -m venv .venv
./.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: ./.venv/bin/python
DML_FAST=1 ./run_all.sh   # chạy thử: 100 cây, khoảng 5 phút   -> outputs_fast/
./run_all.sh              # bản đầy đủ: 1000 cây, 20 lượt lặp   -> outputs/
```

Log của từng bước nằm trong `outputs/logs/`. File tổng hợp để review là `outputs/results_summary.md`.

## Các thí nghiệm

| Script | Thí nghiệm | Ý lý thuyết minh họa | Artifacts chính |
|---|---|---|---|
| `01_eda_baseline.py` | E0: so sánh cân bằng X giữa hai nhóm; E1: các baseline (chênh lệch thô, OLS, OLS linh hoạt, 2SLS) | Confounding, selection bias | `e0_love_plot.png`, `e1_baselines.csv` |
| `02_dml_by_hand.py` | E2: DML-PLR tự viết bằng tay, so với DoubleML trên cùng folds; bỏ cross-fitting; bỏ trực giao | FWL / Robinson, cross-fitting, Neyman orthogonality | `e2_residual_on_residual.png`, `e2_variants.png` |
| `03_dml_main.py` | E3: PLR, IRM (ATE/ATTE); E6: PLIV, IIVM (LATE); E4: overlap và chất lượng nuisance; E7: sensitivity analysis; E8: GATE theo thu nhập | Estimands, overlap, giả định unconfoundedness, tác động không đồng nhất | `e3_dml_main.csv`, `e4_overlap.png`, `e7_sensitivity_contour.png`, `e8_gate_income.png` |
| `04_robustness.py` | E5: lặp cross-fitting (n_rep), đổi learner, đổi số folds K | Phụ thuộc vào cách chia mẫu và vào learner | `e5a_*.png`, `e5b_learners.png`, `e5c_folds.png` |
| `05_report.py` | Forest plot tổng hợp và `results_summary.md` | — | `e3_forest_plot.png` |

Cấu hình giữ đúng như guide: Random Forest 1000 cây, max_depth 8 cho nuisance của outcome và 4 cho các nuisance còn lại, 10 folds, clip propensity ở 0,001, seed 20241111.

## Lưu ý kỹ thuật

- Trong DoubleML, `DoubleMLPLIV` gọi `.predict()` cho `ml_m`/`ml_r`, nên hai learner này phải là regressor (dùng classifier thì thư viện lấy nhãn 0/1 thay vì xác suất, và kết quả bị lệch).
- Cách chia folds là ngẫu nhiên và phiên bản thư viện có khác biệt, nên con số sẽ không khớp tuyệt đối với guide. Xem E5a.
