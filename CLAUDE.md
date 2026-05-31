# THESIS PROJECT CONTEXT — EEG Sleep Prediction

## What this project is
**Bachelor's thesis** by Paola. Goal: predict how many hours a person sleeps per night using EEG brain signals recorded with an Emotiv consumer-grade device.
- 83 subjects total; **ba16 and ba45 are held out** for defense demo (not used in any modelling)
- So 81 subjects used for modelling
- EEG features: FFT power bands — 14 channels × 5 frequency bands = **70 POW features**
- Target: Sleep_Hours (regression) and sleep category (classification)

## Sleep categories
- **3-class fixed**: Short ≤6h | Normal >6h and <8h | Long ≥8h  (27 Short, 46 Normal, 8 Long)
- **3-class tertile**: ≤6.5h | >6.5–7h | >7h  (35 / 27 / 19) — used to fix Long class imbalance
- **Binary**: Below7h <7h | Above7h ≥7h  (35 Below, 46 Above)
- The `make_cat()` function in all scripts handles these correctly
- Tertile thresholds are data-driven (33rd/67th percentiles), not clinically grounded — both schemes reported in thesis

## Key data detail: Emotiv FFT sampling
The Emotiv outputs FFT power values at ~8Hz, NOT 128Hz. So in the raw data, only ~1 in 16 rows has actual POW values — the rest are NaN. pandas `.mean()` handles this correctly (skips NaN). Never use `.dropna()` before windowing — it leaves too few rows.

---

## Folder structure
```
THESIS/
├── all_data/              raw EEG per subject (subfolders ba01, ba02, ...)
├── modelling_tables/      CSV tables used as input to modelling scripts
├── models/                trained final models (for defense demo)
├── results/               CV output (metrics, plots, predictions CSVs)
│   ├── permutation_test/  output of permutation_test.py (null distributions, p-values)
│   └── ...                per-experiment subdirs
├── visuals_fft/           EDA plots (incl. fft_sleep_correlation_bars.png)
├── src/
│   ├── 3_modelling/
│   │   ├── run_cv.py              subject-level CV (exp1, exp2, exp3)
│   │   ├── run_task_cv.py         task/window-level CV (exp4, exp5, exp6)
│   │   ├── permutation_test.py    1000-permutation test for exp3 + exp6 (GroupKFold)
│   │   ├── train_final_model.py   train final model on all data
│   │   └── predict.py             predict for one new subject (defense demo)
│   └── 3_modelling_scripts/
│       ├── modelling_tables.py            creates exp1, exp2, exp3 CSVs
│       └── create_task_level_tables.py    creates exp4, exp5, exp6 CSVs
├── THESIS_WRITINGS/
│   ├── Thesis_Draft.docx          full thesis (12,071 words, all chapters complete)
│   └── modelling_analysis.docx    detailed modelling analysis doc (sections 1–6)
└── CLAUDE.md              (this file)
```

---

## Experiments summary

### Subject-level tables (run with `run_cv.py`)
| Exp | CSV | Description | Rows |
|-----|-----|-------------|------|
| exp1 | exp1_fft_only.csv | All 36 tasks averaged per subject, 70 features | 81 |
| exp2 | exp2_fft_gender.csv | Same + Gender feature | 81 |
| exp3 | exp3_task_specific.csv | Top-5 tasks averaged, top-15 features | 81 |

Top-5 tasks: `gyrus_left_closed_eyes`, `gyrus_right_closed_eyes`, `gyrus_coord`, `prefrontal1`, `prefrontal2`

### Task/window-level tables (run with `run_task_cv.py`)
| Exp | CSV | Description | Rows |
|-----|-----|-------------|------|
| exp4 | exp4_fft_aggregated.csv | 1 row per subject × task (mean within task) | ~2,758 |
| exp5 | exp5_fft_windowed.csv | 1 row per subject × task × window (256-row windows ≈ 2s) | ~32,360 |
| exp6 | exp6_top5_task_level.csv | Like exp4 but top-5 tasks only | ~405 |

---

## Results (complete)

### GroupKFold results (honest — subjects never split across folds)
| Exp | Script | Split | R² | Accuracy | F1 macro | Notes |
|-----|--------|-------|----|----------|----------|-------|
| exp1 | run_cv.py | KFold | -0.12 | 81% | 0.30 | predicts Normal for everyone |
| exp3 | run_cv.py | KFold | 0.05 | 60% | 0.39 | top-5 tasks, 15 features, 3-class |
| exp6 | run_task_cv.py | GroupKFold | 0.10 | 58% | 0.54 | top-5 tasks, 15 features, binary |

Results folder names: `results/exp1_fft_only/`, `results/exp3_task_specific/`, `results/exp6_top5_task_level/`

### StratifiedKFold results (inflated — same subject can be in train and test)
| Exp | Split | R² | Accuracy | F1 macro |
|-----|-------|----|----------|----------|
| exp4 | StratifiedKFold | 0.65 | 88.7% | 0.86 |
| exp5 | StratifiedKFold | 0.67 | 90.2% | 0.88 |

Results folders: `results/exp4_fft_aggregated_stratified/`, `results/exp5_fft_windowed_stratified/`

### Permutation test results (1000 permutations, GroupKFold, subject-level shuffle for exp6)
| Exp | Observed F1 | p-value (raw) | p-value (Bonferroni ×2) | Significant? |
|-----|-------------|---------------|--------------------------|--------------|
| exp3 | 0.39 | 0.008 | 0.016 | ✓ (α=0.05) |
| exp6 | ~0.38 | 0.018 | 0.036 | ✓ (α=0.05) |

Primary metric: F1 macro. Bonferroni correction for 2 simultaneous tests.
R² for exp6 is computed at subject level (majority vote / mean aggregation) to match saved results.
Output in `results/permutation_test/`.

---

## Important finding: StratifiedKFold leakage
When using StratifiedKFold on exp4/exp5 (multiple rows per subject), the model sees ~30 tasks from a subject in training and is tested on the remaining ~7 tasks from the same subject. The model learns "subject BA42's EEG fingerprint → 8.5h" rather than a generalizable EEG→sleep pattern.

Evidence: within-subject prediction std ≈ 0.178h — all tasks of same subject get nearly identical predictions regardless of task type.

Mentor confirmed both approaches should be reported. Both are in the thesis: GroupKFold as the primary honest result, StratifiedKFold reported for completeness with explicit leakage caveat.

## Important finding: Direction reversal
F8.Gamma and F7.Gamma show a **positive** correlation with sleep hours at the row level but flip to **negative** at the subject level. This means the row-level association is driven by within-subject task variance, not genuine between-subject sleep differences — exactly the variance StratifiedKFold exploits. Reported in Chapter 5 and referenced in Chapter 7 discussion.

---

## run_task_cv.py settings
Key settings at the top of the file:
```python
EXPERIMENT = 'exp5_fft_windowed'   # which CSV to load
SPLIT_MODE = 'stratified'          # 'group' = GroupKFold | 'stratified' = StratifiedKFold
K          = 5                     # folds
TOP_N      = 15                    # None = all 70 features, int = top-N
BINARY     = True                  # True = 2 classes | False = 3 classes
```
Output goes to `results/{EXPERIMENT}_{SPLIT_MODE}/`

## run_cv.py settings (for exp1/exp2/exp3 only)
```python
EXPERIMENT = 'exp3_task_specific'
K          = 5
TOP_N      = 15
```
Output goes to `results/{EXPERIMENT}/`

## permutation_test.py
Tests exp3 (KFold) and exp6 (GroupKFold) with 1000 permutations.
- exp3: standard label shuffle
- exp6: subject-level shuffle (all rows for a subject get same permuted label)
- Aggregates predictions to subject level before computing R²/F1 for exp6
- Outputs null distribution plots + p-value summary to `results/permutation_test/`

---

## Current status
- **Thesis writing: COMPLETE** — all chapters written including Conclusion
- Word count: **12,071 words** (Introduction through Conclusion)
- `THESIS_WRITINGS/Thesis_Draft.docx` — final document
- `THESIS_WRITINGS/modelling_analysis.docx` — detailed analysis (Sections 1–6, including permutation test)
- **Pending for defense**: train final model on all 81 subjects (`train_final_model.py`), then run `predict.py` on ba16/ba45 live
- Mentor will send data collection text to incorporate into Chapter 3
