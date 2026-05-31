# Thesis Review — Critical Audit

Reviewer notes for *Analysis of Brain Signal Patterns and Development of a Predictive Model for Sleep Duration* (Paola Caric). Every number below was checked against the files in `results/` and the references against the original papers. Line references are to the current `Thesis_Draft.docx`.

---

## 1. Verified numerical errors and contradictions (fix before submission)

**1.1 Tertile class distribution is wrong (Ch. 7, Phase 3).**
The draft says the tertile scheme yields "a 35 / 27 / 20 distribution" and refers to "(9 vs. 20 subjects)" and "12/20 subjects = 63.2%".
Actual counts from `results/exp6_task_interpretation_tertile/subject_summary.csv`: **Short = 27, Normal = 35, Long = 19** (total 81).
- "35 / 27 / 20" → should be **27 / 35 / 19** in Short/Normal/Long order (you also swapped Short and Normal).
- Long = **19**, not 20. Your own recall math proves it: 12/19 = 63.2% (12/20 would be 60%), and prefrontal2 = 14/19 = 73.7%.
- "9 vs. 20 subjects" → should be **8 vs. 19** (fixed-threshold Long modelling n = 8).

**1.2 Self-contradictory sentence on class counts (Ch. 3, line ~97).**
"...27 short, 47 normal, 9 long... These counts reflect the 81 participants included in modelling."
27 + 47 + 9 = **83**, not 81. The actual modelling counts (verified in `results/exp1_fft_only/..._classification_metrics.txt`, support column) are **27 Short / 46 Normal / 8 Long = 81**.
Fix: either present the all-subject EDA counts and say "across all 83 participants," or give the modelling counts 27/46/8. The current sentence is internally inconsistent.

**1.3 n = 83 vs n = 81 used interchangeably for the model.**
Lines ~89–90 justify Random Forest "due to the sample size (n = 83)" / "robust to the small sample size (n = 83)." The model is trained on **81** (ba16, ba45 held out); the Conclusion correctly says 81.
Recommended convention, stated once and held throughout: **EDA on 83, modelling on 81.** Never call the modelling sample 83. The EDA sections (Figs 5.2, 5.6, Mann–Whitney n = 47/9, n = 63/20) are already internally consistent at 83 — good. Only Ch. 2/Ch. 3 need the fix.

*Note:* exp1's reported figures in the draft (49% accuracy, F1 = 0.24) are **correct** — they match the result file. (A stale note in `CLAUDE.md` lists 81% / 0.30; ignore it, the thesis is right.)

---

## 2. Reference problems (context does not match the source)

**2.1 [8] Othmani et al. — wrong year + overuse.**
Bibliography lists "Neurocomputing, 2025." The paper is **Neurocomputing, vol. 557, art. 126709, 2023** (title: *EEG-based neural networks approaches for fatigue and drowsiness detection: A survey*). Fix the year and the title.
More important: this single fatigue/drowsiness survey is cited as the source for at least six different claims — the 10-20 electrode system (line ~78), frequency-band definitions (line ~84), memory tasks recruiting temporal/parietal regions (~109), visuospatial parietal/occipital (~110), and auditory temporal regions (~112). A drowsiness survey is not the right primary source for basic functional neuroanatomy or for the 10-20 system. Examiners notice when one citation is load-bearing for unrelated facts. Replace with appropriate sources (a cognitive-neuroscience or EEG textbook; Jasper 1958 for 10-20).

**2.2 [3] Roy et al. — misapplied in the Introduction.**
Line ~62: "Changes in electrical brain activity patterns correlate with changes in alertness, fatigue, mental workload and mood [3]." Roy et al. (2019) is a **deep-learning methods** systematic review (domains: epilepsy, sleep, BCI, cognitive/affective monitoring). It is not evidence for that physiological statement. Move the physiology claim to [8] or a primary source. Keep [3] only at line ~90, where it correctly supports the DL-methods sentence — that usage is fine.

**2.3 [7] Sangeetha et al. — partial mismatch.**
Real paper (*Brain & Heart*, 2024), but it is about BCI thought-classification (BCI-IV dataset, KNN ~98%) and a driving-behavior case study — **not sleep**. Line ~75 frames it as supporting "machine learning systems designed to assess cognitive states related to sleep," which the paper does not do. The general feasibility claim is fine; delete the "related to sleep" framing.

**2.4 References that are correct and well-used (keep):**
Breiman [1], Bishop [2], Craik [4], Mueller & Guido [5], Goodfellow [6], Hastie [9], Kohavi [10], Sokolova & Lapalme [11], Klimesch [12], Goncharova [13]. Klimesch for alpha/theta–memory and Goncharova for EMG contamination are excellent, precise choices.

---

## 3. Scientific claims to qualify

**3.1 "Well-rested individuals show higher beta and gamma" (line ~74) is an overclaim.**
The robust, replicated effect is **increased frontal theta** (and theta/alpha) with sleep pressure. The symmetric claim that rest raises beta/gamma is not well supported — gamma effects are inconsistent, and acute deprivation can itself raise beta. Soften to: "whereas high-frequency power shows less consistent changes."

**3.2 The Emotiv device cannot measure Delta — and you never say so.**
The POW output has 5 bands (Theta, Alpha, BetaL, BetaH, Gamma); there is no Delta. Delta / slow-wave is the single most sleep-relevant frequency range. This is a genuine hardware limitation that strengthens your "modest signal" framing — add one sentence in Ch. 4 or the limitations.

**3.3 "Validated for research applications" (line ~78) needs a citation or softening.**
Emotiv EPOC research validity is genuinely contested. Either cite a validation study or soften the wording.

---

## 4. The biggest methodological weakness (likely defense question)

**Selection circularity in the exp1 → exp3/exp6 comparison.**
The "top-5 tasks" were chosen because they correlated most strongly with sleep across these same 81 subjects (Ch. 5). exp3/exp6 then restrict to those tasks and report improved cross-validated performance, and the thesis concludes "task selection adds genuine predictive value." That comparison is partly **circular**: the tasks were selected to maximise the very signal then being measured, on the same subjects, so exp3/exp6's edge over exp1 is optimistically biased.

The permutation test does **not** rescue this. I checked `permutation_test.py`: it loads the already-restricted 5-task CSV and feature selection (`select_features`) runs **once on the real labels, outside** the permutation loop (line ~205); only labels are shuffled inside. So the null never re-does task or feature selection. The p-value answers "given these 5 tasks and 15 features, is the label–feature mapping above chance?" — it does **not** validate the prior claim that these 5 tasks are specially sleep-predictive, nor correct for the optimism of having picked them in-sample.

You already flag the feature-selection bias as "minor optimistic bias" (line ~255) — good. The **task-selection** circularity is larger and currently unflagged.

Recommended, in order of effort:
- *Minimum:* add one sentence stating that exp3/exp6's improvement over exp1 is an **upper bound**, because tasks/features were selected in-sample; the true generalisation gain is smaller.
- *Better:* nested cross-validation — select tasks and features inside each training fold only.
- *Strongest:* hold out a subset of subjects purely for selection.

Pre-empting this in the Discussion will look far better than being asked about it cold.

---

## 5. A strong finding that is missing from the draft

Your `CLAUDE.md` notes a **sign reversal**: F7/F8 Gamma correlate *positively* with sleep at the row level but *negatively* at the subject level — a clean Simpson's-paradox illustration of why row-level / StratifiedKFold analysis leaks within-subject variance. The notes say it's "reported in Chapter 5 and Chapter 7," but I cannot find it in the current draft. It is a genuinely good point that directly reinforces your leakage argument with a concrete example. Consider adding it.

---

## 6. What is done well (credit where due)

- The **GroupKFold vs StratifiedKFold leakage analysis** is sophisticated for a bachelor's thesis. Quantifying the within-subject prediction std (≈ 0.18 h) as direct mechanistic evidence of fingerprinting is genuinely good work.
- **Permutation testing** with subject-level label shuffling and Bonferroni correction shows real statistical maturity.
- The multiple-comparisons accounting is correct: 0.05/70 = 0.0007 (line ~167); 36×70 = 2,520 correlations, ~126 false positives at α = 0.05, Bonferroni 0.05/2,520 ≈ 2e-5 (line ~202). All check out.
- The RF description is technically accurate (sqrt(p) vs p/3 features, ~1/3 bootstrap exclusion, balanced class-weight formula).
- **Honest framing** of weak results ("statistically significant but practically limited") is exactly right and rarer than it should be.
- Reporting both threshold schemes and both CV strategies side by side is transparent and mature.

---

## 7. Priority fix list

1. Tertile counts 27/35/19 and Long = 19 everywhere (§1.1) — factual error.
2. Ch. 3 class-count sentence (§1.2) — contradiction.
3. Othmani year 2023 + reduce its overuse (§2.1).
4. Roy [3] and Sangeetha [7] context fixes (§2.2, §2.3).
5. Add the task-selection-optimism caveat (§4) — most important for the defense.
6. n = 83/81 consistency (§1.3); Delta-band limitation (§3.2).
