# Registration chronology

This file lists every registration behind the results in the paper, grouped by experiment and dated, what was known before each, and how the records can be checked. The registration documents themselves are in `doc/`, in their original language (Korean), unedited. All dates and times are local (KST, UTC+9).

## How the records were kept

- **Rule.** Each experiment was written down (purpose, conditions, decision rules, expected outcomes) before it was run. Results were appended to the end of the same document after the run, with a date; text above was not changed. Settings were chosen on the validation split, and the test split was scored once after the choice was fixed.
- **Original submission (2026-10-08 to 2026-10-14).** There was no version control. The dates are those written in the documents and the file dates. We cannot prove from these files which edits preceded which run, and we say so.
- **Revision (from 2026-10-28).** Each registration document was fixed before any run and its SHA-256 fingerprint was recorded with the time in [`doc/R1_등록_지문.md`](doc/R1_등록_지문.md). Amendments made before running were appended with a new fingerprint. These times are self-recorded; no third-party time stamp was obtained. Because results were only appended, the registered version of each document is the first *k* lines of the file in this repository, and its fingerprint can be recomputed (see below).
- **Not included.** Internal Experiment 0 (inspection and labelling of the private D1 data) and Experiments 2, 5 and 7 (positioning and generalisation, a separate manuscript) are not part of this package. Some documents refer to internal planning files (`doc/PROJECT_BRIEF.md`, `doc/문헌대조_결과.md`) and to revision notes under `paper/` that are not included either.

## Original submission

| Date | Registration (document) | Notes on timing and what was seen before | Used in the paper |
|---|---|---|---|
| 2026-10-09 | Experiment 3, sensor-axis composition (`합성방식_실험계획_최종.md`) | Final plan dated 2026-10-09, the same day as the runs; the selection rule (§6) and hypotheses (§9) are stated as fixed before results, and the checklist (§11) was ticked during the runs. Results: `results/exp3/report.md` (decision 2026-10-10) | Row composition and standardisation used throughout (Sections 3.1 and 3.7) |
| 2026-10-10 | Decisions after Experiment 3 (`결정기록_실험3이후.md`) | Decision 1 (default configuration F01). Decision 2 registered the simple-baseline ladder before its validation run; its test comparison was run once on 2026-10-10 after Experiment 4 had fixed the final model. Decision 3 (D2 real-NLOS protocol) is not used in this paper | Table 3 (simple ladder, node bank) |
| 2026-10-10 | Experiment 4, component ablation and final model (`실험4_계획.md`) | The validation values of the two limiting bases (C05 = Z1, C06 = Z2) on D1 and D2 had been seen before the selection rule was written; the rule was framed so that it does not depend on them. Results appended 2026-10-10 | Section 4.1 |
| 2026-10-10 | Experiment 6 part 1: training amount, window length, missingness (`실험6_계획.md`) | Results 2026-10-10. Same day, after seeing them: a descriptive supplement on the reference error (no selection or decision), and two additions registered before their runs — missing patterns and simple imputation baselines | Sections 3.6, 4.3; Table 5; Figure 3 |
| 2026-10-10 | Experiment 6 addition 3: real missingness on D3 | Designed after a model-free feasibility check (47 % empty cells on a 1 m grid). Run 2026-10-11. The prediction was not supported and detection was near chance (AUC-PR 0.15–0.28 at a positive rate of about 0.05), so the data were judged unsuitable for the question; **not reported in the paper**. Results in the document and `results/exp6/` | — |
| 2026-10-11 | Experiment 6 addition 4: real packet-loss masks from D5 | The descriptive statistics of the loss library had been seen, no model results. Results 2026-10-11 | Table 5, Figure 3 |
| 2026-10-11 | Experiment 6 part 2: threshold percentile, noise, anchors, final T, training contamination | For the final choice of T, the part-1 AUC-PR for each T (validation and test) had already been seen, as the plan states. Results 2026-10-11 | Section 4.4, Table 8 |
| 2026-10-12 | Experiment 6 addition 5: comparators trained under the same training-cell loss | Configurations of Table 4 kept, no re-search under loss; registered before the run. Validation 2026-10-13 (D1, then D2 as a check), test once 2026-10-13 | Table 6, Figure 4 |
| 2026-10-13 | Experiment 6 addition 5b: raw (zero-filled) input for the same comparators | Registered after seeing the addition-5 validation results on D1; exploratory. Validation and test once 2026-10-13 | Section 4.3 (exploratory sentence) |
| 2026-10-11 | Experiment 1, ST-SSD comparison (`실험1_ST-SSD_계획.md`) | Results 2026-10-11. An additional comparison (variants R and D) was registered on 2026-10-12 before it was run. Before running, the same day: a definition defect of R found when re-reading the original paper was corrected; the starting state of D was changed after a smoke test in which one validation value had been seen (D1 hallway, λ̃ = 0.3, γ = 2, AUC-PR 0.39); the γ grid of D was corrected ({0, 1, 2} → {1, 2, 4}, because γ = 0 makes D identical to the static background); and the missing cell-score definition was added. A variant added after seeing validation results (D′) is marked post hoc. Original-variant test 2026-10-12 | Section 4.5, Table 9 |
| 2026-10-11 | Experiment 1, comparators batch 1 (`실험1_비교모델_계획.md`) | Results 2026-10-11 | Table 4 |
| 2026-10-11 | Comparators batch 2 | A correction was registered on 2026-10-12 after re-reading the original papers and before the batch-2 test; some validation values of the affected models had been seen (GDN and LapRPCA on D1; validation and test of the first USAD); one further variant (USADFIX2) was added after a validation result, with the rule for which variant to report fixed before running. Test 2026-10-12 | Table 4 |
| 2026-10-12 | Comparators batch 3 (TranAD, Anomaly Transformer) | Selection recorded before the test; test 2026-10-12 | Table 4 |
| 2026-10-14 | Comparators batch 3 correction (Anomaly Transformer minimax direction) | Registered after comparing the re-implementation with the original paper and before the corrected run; the values of the reversed version had been seen, those of the corrected version had not. Validation and test 2026-10-14 | Table 4 |
| 2026-10-11 | Experiment 8, computing cost (`실험8_계획.md`) | Results 2026-10-12 | Section 5 |
| 2026-10-11 | Experiment 9, SMD/ASD scope (`실험9_계획.md`) | Results 2026-10-12 | Section 4.7 |
| 2026-10-12 | Experiment 10, independent data D6 (`실험10_계획.md`) | Two pre-run amendments: the row-wise SAP path, and a ×3 injection scale added after one value had been printed by a profiling run (scan 1, PS, SC1, seed 142, cell AUC-PR 0.078); the registered main scale stayed the decision basis. Validation and test 2026-10-12 | Section 4.6, Table 10 |
| 2026-10-12 | Experiment 10-2, robust standardisation and outlier-free evaluation | Registered after seeing the Experiment 10 results; post hoc and exploratory | Section 4.6 (exploratory 1) |
| 2026-10-12 | Experiment 10-3, design candidates PS1, PSH, PS1H | Registered after seeing 10-2; post hoc and exploratory | Table 11 |

## Revision

Registration time for R1-3 to R1-6: 2026-10-28 21:32:09. Before registering R1-3 and R1-4 we had seen all results of the original submission. R1-5 uses synthetic data only. Before R1-6 and R1-7 only the structural values in `data_ext/D7_uji/STRUCTURE.md` and `data_ext/D8_hymn/STRUCTURE.md` had been seen; no model had been fitted to those data.

| Registration | Registered version = first *k* lines of | *k* | SHA-256 (recorded) | Recorded at | Runs |
|---|---|---|---|---|---|
| R1-3 inference, thresholds, training seeds, prediction variance | `doc/분석R1-3_계획.md` | 35 | `a0fe9bc321202074ce9ea6749430d155e127cc1f853f2bc200d3e59af06d64be` | 2026-10-28 21:32:09 | 2026-10-28 |
| R1-4 reference × detector factorial; robust reference baselines | `doc/실험R1-4_계획.md` | 78 | `34d9cb39ad1bebcae9c2bfde270bee7321fe3d8a71516391bc4e05103284791a` | 2026-10-28 21:32:09 | part 2: 2026-10-28; part 1: validation 2026-10-28, test 2026-10-29 |
| R1-5 simulation study | `doc/실험R1-5_계획.md` | 81 | `149922db185aeb96d22c829b68f8c6a3d8895561310106d058ec515886e78921` | 2026-10-28 21:32:09 | 2026-10-28 |
| R1-5 pre-run amendment 1 (implementation details) | same | 92 | `eabd903c103a193271256cc99eb35fcb62c43caa819117163f656605978f19ad` | 2026-10-28 21:35:45 | |
| R1-5 amendment 2 (three design cells for two registered predictions that had none; added after the main run had started, before any result was seen) | same | 97 | `7146e08fcc62f2f0f3125fdbc2c056cf166f799108ecb1d6273546426de8f73a` | 2026-10-28 21:38:06 | |
| R1-6 Wi-Fi (UJI) | `doc/실험R1-6_계획.md` | 59 | `6d02480dccbbe87ad4c1bbadbe82866f9149dca19009abdf05064a892cba291b` | 2026-10-28 21:32:09 | 2026-10-28 |
| R1-6 pre-run amendment | same | 70 | `6b01d8dbb2ef5b7701c7de2f5cb5fa4fb70df8dded5b4acf578cc4c2eb051ba4` | 2026-10-28 21:49:13 | |
| R1-6 values seen before registration | `data_ext/D7_uji/STRUCTURE.md` | 28 (whole file) | `d9af5016410208c27a98b704dc25eaccc1ddf390cc48c1a238cc488f9959dfba` | 2026-10-28 21:32:09 | |
| R1-7 HYMN confirmation | `doc/실험R1-7_계획.md` | 77 | `39da7c9f4277722401a1c52525400baed78c42a897d453699152c1aef9e667d5` | 2026-10-29 01:38:11 | validation 02:10–02:11, test 02:11–02:12 |

For R1-7 the data loader, the analysis code and the structural note were fixed together with the plan; their fingerprints, recorded at the same time, are those of the files in this release:

| File | SHA-256 |
|---|---|
| `data_ext/D8_hymn/STRUCTURE.md` | `2a7f0a79c2b967fd52ac3264ebda99e099ef81abba9f1d7fab5714d8d4bfd0a4` |
| `src/data/d8.py` | `e9aa6cb6dc1ddd498ec661c3e7eef015c1393eaf8726849c25eaf07c1841c714` |
| `experiments/expR1_7_hymn.py` | `78183d462726d85787d983f9f0de97891b570b1e9fd8e099c8ba4882709889fc` |
| `src/model/compose.py` | `e6290f26b6ebda69b7722adb69a053d0a035f99bc0d1c7813d81554996476d08` |

Read-only copies of the five R1-7 files as registered are in `doc/registered/R1-7/`. Two of the seven R1-7 predictions were not borne out; the results are reported as found (Section 4.6 of the paper).

Post hoc note on R1-6 (2026-10-29, appended to the plan after line 103; the registered verdicts are unchanged): reading the dataset description after the runs showed that one of the four registered test sets (Test-05) was recorded at the same points but facing the two other directions. `experiments/expR1_6b_uji_by_set.py` splits the registered test by set with the same injections (it reproduces the pooled values to 1e-16), and `analysis/85_uji_a5_direction.py` checks which orientation the month-25 sets of the second phone correspond to. Both are post hoc and are marked so in the paper (Section 4.7, Table S48).

## Checking a fingerprint

The files are stored byte for byte (`.gitattributes` turns off line-ending conversion). To recompute the fingerprint of the first *k* lines:

```
python -c "import hashlib,sys; f,k=sys.argv[1],int(sys.argv[2]); print(hashlib.sha256(b''.join(open(f,'rb').read().splitlines(True)[:k])).hexdigest())" doc/분석R1-3_계획.md 35
```

On Linux or macOS, `head -n 35 doc/분석R1-3_계획.md | sha256sum` gives the same value.
