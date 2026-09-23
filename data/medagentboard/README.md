# MedAgentBoard evaluation

The 100 tasks have human reference solutions. These remain the comparison anchors: correct alternative methods and semantically equivalent outputs are acceptable. The published evaluation reports success rates separately for TJH and MIMIC-IV across data extraction, predictive modeling and visualization.

The modeling references use the following task-specific protocols. The supplementary `processed/test.protocol.jsonl` places these instructions, prediction populations and available fold assignments directly in the task text. It preserves task membership and all reference answers; the original `processed/test.jsonl` remains available.

| qid | dataset | prediction rows | reference protocol |
|---|---|---|---|
| 18 | TJH | 361 | Generate predictions for all keys using 5-fold StratifiedKFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 19 | TJH | 361 | Fit all entities and report fitted predictions and in-sample metrics for all keys; do not substitute OOF predictions. |
| 20 | TJH | 361 | Generate predictions for all keys using 5-fold StratifiedKFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 21 | TJH | 37 | Fit official train+val; predict official test only. |
| 22 | TJH | 37 | Fit official train+val; predict official test only. |
| 23 | TJH | 37 | Fit official train+val; predict official test only. |
| 24 | TJH | 361 | Generate predictions for all keys using 5-fold StratifiedKFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 25 | TJH | 73 | Fit the complement of the supplied prediction keys; predict this custom 20% test set (LOS-quintile stratification, seed 42). |
| 26 | TJH | 37 | Fit official train+val; predict official test only. |
| 27 | TJH | 361 | Generate predictions for all keys using 5-fold StratifiedKFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 28 | TJH | 361 | Generate predictions for all keys using 5-fold KFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 29 | TJH | 361 | Generate predictions for all keys using 5-fold StratifiedKFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 30 | TJH | 37 | Fit official train+val; predict official test only. |
| 31 | TJH | 37 | Fit official train+val; predict official test only. |
| 32 | TJH | 37 | Fit official train+val; predict official test only. |
| 33 | TJH | 361 | Generate predictions for all keys using 5-fold StratifiedKFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 34 | TJH | 361 | Generate predictions for all keys using 5-fold KFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 68 | MIMIC-IV | 12 | Split by MD5(PatientID) into 70/15/15 train/val/test; fit train+val and predict test only. |
| 69 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold StratifiedKFold (shuffle=True, random_state=0, group=None). Report out-of-fold metrics. |
| 70 | MIMIC-IV | 13 | Fit official train+val; predict official test only. |
| 71 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold StratifiedGroupKFold (shuffle=True, random_state=42, group=PatientID). Report out-of-fold metrics. |
| 72 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold KFold (shuffle=True, random_state=72, group=None). Report out-of-fold metrics. |
| 73 | MIMIC-IV | 13 | Fit official train+val; predict official test only. |
| 74 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold KFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 75 | MIMIC-IV | 13 | Fit official train+val; predict official test only. |
| 76 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold KFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 77 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold StratifiedKFold (shuffle=True, random_state=42, group=None). Report out-of-fold metrics. |
| 78 | MIMIC-IV | 128 | Fit all entities and report fitted predictions and in-sample metrics for all keys; do not substitute OOF predictions. |
| 79 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold StratifiedKFold (shuffle=True, random_state=0, group=None). Report out-of-fold metrics. |
| 80 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold KFold (shuffle=False, random_state=None, group=None). Report both fitted and CV outputs separately. |
| 81 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold StratifiedGroupKFold (shuffle=True, random_state=42, group=PatientID). Report out-of-fold metrics. |
| 82 | MIMIC-IV | 13 | Fit official train; predict official test only. |
| 83 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold StratifiedKFold (shuffle=True, random_state=0, group=None). Report out-of-fold metrics. |
| 84 | MIMIC-IV | 128 | Generate predictions for all keys using 5-fold GroupKFold (shuffle=False, random_state=None, group=PatientID). Report out-of-fold metrics. |

q25 uses an 80/20 test split with seed 42, stratified by `pd.qcut(LOS_days, q=5, duplicates="drop")`; `LOS_days` is discharge minus admission in whole days. Its 73 prediction keys differ from the official validation set. q68 uses the PatientID hash split described in the annotated task; its time origin is the first recorded observation. q82 fits only the official train split. The other official-test tasks fit train+val.

The original `scripts/evaluate.py` and judge prompts are unchanged. That implementation uses a 0–10 rubric with threshold 7; the published paper describes Gemini 3.1 Flash-Lite binary success, allowing approximately 10% metric deviation when model behavior is materially consistent. These evaluation outputs should be labeled by the implementation used. The annotated task view adds protocol information, so new runs should also identify that input view.
