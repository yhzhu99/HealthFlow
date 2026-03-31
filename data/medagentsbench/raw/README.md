---
language:
- en
license: mit
size_categories:
- 10K<n<100K
task_categories:
- question-answering
pretty_name: MedAgentsBench
dataset_info:
- config_name: AfrimedQA
  features:
  - name: realidx
    dtype: string
  - name: question
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
    - name: E
      dtype: string
  - name: answer_idx
    dtype: string
  - name: reason
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test_hard
    num_bytes: 19024
    num_examples: 32
  - name: test
    num_bytes: 101481
    num_examples: 174
  download_size: 98255
  dataset_size: 120505
- config_name: MMLU
  features:
  - name: realidx
    dtype: int64
  - name: question
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
  - name: answer_idx
    dtype: string
  - name: subject
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test_hard
    num_bytes: 34527
    num_examples: 73
  - name: test
    num_bytes: 530715
    num_examples: 1089
  download_size: 333719
  dataset_size: 565242
- config_name: MMLU-Pro
  features:
  - name: realidx
    dtype: int64
  - name: question
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
    - name: E
      dtype: string
    - name: F
      dtype: string
    - name: G
      dtype: string
    - name: H
      dtype: string
    - name: I
      dtype: string
    - name: J
      dtype: string
  - name: answer_idx
    dtype: string
  - name: category
    dtype: string
  - name: src
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test_hard
    num_bytes: 67251
    num_examples: 100
  - name: test
    num_bytes: 562708
    num_examples: 818
  download_size: 355299
  dataset_size: 629959
- config_name: MedBullets
  features:
  - name: realidx
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
    - name: E
      dtype: string
  - name: answer_idx
    dtype: string
  - name: reason
    dtype: string
  splits:
  - name: test_hard
    num_bytes: 344730
    num_examples: 89
  - name: test
    num_bytes: 1215107
    num_examples: 308
  download_size: 803568
  dataset_size: 1559837
- config_name: MedExQA
  features:
  - name: realidx
    dtype: string
  - name: question
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
  - name: answer_idx
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test_hard
    num_bytes: 31640
    num_examples: 100
  - name: test
    num_bytes: 259180
    num_examples: 935
  download_size: 188019
  dataset_size: 290820
- config_name: MedMCQA
  features:
  - name: question
    dtype: string
  - name: realidx
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
  - name: answer_idx
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test_hard
    num_bytes: 23787
    num_examples: 100
  - name: test
    num_bytes: 695464
    num_examples: 2816
  download_size: 556606
  dataset_size: 719251
- config_name: MedQA
  features:
  - name: realidx
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
  - name: meta_info
    dtype: string
  - name: answer_idx
    dtype: string
  - name: metamap_phrases
    sequence: string
  splits:
  - name: test_hard
    num_bytes: 166219
    num_examples: 100
  - name: test
    num_bytes: 1956214
    num_examples: 1273
  download_size: 1132393
  dataset_size: 2122433
- config_name: MedXpertQA-R
  features:
  - name: realidx
    dtype: string
  - name: question
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
    - name: E
      dtype: string
    - name: F
      dtype: string
    - name: G
      dtype: string
    - name: H
      dtype: string
    - name: I
      dtype: string
    - name: J
      dtype: string
  - name: answer_idx
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test_hard
    num_bytes: 157828
    num_examples: 100
  - name: test
    num_bytes: 3039101
    num_examples: 1861
  download_size: 1934767
  dataset_size: 3196929
- config_name: MedXpertQA-U
  features:
  - name: realidx
    dtype: string
  - name: question
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
    - name: E
      dtype: string
    - name: F
      dtype: string
    - name: G
      dtype: string
    - name: H
      dtype: string
    - name: I
      dtype: string
    - name: J
      dtype: string
  - name: answer_idx
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test_hard
    num_bytes: 129016
    num_examples: 100
  - name: test
    num_bytes: 765350
    num_examples: 589
  download_size: 590827
  dataset_size: 894366
- config_name: PubMedQA
  features:
  - name: realidx
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: answer_rationale
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
  - name: answer_idx
    dtype: string
  splits:
  - name: test_hard
    num_bytes: 181356
    num_examples: 100
  - name: test
    num_bytes: 884114
    num_examples: 500
  download_size: 602182
  dataset_size: 1065470
configs:
- config_name: AfrimedQA
  data_files:
  - split: test_hard
    path: AfrimedQA/test_hard-*
  - split: test
    path: AfrimedQA/test-*
- config_name: MMLU
  data_files:
  - split: test_hard
    path: MMLU/test_hard-*
  - split: test
    path: MMLU/test-*
- config_name: MMLU-Pro
  data_files:
  - split: test_hard
    path: MMLU-Pro/test_hard-*
  - split: test
    path: MMLU-Pro/test-*
- config_name: MedBullets
  data_files:
  - split: test_hard
    path: MedBullets/test_hard-*
  - split: test
    path: MedBullets/test-*
- config_name: MedExQA
  data_files:
  - split: test_hard
    path: MedExQA/test_hard-*
  - split: test
    path: MedExQA/test-*
- config_name: MedMCQA
  data_files:
  - split: test_hard
    path: MedMCQA/test_hard-*
  - split: test
    path: MedMCQA/test-*
- config_name: MedQA
  data_files:
  - split: test_hard
    path: MedQA/test_hard-*
  - split: test
    path: MedQA/test-*
- config_name: MedXpertQA-R
  data_files:
  - split: test_hard
    path: MedXpertQA-R/test_hard-*
  - split: test
    path: MedXpertQA-R/test-*
- config_name: MedXpertQA-U
  data_files:
  - split: test_hard
    path: MedXpertQA-U/test_hard-*
  - split: test
    path: MedXpertQA-U/test-*
- config_name: PubMedQA
  data_files:
  - split: test_hard
    path: PubMedQA/test_hard-*
  - split: test
    path: PubMedQA/test-*
tags:
- medical
---

# MedAgentsBench Dataset

## Overview
This dataset is part of the MedAgentsBench, which focuses on benchmarking thinking models and agent frameworks for complex medical reasoning. The benchmark contains challenging medical questions specifically selected where models achieve less than 50% accuracy.

## Dataset Structure
The benchmark includes the following medical question-answering datasets:

| Dataset | Description |
|---------|-------------|
| MedQA | Medical domain question answering dataset |
| PubMedQA | Questions based on PubMed abstracts |
| MedMCQA | Multiple-choice questions from medical entrance exams |
| MedBullets | Clinical case-based questions |
| MMLU | Medical subset from Massive Multitask Language Understanding |
| MMLU-Pro | Advanced version of MMLU with more complex questions |
| AfrimedQA | Medical questions focused on African healthcare contexts |
| MedExQA | Expert-level medical questions |
| MedXpertQA-R | Medical expert reasoning questions |
| MedXpertQA-U | Medical expert understanding questions |

## Dataset Splits
Each dataset contains:
- `test_hard`: Specifically curated hard questions (accuracy <50%)
- `test`: Complete test set

## Citation
If you use this dataset in your research, please cite the MedAgentsBench paper:
```
@inproceedings{tang2025medagentsbench,
  title={MedAgentsBench: Benchmarking Thinking Models and Agent Frameworks for Complex Medical Reasoning},
  author = {Tang, Xiangru and Shao, Daniel and Sohn, Jiwoong and Chen, Jiapeng and Zhang, Jiayi and Xiang, Jinyu and Wu, Fang and Zhao, Yilun and Wu, Chenglin and Shi, Wenqi and Cohan, Arman and Gerstein, Mark},
  journal = {arXiv preprint arXiv:2503.07459},
  year = {2025},
}
```

## License
Please refer to the repository for license information.

## Additional Information
For more details, visit the [MedAgentsBench repository](https://github.com/gersteinlab/medagents-benchmark) or the [paper](https://huggingface.co/papers/2503.07459).