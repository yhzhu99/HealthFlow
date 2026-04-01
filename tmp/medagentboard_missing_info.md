# MedAgentBoard 缺失关键信息清单

判定标准：
- 仅记录那些会影响题目是否可直接执行，或会让 benchmark 标准答案不可复现的缺失项。
- 不把“模型/统计方法可以自由选择”这类普通开放性要求全部算作缺失；这里只保留会影响核心数据抽取、标签定义、时间窗、阈值、聚类数、目标变量等的情况。

## 全局问题

### G1. `Data Path` 与当前仓库实际文件不一致
- 影响题号：`Q1-Q100`
- 现状：
  - 题面统一写的是 `/home/projects/HealthFlow/healthflow_datasets/TJH.csv` 或 `/home/projects/HealthFlow/healthflow_datasets/MIMIC-IV.parquet`
  - 当前仓库里实际可用的数据文件是：
    - `data/medagentboard/processed/tjh/tjh_formatted_ehr.parquet`
    - `data/medagentboard/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet`
- 补充建议：
  - 直接把题面中的 `Data Path` 改成仓库内相对路径。
  - 最好统一为：
    - `data/medagentboard/processed/tjh/tjh_formatted_ehr.parquet`
    - `data/medagentboard/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet`

### G2. MIMIC 题面字段名与当前 parquet 实际字段格式不一致
- 影响题号：`Q51-Q100`
- 现状：
  - 题面示例使用标量字段，例如 `Capillary refill rate`、`Glascow coma scale total`
  - 当前 parquet 里这些字段已经做成 one-hot / expanded columns，例如：
    - `Capillary refill rate->0.0`
    - `Glascow coma scale total->15`
- 补充建议：
  - 二选一：
    - 提供一份与题面字段完全一致的“未 one-hot 版本” MIMIC parquet
    - 在题面附上一段固定映射规则，说明如何从 one-hot 字段还原到题面中的标量字段

## A. 明确缺少必要实体或目标定义

### Q54
- 缺失信息：
  - 题目要求画某个 `specific AdmissionID` 的心率曲线，但没有给出具体 `AdmissionID`
- 为什么算硬缺失：
  - 不给 admission，就没有唯一对象可画
- 补充建议：
  - 在题面中直接指定 `AdmissionID`
  - 建议写成固定值，例如：`Use AdmissionID = 23488965`

### Q88
- 缺失信息：
  - 题目要求预测 “a specific physiological measurement”，但没有指定具体是哪一个 physiological measurement
- 为什么算硬缺失：
  - 目标变量未定义，任务本身不成立
- 补充建议：
  - 明确写出目标变量名
  - 建议补成：
    - `Predict the next recorded value of Heart Rate`
    - 或 `Predict the next recorded value of Mean blood pressure`
  - 同时固定评价指标，例如 `MAE + RMSE + R2`

## B. 标签、阈值或目标构造规则缺失

### Q39
- 缺失信息：
  - “critical laboratory value”
  - “predefined set of critical lab values”
  - “normal clinical range”
- 为什么会影响 benchmark：
  - 不同实验室项目和正常范围会直接改变标签 `0/1`
- 补充建议：
  - 固定一个 lab 列表
  - 为每个 lab 明确给出上下界
  - 例：
    - `critical labs = [White blood cell count, Platelet count, Creatinine, Hypersensitive c-reactive protein]`
    - 并逐项给出阈值表

### Q43
- 缺失信息：
  - `Increasing / Decreasing / Stable` 的划分标准没有定义
- 为什么会影响 benchmark：
  - 类别标签取决于阈值，直接影响训练集标签与评估结果
- 补充建议：
  - 固定为相对变化率或绝对变化量
  - 例如：
    - `relative_change >= +10% => Increasing`
    - `relative_change <= -10% => Decreasing`
    - `otherwise => Stable`

### Q83
- 缺失信息：
  - 各 vital sign 的 abnormal / critical thresholds 没有固定
- 为什么会影响 benchmark：
  - 阈值不同，会改变“异常计数”和“异常组合”
- 补充建议：
  - 在题面直接附一个阈值表
  - 例如固定：
    - `Heart Rate < 60 or > 100`
    - `Mean blood pressure < 65`
    - `Respiratory rate < 12 or > 20`
    - `Oxygen saturation < 90`
    - `Temperature < 36 or > 38`
    - `Glucose < 70 or > 180`
    - `pH < 7.35 or > 7.45`

### Q89
- 缺失信息：
  - “significant clinical deterioration” 没有固定定义
- 为什么会影响 benchmark：
  - 事件标签完全依赖该定义
- 补充建议：
  - 在题面中固定 deterioration 规则
  - 例如：
    - 后续 24 小时内出现任一情况即记为 1：
    - `Mean blood pressure < 60`
    - `Oxygen saturation < 90`
    - `Respiratory rate > 30`
    - `Glascow coma scale total < 8`

### Q90
- 缺失信息：
  - “overall physiological instability” 的定量方式不够固定
  - “key vital signs” 没有明确列表
  - “variability” 没有明确用 `std`、`range`、`IQR` 还是别的定义
- 为什么会影响 benchmark：
  - 目标变量本身不唯一
- 补充建议：
  - 固定 instability 的计算公式
  - 例如：
    - `instability = mean(zscore(std(Heart Rate), std(Mean blood pressure), std(Respiratory rate), std(Oxygen saturation), std(Temperature)))`
  - 或更简单：
    - 对固定 5 个 vital sign 求 admission 内标准差，再取平均

## C. 任务协议未固定，导致标准答案不可复现

### Q21
- 缺失信息：
  - prediction time points `T` 未固定
- 补充建议：
  - 固定为 `T in {1, 3, 5, 7, 10, 14}` days after admission

### Q25
- 缺失信息：
  - early window 写成 `48 or 72 hours`
  - lab subset 未固定
- 补充建议：
  - 二选一固定，例如只用 `72 hours`
  - 再固定一组 lab 列名

### Q26
- 缺失信息：
  - selected parameters 未固定
  - analytical window 未固定
  - `LOS` 是做 regression 还是 duration bins classification 未固定
- 补充建议：
  - 固定参数列表
  - 固定分析窗口，例如整个住院期
  - 固定 `LOS` 任务形式，例如只做 regression

### Q27
- 缺失信息：
  - intra-patient variability metric 未固定
- 补充建议：
  - 固定为 `std`
  - 若测量次数少于 2，则该病人的该指标 variability 记为缺失并从该指标分析中剔除

### Q28
- 缺失信息：
  - 使用 absolute change 还是 percentage change 未固定
- 补充建议：
  - 固定一种
  - 建议优先固定为 absolute change，避免分母接近 0 的问题

### Q29
- 缺失信息：
  - parameter subset 未固定
  - pre-discharge window 内若某参数有多条记录，取值规则虽写“latest”，但缺失处理未固定
- 补充建议：
  - 固定参数列表
  - 固定缺失策略，例如：
    - 不做插补
    - 只在该参数有足够样本的病人上比较

### Q30
- 缺失信息：
  - parameter subset 未固定
  - clustering algorithm 未固定
  - cluster number 未固定
- 补充建议：
  - 固定算法，例如 `KMeans`
  - 固定 `k = 3`
  - 固定输入参数列表和缺失值处理方式

### Q31
- 缺失信息：
  - parameter subset 未固定
  - “measurement density” 的定义未固定
- 补充建议：
  - 明确 density 为：
    - `measurement_count / LOS_days`
  - 固定参数列表

### Q32
- 缺失信息：
  - parameter subset 未固定
  - 只有一次测量时如何处理 peak/nadir 的规则未固定
- 补充建议：
  - 固定参数列表
  - 明确：
    - 若仅一次测量，则 `peak = nadir = observed value`

### Q33
- 缺失信息：
  - “typical range” 的统计定义未固定
- 补充建议：
  - 固定一个方法，例如：
    - `typical range = [Q1 - 1.5*IQR, Q3 + 1.5*IQR]`
  - 或
    - `typical range = mean ± 2*std`

### Q34
- 缺失信息：
  - target day 未固定
  - search window 未固定
  - parameter subset 未固定
  - missing value handling 未固定
  - significance level 未固定
- 说明：
  - 当前参考答案实际采用了隐藏默认值，如 `day 3`、`±1 day`
- 补充建议：
  - 直接把以下内容写入题面：
    - `target day = 3`
    - `search window = +/- 1 day`
    - 固定参数列表
    - `alpha = 0.05`
    - 明确缺失策略，例如中位数插补或完整案例分析

### Q35
- 缺失信息：
  - frequent itemset mining 的 `minimum support threshold` 未固定
- 补充建议：
  - 固定支持度阈值，例如 `min_support = 0.1`
  - 同时固定只比较 top `k=10` itemsets

### Q68
- 缺失信息：
  - sequence model 的观察窗口/截断策略未固定
  - “progressively longer segments” 的时间切片未固定
- 补充建议：
  - 固定时间切片，例如 `6h, 12h, 24h, 48h, full stay`
  - 固定序列截断或 padding 规则

### Q71
- 缺失信息：
  - clustering algorithm 未固定
  - cluster 数量未固定
- 补充建议：
  - 固定为 `KMeans(k=3)` 或 `GaussianMixture(n_components=3)`
  - 若允许自动选 `k`，则需明确评价准则，例如 `silhouette score`

### Q73
- 缺失信息：
  - 报告按 `Outcome` 分组还是按 `Readmission` 分组未固定
  - 关键时间变量与“critical points” 判定标准未固定
- 补充建议：
  - 固定主分组目标，例如仅按 `Outcome`
  - 固定要分析的时间轴，例如相对 admission 的小时数/天数

### Q76
- 缺失信息：
  - “key vital signs and clinical scores” 的最终列表未固定
  - 不同变量的 “worst” 方向只举例没有完全表化
- 补充建议：
  - 给出完整指标表
  - 对每个指标明确 worst direction：
    - `Heart Rate -> max`
    - `Oxygen saturation -> min`
    - `Glascow coma scale total -> min`
    - 等

### Q77
- 缺失信息：
  - variability measure 给了多个候选：`std / IQR / range`
- 补充建议：
  - 固定只用一种，例如 `std`

### Q78
- 缺失信息：
  - early change 窗口写成 `12 to 24 hours`，未固定
- 补充建议：
  - 固定为：
    - `first measurement vs closest measurement to +24h`
  - 或
    - `0-12h mean vs 12-24h mean`

### Q82
- 缺失信息：
  - `key physiological indicator` 未固定
- 补充建议：
  - 在题面明确指定一个指标
  - 例如：
    - `lowest recorded Oxygen saturation`
  - 并固定年龄分层规则，例如：
    - `<40`, `40-64`, `65+`

### Q84
- 缺失信息：
  - first 24h 特征提取写成 `first value or mean value`
- 补充建议：
  - 固定一种
  - 建议统一为：
    - 对连续变量取 `first 24h mean`
    - 对静态变量取首条记录值

### Q91
- 缺失信息：
  - 24 小时附近若有多条 GCS Total 记录，题面允许 `closest` 或 `suitable aggregate`
- 补充建议：
  - 固定为：
    - `target = GCS Total at the record closest to admission + 24h`
  - 如有并列，固定选时间更晚的一条

## 建议优先补充顺序

### 第一优先级
- `G1`
- `G2`
- `Q54`
- `Q88`
- `Q39`
- `Q43`
- `Q83`
- `Q89`
- `Q90`

### 第二优先级
- `Q21`
- `Q25-Q35`
- `Q68`
- `Q71`
- `Q73`
- `Q76`
- `Q77`
- `Q78`
- `Q82`
- `Q84`
- `Q91`

## 最小修补方案

如果你想先把这批题修到“可以稳定评测”的程度，建议至少补这 4 类：
- 把所有 `Data Path` 改成仓库内真实路径
- 给所有 MIMIC 题补字段映射说明，或换成未 one-hot 的 parquet
- 给 `Q54`、`Q88` 补实体/目标变量
- 给 `Q39`、`Q43`、`Q83`、`Q89`、`Q90` 补标签与阈值定义
