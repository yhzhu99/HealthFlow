export const projectMeta = {
  name: 'HealthFlow',
  eyebrow: 'HealthFlow',
  title: 'HealthFlow: Automating electronic health record analysis via a strategically self-evolving multi-agent framework',
  abstract:
    'Electronic health records (EHRs) are a rich source of real-world clinical data, but turning them into valid analyses remains slow, brittle, and expert-intensive. Although recent AI agents can answer medical questions and use tools, automating full EHR workflows remains difficult because small planning or execution errors can invalidate otherwise plausible analyses. Here we present HealthFlow, a multi-agent framework that converts prior EHR analyses into structured, governed experience for planning under dataset-specific and methodological constraints. We also introduce EHRFlowBench, a benchmark of realistic EHR analysis tasks derived from 51,280 peer-reviewed papers. Across EHRFlowBench and four established benchmarks (MedAgentBoard, MedAgentsBench, HLE, and CureBench), HealthFlow consistently outperforms strong baselines in generating valid clinical artifacts and completing complex EHR analysis pipelines. These results show that governed reuse of prior analytical experience improves robustness in health data science and provides a scalable path to automating open-ended EHR analysis.',
  authors: [
    { name: 'Yinghao Zhu', marks: ['1', '2', '*'] },
    { name: 'Zixiang Wang', marks: ['1', '*'] },
    { name: 'Yifan Qi', marks: ['1', '*'] },
    { name: 'Lei Gu', marks: ['1'] },
    { name: 'Dehao Sui', marks: ['1'] },
    { name: 'Haoran Hu', marks: ['1'] },
    { name: 'Xichen Zhang', marks: ['3'] },
    { name: 'Ziyi He', marks: ['2'] },
    { name: 'Yasha Wang', marks: ['1'] },
    { name: 'Junjun He', marks: ['4'] },
    { name: 'Liantao Ma', marks: ['1', '†'] },
    { name: 'Lequan Yu', marks: ['2', '†'] },
  ],
  affiliations: [
    {
      mark: '1',
      text: 'National Engineering Research Center for Software Engineering, Peking University, Beijing, China, 100871',
    },
    {
      mark: '2',
      text: 'School of Computing and Data Science, The University of Hong Kong, Hong Kong SAR, China, 999077',
    },
    {
      mark: '3',
      text: 'Department of Computer Science and Engineering, The Hong Kong University of Science and Technology, Hong Kong SAR, China, 999077',
    },
    {
      mark: '4',
      text: 'Shanghai Artificial Intelligence Laboratory, Shanghai, China, 200232',
    },
  ],
  notes: [
    { mark: '*', text: 'These authors contributed equally to this work' },
    { mark: '†', text: 'Correspondence to malt@pku.edu.cn, lqyu@hku.hk' },
  ],
  links: [
    {
      label: 'Code',
      href: 'https://github.com/yhzhu99/HealthFlow',
      kind: 'primary' as const,
    },
    {
      label: 'Local Evaluation',
      href: '/evaluation',
      kind: 'secondary' as const,
    },
  ],
}

export const sectionNav = [
  { id: 'overview', label: 'Overview' },
  { id: 'framework', label: 'Framework' },
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'results', label: 'Results' },
]

export const projectFacts = [
  {
    label: 'Runtime',
    value: 'Meta -> Executor -> Evaluator -> Reflector',
    note: 'Four-agent loop that plans, executes, critiques, and writes back governed experience.',
  },
  {
    label: 'Benchmarks',
    value: '5 benchmark suites',
    note: 'EHRFlowBench, MedAgentBoard, MedAgentsBench, HLE, and CureBench.',
  },
  {
    label: 'Comparison set',
    value: 'HealthFlow + 15 baselines',
    note: 'The overall comparison spans general LLMs, medical LLMs, collaborative medical agents, general agents, and biomedical agents.',
  },
]

export const frameworkStages = [
  {
    id: 'meta',
    shortLabel: 'Meta',
    title: 'Meta agent',
    kicker: 'Strategic planner',
    description:
      'Profiles the task, retrieves bounded EHR-aware experience, and writes the next attempt-specific plan under dataset, schema, and risk constraints.',
    outputs: ['Task family + dataset signature', 'Retrieved safeguards and workflows', 'Attempt-specific plan'],
  },
  {
    id: 'executor',
    shortLabel: 'Executor',
    title: 'Executor agent',
    kicker: 'Faithful implementation',
    description:
      'Turns the plan into code, tool calls, and artifacts through an external coding-agent backend in a sandboxed workspace.',
    outputs: ['CLI and tool actions', 'Intermediate artifacts', 'Auditable execution trace'],
  },
  {
    id: 'evaluator',
    shortLabel: 'Evaluator',
    title: 'Evaluator agent',
    kicker: 'Artifact-and-trace critique',
    description:
      'Reviews runtime traces and generated outputs so the system can repair hidden methodological errors instead of only catching crashes.',
    outputs: ['Success verdict', 'Violated constraints', 'Targeted repair guidance'],
  },
  {
    id: 'reflector',
    shortLabel: 'Reflector',
    title: 'Reflector agent',
    kicker: 'Governed memory writeback',
    description:
      'Synthesizes safeguards, reusable workflows, dataset anchors, and code snippets from the completed trajectory so later tasks can reuse validated experience.',
    outputs: ['Safeguards', 'Reusable workflows', 'Dataset anchors + snippets'],
  },
]

export const benchmarkDeck = [
  {
    id: 'ehrflowbench',
    label: 'EHRFlowBench',
    kicker: 'Primary open-ended benchmark',
    title: 'Literature-derived end-to-end EHR analysis.',
    body:
      'EHRFlowBench evaluates whether an agent can move from a clinical objective to cohort construction, feature engineering, modeling, visualization, and report writing under real dataset constraints.',
    metrics: [
      { label: 'Papers screened', value: '51,280' },
      { label: 'Seed papers retained', value: '118' },
      { label: 'Final tasks', value: '100' },
    ],
    bullets: [
      'Collect 51,280 papers from major AI and data mining venues between 2020 and 2025.',
      'Filter to 162 EHR-relevant candidates, then manually retain 118 seed papers.',
      'Generate 236 dataset-grounded tasks paired across TJH and the MIMIC-IV public sample release.',
      'Sample a balanced 100-task benchmark with 50 TJH tasks and 50 MIMIC-IV tasks.',
    ],
  },
  {
    id: 'medagentboard',
    label: 'MedAgentBoard',
    kicker: 'Executable artifact benchmark',
    title: 'Workflow-grounded review of data, models, and visualizations.',
    body:
      'MedAgentBoard complements EHRFlowBench with a more deterministic task surface spanning data processing, predictive modeling, and visualization, making artifact quality directly inspectable.',
    metrics: [
      { label: 'Executable tasks', value: '100' },
      { label: 'Datasets', value: 'TJH + MIMIC-IV public sample release' },
      { label: 'Task families', value: '3' },
    ],
    bullets: [
      'Balanced over 50 TJH tasks and 50 MIMIC-IV tasks.',
      'Targets data processing, predictive modeling, and visualization in equal proportion.',
      'Supports artifact-first review through images, CSV files, markdown, and reports.',
    ],
  },
  {
    id: 'medagentsbench',
    label: 'MedAgentsBench',
    kicker: 'Deterministic reasoning benchmark',
    title: 'Hard-set medical knowledge and reasoning.',
    body:
      'Used to measure whether stronger workflow performance still preserves strong medical reasoning on deterministic multiple-choice questions.',
    metrics: [
      { label: 'Format', value: 'Multiple choice' },
      { label: 'Focus', value: 'Medical reasoning' },
      { label: 'Role', value: 'Knowledge floor' },
    ],
    bullets: [
      'Evaluates hard medical QA rather than open-ended execution.',
      'Acts as a reasoning anchor beside workflow-heavy evaluations.',
    ],
  },
  {
    id: 'hle',
    label: 'HLE',
    kicker: 'External benchmark coverage',
    title: 'Selected Biology and Medicine questions from Humanity’s Last Exam.',
    body:
      'Adds broader scientific reasoning pressure so the evaluation suite is not limited to EHR-native tasks alone.',
    metrics: [
      { label: 'Source', value: 'Humanity’s Last Exam' },
      { label: 'Subset', value: 'Biology + Medicine' },
      { label: 'Role', value: 'Cross-domain pressure' },
    ],
    bullets: [
      'Tests broader science reasoning under deterministic evaluation.',
      'Helps separate workflow gains from general question-answering ability.',
    ],
  },
  {
    id: 'curebench',
    label: 'CureBench',
    kicker: 'Biomedical decision-making',
    title: 'Tool-augmented biomedical reasoning under controlled grading.',
    body:
      'Adds a biomedical decision-making benchmark so the suite covers not only workflow orchestration but also practical biomedical reasoning with structured answers.',
    metrics: [
      { label: 'Format', value: 'Deterministic subset' },
      { label: 'Focus', value: 'Biomedical decisions' },
      { label: 'Role', value: 'Practical reasoning' },
    ],
    bullets: [
      'Used as a deterministic external check on biomedical reasoning.',
      'Completes the five-benchmark evaluation surface reported in the manuscript.',
    ],
  },
]

export const resultDeck = [
  {
    id: 'open-ended',
    label: 'Open-ended EHR',
    title: 'HealthFlow leads on the hardest open-ended workflows.',
    summary:
      'On EHRFlowBench, HealthFlow reaches an LLM score of 4.01 +/- 0.12, ahead of the strongest baseline, Biomni, at 3.76 +/- 0.16.',
    stats: [
      { label: 'HealthFlow', value: '4.01 +/- 0.12' },
      { label: 'Best baseline', value: '3.76 +/- 0.16' },
      { label: 'Benchmark tasks', value: '100' },
    ],
  },
  {
    id: 'artifacts',
    label: 'Artifacts',
    title: 'Artifact-heavy workflow tasks show the clearest executable gains.',
    summary:
      'On MedAgentBoard, HealthFlow reaches 63.05% +/- 4.78 success, compared with 50.10% +/- 5.01 for the strongest baseline.',
    stats: [
      { label: 'HealthFlow', value: '63.05% +/- 4.78' },
      { label: 'Best baseline', value: '50.10% +/- 5.01' },
      { label: 'Task families', value: '3' },
    ],
  },
  {
    id: 'reasoning',
    label: 'Reasoning',
    title: 'The benchmark suite still checks medical reasoning, not only execution.',
    summary:
      'HealthFlow also leads the deterministic reasoning benchmarks, reaching 32.40% on MedAgentsBench, 10.20% on HLE, and 93.02% on CureBench.',
    stats: [
      { label: 'MedAgentsBench', value: '32.40% +/- 4.56' },
      { label: 'HLE', value: '10.20% +/- 2.98' },
      { label: 'CureBench', value: '93.02% +/- 2.53' },
    ],
  },
  {
    id: 'human-eval',
    label: 'Human evaluation',
    title: 'Blinded expert review favors HealthFlow outputs.',
    summary:
      'Twelve domain experts reviewed 20 tasks in a blinded online interface and consistently preferred HealthFlow, with 71.54% average pairwise inter-rater agreement.',
    stats: [
      { label: 'Experts', value: '12' },
      { label: 'Tasks', value: '20' },
      { label: 'Agreement', value: '71.54%' },
    ],
  },
]

type ResultMetricCell = {
  value: string
  std?: string
  muted?: boolean
}

const metricCell = (value: string, std?: string): ResultMetricCell => ({ value, std })
const mutedCell = (): ResultMetricCell => ({ value: '--', muted: true })

export const resultTables = [
  {
    id: 'overall',
    label: 'Overall',
    title: 'Overall benchmark comparison',
    description:
      'HealthFlow achieves the best performance across all five benchmarks, with the clearest gains on workflow-intensive EHRFlowBench and MedAgentBoard tasks.',
    columns: [
      'EHRFlowBench LLM Score',
      'MedAgentBoard Success Rate (%)',
      'MedAgentsBench Accuracy (%)',
      'HLE Accuracy (%)',
      'CureBench Accuracy (%)',
    ],
    rows: [
      {
        category: 'General LLM',
        method: 'DeepSeek-V3.2',
        cells: [mutedCell(), mutedCell(), metricCell('17.98', '3.88'), metricCell('8.95', '2.86'), metricCell('82.95', '3.77')],
      },
      {
        category: 'General LLM',
        method: 'DeepSeek-V3.2-Think',
        cells: [mutedCell(), mutedCell(), metricCell('31.90', '4.67'), metricCell('9.90', '2.95'), metricCell('83.40', '3.70')],
      },
      {
        category: 'Medical LLM',
        method: 'Baichuan-M2',
        cells: [mutedCell(), mutedCell(), metricCell('16.00', '3.67'), metricCell('7.02', '2.57'), metricCell('74.06', '4.39')],
      },
      {
        category: 'Medical LLM',
        method: 'HuatuoGPT-o1',
        cells: [mutedCell(), mutedCell(), metricCell('8.05', '2.70'), metricCell('9.60', '2.85'), metricCell('62.99', '4.82')],
      },
      {
        category: 'Medical LLM',
        method: 'MedGemma',
        cells: [mutedCell(), mutedCell(), metricCell('22.00', '4.13'), metricCell('8.00', '2.69'), metricCell('69.92', '4.63')],
      },
      {
        category: 'Multi-Agent Collaboration',
        method: 'ColaCare',
        cells: [mutedCell(), mutedCell(), metricCell('21.00', '4.06'), metricCell('7.98', '2.74'), metricCell('80.02', '4.01')],
      },
      {
        category: 'Multi-Agent Collaboration',
        method: 'MDAgents',
        cells: [mutedCell(), mutedCell(), metricCell('17.97', '3.83'), metricCell('9.70', '2.90'), metricCell('79.97', '4.01')],
      },
      {
        category: 'Multi-Agent Collaboration',
        method: 'MedAgents',
        cells: [mutedCell(), mutedCell(), metricCell('19.93', '4.01'), metricCell('9.20', '2.82'), metricCell('80.94', '3.93')],
      },
      {
        category: 'Multi-Agent Collaboration',
        method: 'EvoMDT',
        cells: [mutedCell(), mutedCell(), metricCell('23.02', '4.21'), metricCell('7.00', '2.53'), metricCell('83.03', '3.74')],
      },
      {
        category: 'General Agent',
        method: 'AFlow',
        cells: [metricCell('1.05', '0.29'), metricCell('41.06', '4.93'), metricCell('30.40', '4.60'), metricCell('9.60', '2.88'), metricCell('79.98', '4.04')],
      },
      {
        category: 'General Agent',
        method: 'Alita',
        cells: [metricCell('0.50', '0.24'), metricCell('36.99', '4.82'), metricCell('19.99', '3.99'), metricCell('9.07', '2.83'), metricCell('81.98', '3.86')],
      },
      {
        category: 'Biomedical Agent',
        method: 'Biomni',
        cells: [metricCell('3.76', '0.16'), metricCell('50.10', '5.01'), metricCell('30.94', '4.59'), metricCell('9.95', '2.94'), metricCell('86.00', '3.46')],
      },
      {
        category: 'Biomedical Agent',
        method: 'STELLA',
        cells: [metricCell('1.44', '0.32'), metricCell('37.98', '4.86'), metricCell('13.99', '3.47'), metricCell('5.98', '2.40'), metricCell('81.97', '3.88')],
      },
      {
        category: 'Biomedical Agent',
        method: 'BioDSA',
        cells: [metricCell('2.94', '0.17'), metricCell('47.98', '4.99'), metricCell('28.02', '4.50'), metricCell('9.80', '2.92'), metricCell('87.01', '3.38')],
      },
      {
        category: 'Biomedical Agent',
        method: 'BioMedAgent',
        cells: [metricCell('0.40', '0.23'), metricCell('33.93', '4.74'), metricCell('21.09', '4.09'), metricCell('6.99', '2.56'), metricCell('80.96', '3.92')],
      },
      {
        category: 'Ours',
        method: 'HealthFlow',
        highlight: true,
        cells: [metricCell('4.01', '0.12'), metricCell('63.05', '4.78'), metricCell('32.40', '4.56'), metricCell('10.20', '2.98'), metricCell('93.02', '2.53')],
      },
    ],
  },
  {
    id: 'ehrflowbench',
    label: 'EHRFlowBench',
    title: 'Fine-grained EHRFlowBench breakdown',
    description:
      'Scores are reported for TJH and MIMIC-IV across method soundness, presentation quality, artifact generation, and overall performance.',
    columns: [
      'TJH Method Soundness',
      'TJH Presentation Quality',
      'TJH Artifact Generation',
      'TJH Overall Score',
      'MIMIC-IV Method Soundness',
      'MIMIC-IV Presentation Quality',
      'MIMIC-IV Artifact Generation',
      'MIMIC-IV Overall Score',
    ],
    rows: [
      {
        category: 'General Agent',
        method: 'AFlow',
        cells: [metricCell('1.10', '0.41'), metricCell('1.50', '0.51'), metricCell('1.20', '0.45'), metricCell('1.10', '0.41'), metricCell('1.00', '0.40'), metricCell('1.40', '0.55'), metricCell('1.20', '0.49'), metricCell('1.01', '0.40')],
      },
      {
        category: 'General Agent',
        method: 'Alita',
        cells: [metricCell('0.80', '0.41'), metricCell('1.10', '0.54'), metricCell('0.79', '0.42'), metricCell('0.80', '0.42'), metricCell('0.20', '0.19'), metricCell('0.30', '0.28'), metricCell('0.20', '0.19'), metricCell('0.20', '0.19')],
      },
      {
        category: 'Biomedical Agent',
        method: 'Biomni',
        cells: [metricCell('3.88', '0.23'), metricCell('4.08', '0.23'), metricCell('3.88', '0.23'), metricCell('3.80', '0.23'), metricCell('3.74', '0.22'), metricCell('4.30', '0.20'), metricCell('3.82', '0.21'), metricCell('3.72', '0.22')],
      },
      {
        category: 'Biomedical Agent',
        method: 'STELLA',
        cells: [metricCell('1.50', '0.52'), metricCell('1.70', '0.60'), metricCell('1.50', '0.51'), metricCell('1.50', '0.52'), metricCell('1.41', '0.40'), metricCell('2.00', '0.53'), metricCell('1.40', '0.41'), metricCell('1.40', '0.40')],
      },
      {
        category: 'Biomedical Agent',
        method: 'BioDSA',
        cells: [metricCell('2.74', '0.26'), metricCell('3.32', '0.27'), metricCell('2.78', '0.25'), metricCell('2.66', '0.25'), metricCell('3.24', '0.23'), metricCell('3.88', '0.20'), metricCell('3.37', '0.22'), metricCell('3.22', '0.23')],
      },
      {
        category: 'Biomedical Agent',
        method: 'BioMedAgent',
        cells: [metricCell('0.60', '0.41'), metricCell('0.70', '0.45'), metricCell('0.71', '0.45'), metricCell('0.59', '0.40'), metricCell('0.20', '0.19'), metricCell('0.20', '0.19'), metricCell('0.30', '0.28'), metricCell('0.20', '0.19')],
      },
      {
        category: 'Ours',
        method: 'HealthFlow',
        highlight: true,
        cells: [metricCell('4.00', '0.16'), metricCell('4.58', '0.10'), metricCell('4.16', '0.16'), metricCell('3.98', '0.17'), metricCell('4.04', '0.16'), metricCell('4.60', '0.09'), metricCell('4.14', '0.16'), metricCell('4.04', '0.16')],
      },
    ],
  },
  {
    id: 'medagentboard',
    label: 'MedAgentBoard',
    title: 'Fine-grained MedAgentBoard workflow breakdown',
    description:
      'Success rates are reported for TJH and MIMIC-IV across data extraction, predictive modeling, visualization, and overall task completion.',
    columns: [
      'TJH Data Extraction',
      'TJH Predictive Modeling',
      'TJH Visualization',
      'TJH Overall Score',
      'MIMIC-IV Data Extraction',
      'MIMIC-IV Predictive Modeling',
      'MIMIC-IV Visualization',
      'MIMIC-IV Overall Score',
    ],
    rows: [
      {
        category: 'General Agent',
        method: 'AFlow',
        cells: [metricCell('70.60', '10.95'), metricCell('17.74', '9.17'), metricCell('31.30', '11.56'), metricCell('39.93', '6.92'), metricCell('82.21', '9.26'), metricCell('17.59', '9.21'), metricCell('25.07', '10.75'), metricCell('42.16', '6.99')],
      },
      {
        category: 'General Agent',
        method: 'Alita',
        cells: [metricCell('58.79', '11.87'), metricCell('5.81', '5.64'), metricCell('31.35', '11.59'), metricCell('32.05', '6.72'), metricCell('82.26', '9.17'), metricCell('11.74', '7.70'), metricCell('31.28', '11.62'), metricCell('42.04', '6.93')],
      },
      {
        category: 'Biomedical Agent',
        method: 'Biomni',
        cells: [metricCell('82.42', '9.13'), metricCell('17.77', '9.22'), metricCell('49.95', '12.61'), metricCell('50.11', '7.12'), metricCell('88.24', '7.83'), metricCell('29.48', '10.97'), metricCell('31.28', '11.58'), metricCell('50.09', '7.15')],
      },
      {
        category: 'Biomedical Agent',
        method: 'STELLA',
        cells: [metricCell('76.62', '10.21'), metricCell('5.88', '5.73'), metricCell('37.50', '12.11'), metricCell('40.04', '6.97'), metricCell('70.51', '11.10'), metricCell('11.83', '7.80'), metricCell('25.05', '10.94'), metricCell('36.05', '6.81')],
      },
      {
        category: 'Biomedical Agent',
        method: 'BioDSA',
        cells: [metricCell('82.35', '9.38'), metricCell('11.79', '7.88'), metricCell('43.70', '12.34'), metricCell('45.97', '7.01'), metricCell('82.50', '9.22'), metricCell('35.42', '11.46'), metricCell('31.34', '11.50'), metricCell('50.02', '7.04')],
      },
      {
        category: 'Biomedical Agent',
        method: 'BioMedAgent',
        cells: [metricCell('46.99', '12.29'), metricCell('5.85', '5.74'), metricCell('31.15', '11.48'), metricCell('27.87', '6.35'), metricCell('82.47', '9.29'), metricCell('11.76', '7.80'), metricCell('25.20', '10.91'), metricCell('40.01', '6.97')],
      },
      {
        category: 'Ours',
        method: 'HealthFlow',
        highlight: true,
        cells: [metricCell('94.18', '5.71'), metricCell('41.09', '11.98'), metricCell('56.03', '12.49'), metricCell('64.01', '6.75'), metricCell('94.09', '5.72'), metricCell('47.06', '12.07'), metricCell('43.66', '12.34'), metricCell('61.99', '6.90')],
      },
    ],
  },
]
