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
      label: 'Evaluation',
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
    value: 'Five peer systems',
    note: 'The manuscript reports five primary comparison systems while the reviewer-facing workspace stays blinded.',
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
      'On EHRFlowBench, HealthFlow reaches an LLM score of 3.82 and 3.98 with tool support, outperforming the strongest highlighted peer result in the manuscript.',
    stats: [
      { label: 'HealthFlow', value: '3.82' },
      { label: 'HealthFlow + tools', value: '3.98' },
      { label: 'Best highlighted peer result', value: '3.31' },
    ],
  },
  {
    id: 'artifacts',
    label: 'Artifacts',
    title: 'Artifact-heavy tasks improve even more with tool access.',
    summary:
      'On MedAgentBoard, HealthFlow reaches 66.09% success and 81.89% with tool access, well above the strongest highlighted peer result in the manuscript.',
    stats: [
      { label: 'HealthFlow', value: '66.09%' },
      { label: 'HealthFlow + tools', value: '81.89%' },
      { label: 'Best highlighted peer result', value: '45.61%' },
    ],
  },
  {
    id: 'reasoning',
    label: 'Reasoning',
    title: 'The benchmark suite still checks medical reasoning, not only execution.',
    summary:
      'HealthFlow is evaluated across MedAgentsBench, HLE, and CureBench in addition to the workflow-heavy tasks, so the platform narrative stays anchored in five benchmarks rather than only two.',
    stats: [
      { label: 'Reasoning benchmarks', value: '3' },
      { label: 'Workflow benchmarks', value: '2' },
      { label: 'Total suite', value: '5' },
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
