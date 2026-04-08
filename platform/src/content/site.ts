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
    {
      label: 'Citation',
      href: '#citation',
      kind: 'ghost' as const,
    },
  ],
  details: [
    { label: 'Runtime', value: 'Meta -> Executor -> Evaluator -> Reflector' },
    { label: 'Benchmarks', value: 'EHRFlowBench, MedAgentBoard, MedAgentsBench, HLE, CureBench' },
    { label: 'Main Baselines', value: 'Alita, Biomni, STELLA, BioDSA, BioMedAgent' },
  ],
}

export const frameworkStages = [
  {
    title: 'Meta agent',
    description:
      'Profiles the task, retrieves bounded EHR-aware experience, and writes an attempt-specific plan under dataset, schema, and risk constraints.',
  },
  {
    title: 'Executor agent',
    description:
      'Carries out code and tool actions through an external coding-agent backend in a sandboxed workspace with inspectable artifacts.',
  },
  {
    title: 'Evaluator agent',
    description:
      'Judges both execution traces and generated artifacts so the system can recover from hidden methodological flaws, not only runtime crashes.',
  },
  {
    title: 'Reflector agent',
    description:
      'Synthesizes safeguards, workflows, dataset anchors, and code snippets from the full trajectory so later tasks can reuse governed experience.',
  },
]

export const benchmarkOverview = {
  intro:
    'HealthFlow is evaluated across five benchmarks spanning open-ended EHR workflow orchestration, executable artifact generation, deterministic medical reasoning, and tool-augmented biomedical decision-making.',
  benchmarkNames: ['EHRFlowBench', 'MedAgentBoard', 'MedAgentsBench', 'HLE', 'CureBench'],
  baselineNames: ['Alita', 'Biomni', 'STELLA', 'BioDSA', 'BioMedAgent'],
  ehrflowbench: {
    title: 'EHRFlowBench',
    subtitle: 'Open-ended EHR analysis benchmark derived from peer-reviewed literature',
    body:
      'EHRFlowBench is the core benchmark for full-cycle EHR analysis. It evaluates whether an agent can move from clinical objective to cohort construction, feature engineering, modeling, visualization, and report writing under realistic dataset constraints.',
    pipeline: [
      'Start from 51,280 papers published between 2020 and 2025 across major AI and data mining venues.',
      'Filter to 162 EHR-relevant candidates, then manually retain 118 seed papers for task extraction.',
      'Generate 236 dataset-grounded candidate tasks, paired across TJH and the MIMIC-IV Public Demo Release.',
      'Build the final 100-task benchmark as 50 TJH tasks and 50 MIMIC-IV tasks.',
    ],
  },
  medagentboard: {
    title: 'MedAgentBoard',
    subtitle: 'Workflow-grounded executable benchmark for clinical artifacts',
    body:
      'MedAgentBoard complements EHRFlowBench with a more deterministic executable task surface spanning data extraction, predictive modeling, and visualization. It is where artifact quality, chart validity, and structured outputs can be inspected directly.',
    highlights: [
      'Balanced over TJH and the MIMIC-IV Public Demo Release.',
      'Covers data processing, predictive modeling, and visualization tasks.',
      'Supports artifact-first review through images, tables, CSV files, markdown, and reports.',
    ],
  },
  additionalBenchmarks: [
    {
      title: 'MedAgentsBench',
      body: 'Deterministic hard-set multiple-choice benchmark for medical knowledge and reasoning.',
    },
    {
      title: 'HLE',
      body: 'Selected Biology and Medicine questions from Humanity’s Last Exam.',
    },
    {
      title: 'CureBench',
      body: 'Selected benchmark questions for biomedical tool-augmented decision-making and reasoning.',
    },
  ],
}

export const citation = `@misc{zhu2025healthflow,
  title={HealthFlow: Automating electronic health record analysis via a strategically self-evolving multi-agent framework},
  author={Yinghao Zhu and Zixiang Wang and Yifan Qi and Lei Gu and Dehao Sui and Haoran Hu and Xichen Zhang and Ziyi He and Yasha Wang and Junjun He and Liantao Ma and Lequan Yu},
  year={2025},
  note={Manuscript},
  url={https://github.com/yhzhu99/HealthFlow},
}`
