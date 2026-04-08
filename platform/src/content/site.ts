export const projectMeta = {
  name: 'HealthFlow',
  title: 'HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research',
  eyebrow: 'Paper + Platform',
  abstract: [
    'HealthFlow is a research framework for self-evolving task execution with a four-stage Meta -> Executor -> Evaluator -> Reflector loop.',
    'The core runtime is organized around planning, CodeAct-style execution, structured evaluation, per-task runtime artifacts, and long-term reflective memory.',
    'Dataset preparation and benchmark evaluation workflows can still live in the repository under data/, but they are intentionally decoupled from the healthflow/ runtime package.',
  ],
  authors: [
    'Yinghao Zhu',
    'Yifan Qi',
    'Zixiang Wang',
    'Lei Gu',
    'Dehao Sui',
    'Haoran Hu',
    'Xichen Zhang',
    'Ziyi He',
    'Junjun He',
    'Liantao Ma',
    'Lequan Yu',
  ],
  affiliations: ['Peking University', 'Zhejiang University', 'Tongji Medical College', 'Zhongnan Hospital of Wuhan University'],
  links: [
    {
      label: 'Paper',
      href: 'https://arxiv.org/abs/2508.02621',
      kind: 'primary' as const,
    },
    {
      label: 'Code',
      href: 'https://github.com/yhzhu99/HealthFlow',
      kind: 'secondary' as const,
    },
    {
      label: 'Evaluation',
      href: '/evaluation',
      kind: 'ghost' as const,
    },
    {
      label: 'Citation',
      href: '#citation',
      kind: 'ghost' as const,
    },
  ],
  facts: [
    { label: 'Paper', value: 'arXiv 2508.02621' },
    { label: 'Runtime', value: 'Meta -> Executor -> Evaluator -> Reflector' },
    { label: 'Benchmarks', value: '5 benchmark suites' },
    { label: 'Demo Surface', value: 'MedAgentBoard + EHRFlowBench' },
  ],
}

export const frameworkStages = [
  {
    title: 'Meta',
    description:
      'Retrieves safeguards, workflows, and dataset anchors, then emits a structured execution plan aligned with the current healthcare task family.',
  },
  {
    title: 'Executor',
    description:
      'Carries out the plan inside an inspectable workspace, preserving commands, outputs, intermediate files, and final deliverables for later review.',
  },
  {
    title: 'Evaluator',
    description:
      'Scores the latest attempt, classifies success or repair states, and provides concrete retry instructions instead of a single opaque scalar.',
  },
  {
    title: 'Reflector',
    description:
      'Synthesizes reusable safeguards, workflows, dataset anchors, and code snippets from the full trajectory after the task session ends.',
  },
]

export const benchmarkSuites = [
  {
    title: 'MedAgentBoard',
    category: 'Artifact-centric workflow benchmark',
    body: 'Deterministic healthcare workflows with visible figures, tables, and structured artifacts.',
  },
  {
    title: 'EHRFlowBench',
    category: 'Report-generation benchmark',
    body: 'Paper-inspired EHR studies rebuilt into local report-generation tasks for TJH and MIMIC-IV-demo.',
  },
  {
    title: 'MedAgentsBench',
    category: 'Multi-choice benchmark',
    body: '110 sampled questions spanning 10 medical QA sub-datasets.',
  },
  {
    title: 'HLE',
    category: 'Multi-choice benchmark',
    body: '110 Biology/Medicine multiple-choice questions rebuilt from the HLE test set.',
  },
  {
    title: 'CureBench',
    category: 'Multi-choice benchmark',
    body: '110 sampled multiple-choice questions filtered from the CureBench validation pool.',
  },
]

export const featuredBenchmarks = [
  {
    title: 'MedAgentBoard',
    eyebrow: 'Artifact-heavy review surface',
    body: 'MedAgentBoard focuses on outputs that need to be inspected directly: figures, tables, markdown summaries, and structured files. The platform keeps those artifacts visible so reviewers can judge the work itself instead of relying on a short final sentence.',
    bullets: [
      'Visualization, extraction, and structured-output tasks stay attached to their artifacts.',
      'Reviewers can inspect image and CSV evidence without leaving the page.',
      'Framework tabs make side-by-side baseline comparison a one-click action.',
    ],
  },
  {
    title: 'EHRFlowBench',
    eyebrow: 'Paper-derived local rebuild',
    body: 'EHRFlowBench turns paper-inspired EHR projects into repository-local report_generation tasks. The platform needs to explain both the benchmark intent and the rebuild process clearly, because the reference surface is a manifest-style local proxy rather than a copied paper leaderboard.',
    bullets: [
      'Generate intermediate task bundles from paper-derived prompts.',
      'Infer dataset requirements, then sample a balanced 55 TJH + 55 MIMIC-IV-demo subset with seed 42.',
      'Split the subset into 10 train tasks and 100 test tasks.',
      'Write processed JSONL files plus manifest-only reference answers for expected deliverables.',
    ],
  },
]

export const citationLinks = [
  {
    label: 'Paper',
    href: 'https://arxiv.org/abs/2508.02621',
    body: 'Open the latest public manuscript on arXiv.',
  },
  {
    label: 'Code',
    href: 'https://github.com/yhzhu99/HealthFlow',
    body: 'Inspect the runtime, benchmark workflows, and this platform in one repository.',
  },
]

export const citation = `@misc{zhu2025healthflow,
  title={HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research},
  author={Yinghao Zhu and Yifan Qi and Zixiang Wang and Lei Gu and Dehao Sui and Haoran Hu and Xichen Zhang and Ziyi He and Junjun He and Liantao Ma and Lequan Yu},
  year={2025},
  eprint={2508.02621},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2508.02621},
}`
