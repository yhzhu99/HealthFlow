export const projectMeta = {
  name: 'HealthFlow',
  title: 'HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research',
  subtitle:
    'A unified project page and evaluation surface for artifact-centric healthcare agent research across MedAgentBoard and EHRFlowBench.',
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
      label: 'Read Paper',
      href: 'https://arxiv.org/abs/2508.02621',
      kind: 'primary' as const,
    },
    {
      label: 'View Code',
      href: 'https://github.com/yhzhu99/HealthFlow',
      kind: 'secondary' as const,
    },
    {
      label: 'Open Evaluation',
      href: '/evaluation',
      kind: 'ghost' as const,
    },
  ],
  facts: [
    { label: 'Paper', value: 'arXiv 2508.02621' },
    { label: 'Runtime', value: 'Meta -> Executor -> Evaluator -> Reflector' },
    { label: 'Benchmarks', value: 'MedAgentBoard + EHRFlowBench' },
  ],
}

export const heroSignals = [
  {
    title: 'Artifact-native review',
    body: 'Plots, reports, CSVs, PDFs, and runtime deliverables remain visible after the run, so reviewers inspect evidence rather than only summaries.',
  },
  {
    title: 'Two benchmark families',
    body: 'MedAgentBoard emphasizes generated visual and structured artifacts, while EHRFlowBench emphasizes report-centric clinical analysis.',
  },
  {
    title: 'One calm interface',
    body: 'The platform stays intentionally small: one paper-facing project page and one evaluation workspace with dataset and framework tabs.',
  },
]

export const abstractParagraphs = [
  'HealthFlow is a self-evolving AI agent framework for autonomous healthcare research. Instead of treating execution as a one-shot chain, it keeps a narrow loop around planning, workspace execution, evaluation, and reflection so later tasks can benefit from earlier trajectories.',
  'The platform here is designed as the paper-facing surface for the project. It brings together the framework narrative, benchmark setup, code and paper links, citation metadata, and a direct path into artifact review without splitting the story across several disconnected mini-sites.',
]

export const frameworkStages = [
  {
    title: 'Meta agent',
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

export const benchmarkCards = [
  {
    title: 'MedAgentBoard',
    eyebrow: 'Artifact-heavy workflow benchmark',
    body: 'Task quality is tied to the produced image, table, and structured outputs. Reviewers should be able to render figures directly and inspect lightweight tabular evidence without leaving the page.',
    bullets: ['Image rendering matters', 'CSV / JSON / markdown outputs matter', 'Frameworks can be compared side by side through fast tabs'],
  },
  {
    title: 'EHRFlowBench',
    eyebrow: 'Report-centric benchmark',
    body: 'Paper-derived EHR projects are rebuilt into report-generation tasks. The key deliverable is a readable markdown report, with PDF preview when the export exists in the run artifacts.',
    bullets: ['Markdown reports are first-class', 'PDF rendering is supported in the workspace', 'Manifest-style expectations are supported when no gold report is committed'],
  },
]

export const resultHighlights = [
  {
    title: 'Lean runtime boundary',
    body: 'Dataset preparation and benchmark-side evaluation stay separate from the core runtime, which keeps HealthFlow easier to reason about and audit.',
  },
  {
    title: 'Inspectable evidence',
    body: 'Each task leaves behind a short-paper-style report plus the concrete artifacts needed for rebuttal, benchmarking, and human review.',
  },
  {
    title: 'Framework comparison',
    body: 'The evaluation workspace can group multiple framework outputs under each benchmark so switching between baselines is a single click rather than a hidden config change.',
  },
]

export const resourceLinks = [
  {
    title: 'Paper',
    href: 'https://arxiv.org/abs/2508.02621',
    label: 'HealthFlow on arXiv',
    body: 'Full paper abstract, metadata, citation record, and the latest public manuscript.',
  },
  {
    title: 'Code',
    href: 'https://github.com/yhzhu99/HealthFlow',
    label: 'GitHub Repository',
    body: 'Runtime code, data workflows, benchmark assets, and the platform implementation live in the same repository.',
  },
  {
    title: 'Evaluation',
    href: '/evaluation',
    label: 'Artifact Review Workspace',
    body: 'Jump directly into the evaluation UI with simulated benchmark data and fast framework switching.',
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
