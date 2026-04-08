export const overviewStats = [
  { label: 'Execution loop', value: 'Meta -> Executor -> Evaluator -> Reflector' },
  { label: 'Workspace artifacts', value: 'Structured runtime logs, reports, and deliverables' },
  { label: 'Benchmarks', value: 'MedAgentBoard + EHRFlowBench in one review surface' },
  { label: 'Review mode', value: 'Blind pairwise preference with local export' },
]

export const frameworkStages = [
  {
    title: 'Meta agent',
    description:
      'Retrieves safeguards, workflows, and dataset anchors, then emits a structured plan tuned for the current task family.',
  },
  {
    title: 'Executor',
    description:
      'Runs CodeAct-style work inside an inspectable workspace, preserving commands, outputs, artifacts, and intermediate files.',
  },
  {
    title: 'Evaluator',
    description:
      'Scores the latest attempt, classifies success or repair states, and produces actionable retry instructions instead of a bare scalar score.',
  },
  {
    title: 'Reflector',
    description:
      'Synthesizes reusable safeguards, workflows, and code snippets from the full trajectory after the task session ends.',
  },
]

export const resultHighlights = [
  {
    title: 'Benchmark-facing runtime',
    body: 'HealthFlow separates the runtime loop from dataset preparation and benchmark-side evaluation, so experimentation remains reproducible and auditable.',
  },
  {
    title: 'Inspectable artifacts',
    body: 'Each task writes a short-paper-style runtime report, structured JSON summaries, and concrete deliverables that can be reviewed outside the CLI.',
  },
  {
    title: 'Role-specific models',
    body: 'Planning, execution, evaluation, and reflection can use different model selections to reduce single-model coupling in experiments.',
  },
]

export const benchmarkCards = [
  {
    id: 'medagentboard',
    title: 'MedAgentBoard',
    eyebrow: 'Artifact-heavy workflow benchmark',
    body: 'Deterministic workflow tasks grounded in local TJH and MIMIC demo data. Outputs often include figures, tables, JSON summaries, and other directly previewable artifacts.',
    bullets: ['Strong emphasis on generated artifacts', 'Image and table rendering matter', 'Blind reviewer compares end products, not model names'],
  },
  {
    id: 'ehrflowbench',
    title: 'EHRFlowBench',
    eyebrow: 'Paper-derived report generation benchmark',
    body: 'Self-contained report-generation tasks rebuilt from paper ideas. Expected outputs are manifest-driven and usually culminate in `report.md` plus analytical evidence.',
    bullets: ['Markdown-first report review', 'PDF preview when a rendered export exists', 'Manifest-only references supported when no gold report is committed'],
  },
]

export const resourceCards = [
  {
    title: 'Paper',
    href: 'https://arxiv.org/abs/2508.02621',
    label: 'arXiv 2508.02621',
    body: 'HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research.',
  },
  {
    title: 'Runtime CLI',
    href: 'https://arxiv.org/abs/2508.02621',
    label: 'HealthFlow run / web',
    body: 'The core runtime ships a non-interactive CLI, interactive CLI, and a browser-based task session interface.',
  },
  {
    title: 'Snapshot builder',
    href: '/evaluation',
    label: 'npm run snapshot',
    body: 'Transforms local benchmark results into a static evaluation snapshot and copies previewable artifacts for the frontend.',
  },
]

export const citation = `@misc{zhu2025healthflow,
  title={HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research},
  author={Yinghao Zhu and Yifan Qi and Zixiang Wang and Lei Gu and Dehao Sui and Haoran Hu and Xichen Zhang and Ziyi He and and Junjun He and Liantao Ma and Lequan Yu},
  year={2025},
  eprint={2508.02621},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2508.02621},
}`
