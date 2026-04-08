# HealthFlow Platform

Two-page frontend for the HealthFlow project:

- `/` home page
- `/evaluation` evaluation workspace

The home page is the single paper-facing surface. It now includes the full project title, author and affiliation block, code and paper links, benchmark explanation, and citation instead of splitting those details across several routes.

## Demo Evaluation

The repo now ships a committed mock snapshot at `src/content/demo-evaluation.snapshot.json`, so the Evaluation page boots immediately in local development without requiring any generated public snapshot.

Current demo payload:

- 1 MedAgentBoard-style question
- 1 EHRFlowBench-style question
- 6 framework baselines per benchmark
- image, CSV, markdown, and PDF preview coverage
- default reviewer auto-loaded as `demo-reviewer`
- compact human-evaluation workspace with benchmark tabs plus framework tabs

## Real Snapshot Workflow

When real benchmark outputs are available, the same UI can still consume a generated snapshot:

```bash
npm install
npm run snapshot
npm run dev
```

The snapshot builder reads local processed benchmark files plus `benchmark_results/**/<qid>/` and writes a live override snapshot under `public/data/` together with frontend-ready files under `public/evaluation-assets/`.

## Verification

```bash
npm test
npm run build
```
