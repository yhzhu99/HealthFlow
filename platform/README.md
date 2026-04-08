# HealthFlow Platform

Unified frontend for the HealthFlow paper narrative and blind human evaluation workflow.

## Stack

- Vue 3
- TypeScript
- Vite
- Tailwind CSS v4

## Routes

- `/` overview / project homepage
- `/benchmarks` benchmark definitions and reviewer protocol
- `/evaluation` blind reviewer workspace
- `/resources` commands, links, and citation

## Snapshot Workflow

The evaluation UI reads a static snapshot at `public/data/evaluation.snapshot.json`.

1. Edit [`content/evaluation.config.json`](/home/yhzhu/projects/HealthFlow/platform/content/evaluation.config.json) so it points at the local processed benchmark files and result directories you want to review.
2. Run:

```bash
npm install
npm run snapshot
npm run dev
```

The snapshot builder:

- loads selected qids from local benchmark JSONL files
- resolves reference manifests / reference artifacts when available
- copies previewable files from `benchmark_results/**/<qid>/`
- writes `public/data/evaluation.snapshot.json`
- writes copied assets under `public/evaluation-assets/`

If no snapshot exists yet, the Evaluation route stays in a non-crashing empty state and shows the build command.

## Verification

```bash
npm test
npm run build
```
