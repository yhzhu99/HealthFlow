# HealthFlow Platform

Two-page frontend for the HealthFlow project:

- `/` home page
- `/evaluation` evaluation workspace

The home page is the single paper-facing surface. It now includes the full project title, author and affiliation block, code and paper links, benchmark explanation, and citation instead of splitting those details across several routes.

## Demo Evaluation

The repo now ships a tiny embedded text-only demo in source code, so the Evaluation page still has a safe fallback UI when local evaluation data are missing.

Current demo payload:

- 1 MedAgentBoard-style question
- 1 EHRFlowBench-style question
- 6 framework baselines per benchmark
- text-only comparison coverage
- default reviewer auto-loaded as `demo-reviewer`
- compact human-evaluation workspace with benchmark tabs plus framework tabs

## Local Dev Evaluation

The real dev workflow no longer depends on a committed `snapshot.json`.

Place your private evaluation data under:

```text
platform/evaluation-data/
  benchmarks/
    <benchmark-id>/
      benchmark.json
      cases/
        <case-id>/
          case.json
          reference/
            answer.md
            files/
          frameworks/
            <framework-id>/
              manifest.json
              answer.md
              files/
```

The Vite dev server reads this tree through `GET /__eval/manifest` and serves local artifacts through `GET /__eval/artifacts/*`.

These local evaluation directories and generated review artifacts are ignored by git on purpose.

## Static Export

When you do want a static export, build it from the local evaluation tree:

```bash
npm install
npm run snapshot
npm run dev
```

The snapshot builder now reads `platform/evaluation-data/` and writes an export snapshot under `public/data/` together with generated files under `public/evaluation-assets/`.

## Verification

```bash
npm test
npm run build
```
