# HealthFlow Platform

Two-page frontend for the HealthFlow project:

- `/` home page
- `/evaluation` evaluation workspace

The home page is the manuscript-facing surface. The evaluation workspace is a blinded local review tool backed by `platform/evaluation-data`.

## Local Dev Evaluation

The committed source of truth for evaluation content is:

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

## Static Export

When you want a fast local deployment, build from the same `evaluation-data` tree:

```bash
npm install
npm run build
```

The build now emits the evaluation bundle directly into `dist/data/` and `dist/evaluation-assets/`. No committed `public/data/` or `public/evaluation-assets/` mirror is required.

## Verification

```bash
npm test
npm run build
```
