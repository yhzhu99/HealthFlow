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

## Cloudflare Pages + R2

For production, deploy the frontend with Cloudflare Pages and expose the R2 bucket on the same origin under:

```text
/evaluation-data/
```

Upload the raw evaluation tree to R2 with the same structure shown above. Also upload a generated payload to:

```text
/evaluation-data/data/evaluation.payload.json
```

The production app loads that R2 payload first, then reads artifacts directly from paths such as:

```text
/evaluation-data/benchmarks/<benchmark-id>/cases/<case-id>/frameworks/<framework-id>/files/report.md
```

The app does not use the local `platform` project path in production URLs.

## Static Payload Export

When you need to generate the R2 payload from a local `evaluation-data` tree:

```bash
npm install
npm run build
```

The build emits `dist/data/evaluation.payload.json` and `dist/data/evaluation.snapshot.json` when local data exists. Upload the payload JSON to the R2 key `data/evaluation.payload.json`. Artifact files are not copied into `dist`; they are expected to be served from R2 under `/evaluation-data/`.

If Cloudflare Pages builds without a local `evaluation-data` directory, the build still succeeds and writes a diagnostic bundled payload. The browser will still prefer the R2 payload at `/evaluation-data/data/evaluation.payload.json`.

## Verification

```bash
npm test
npm run build
```
