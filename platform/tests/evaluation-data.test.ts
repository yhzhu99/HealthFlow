import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import {
  buildEvaluationCasePayload,
  buildEvaluationManifestPayload,
  buildEvaluationSnapshotBundle,
  evaluationDataRootForProject,
  exportStaticEvaluationBundle,
} from '../dev/evaluation-data'

const tempRoots: string[] = []

afterEach(async () => {
  await Promise.allSettled(
    tempRoots.splice(0).map(async (target) => {
      const { rm } = await import('node:fs/promises')
      await rm(target, { recursive: true, force: true })
    }),
  )
})

describe('evaluation data bundle', () => {
  it('builds a live manifest payload from a local evaluation-data tree', async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'healthflow-live-'))
    tempRoots.push(tempRoot)

    const benchmarkRoot = path.join(tempRoot, 'evaluation-data', 'benchmarks', 'medagentboard', 'cases', '0001')
    await mkdir(path.join(benchmarkRoot, 'reference', 'files'), { recursive: true })
    await mkdir(path.join(benchmarkRoot, 'frameworks', 'healthflow', 'files'), { recursive: true })
    await writeFile(
      path.join(tempRoot, 'evaluation-data', 'benchmarks', 'medagentboard', 'benchmark.json'),
      JSON.stringify({
        id: 'medagentboard',
        label: 'MedAgentBoard',
        description: 'Local benchmark',
        frameworkOrder: ['healthflow'],
      }),
    )
    await writeFile(
      path.join(benchmarkRoot, 'case.json'),
      JSON.stringify({
        id: 'medagentboard:1',
        qid: '1',
        task: 'Review this local case.',
        taskBrief: 'Local smoke test',
        expectedOutputs: [],
      }),
    )
    await writeFile(path.join(benchmarkRoot, 'reference', 'answer.md'), 'Reference answer')
    await writeFile(path.join(benchmarkRoot, 'reference', 'files', 'reference.md'), '# Reference')
    await writeFile(
      path.join(benchmarkRoot, 'frameworks', 'healthflow', 'manifest.json'),
      JSON.stringify({
        runId: 'medagentboard-healthflow',
        runLabel: 'HealthFlow',
        modelId: 'healthflow',
        summary: 'Strong answer',
      }),
    )
    await writeFile(path.join(benchmarkRoot, 'frameworks', 'healthflow', 'answer.md'), 'HealthFlow answer')
    await writeFile(path.join(benchmarkRoot, 'frameworks', 'healthflow', 'files', 'report.md'), '# Report')

    const payload = await buildEvaluationManifestPayload({
      projectRoot: tempRoot,
    })

    expect(payload.mode).toBe('live')
    if (payload.mode !== 'live') {
      throw new Error('Expected live payload')
    }

    expect(payload.manifest.benchmarks.map((item) => item.id)).toEqual(['medagentboard'])
    expect(payload.manifest.questions).toHaveLength(1)
    expect(payload.manifest.questions[0]?.candidateCount).toBe(1)
    expect(payload.manifest.runs[0]?.id).toBe('medagentboard-healthflow')
  })

  it('builds an on-demand case payload with dev artifact urls', async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'healthflow-case-'))
    tempRoots.push(tempRoot)

    const benchmarkRoot = path.join(tempRoot, 'evaluation-data', 'benchmarks', 'medagentboard', 'cases', '0001')
    await mkdir(path.join(benchmarkRoot, 'reference', 'files'), { recursive: true })
    await mkdir(path.join(benchmarkRoot, 'frameworks', 'healthflow', 'files'), { recursive: true })
    await writeFile(
      path.join(tempRoot, 'evaluation-data', 'benchmarks', 'medagentboard', 'benchmark.json'),
      JSON.stringify({
        id: 'medagentboard',
        label: 'MedAgentBoard',
        description: 'Local benchmark',
        frameworkOrder: ['healthflow'],
      }),
    )
    await writeFile(
      path.join(benchmarkRoot, 'case.json'),
      JSON.stringify({
        id: 'medagentboard:1',
        qid: '1',
        task: 'Review this local case.',
        taskBrief: 'Local smoke test',
        expectedOutputs: [],
      }),
    )
    await writeFile(path.join(benchmarkRoot, 'reference', 'answer.md'), 'Reference answer')
    await writeFile(path.join(benchmarkRoot, 'reference', 'files', 'reference.md'), '# Reference')
    await writeFile(
      path.join(benchmarkRoot, 'frameworks', 'healthflow', 'manifest.json'),
      JSON.stringify({
        runId: 'medagentboard-healthflow',
        runLabel: 'HealthFlow',
        modelId: 'healthflow',
        summary: 'Strong answer',
      }),
    )
    await writeFile(path.join(benchmarkRoot, 'frameworks', 'healthflow', 'answer.md'), 'HealthFlow answer')
    await writeFile(path.join(benchmarkRoot, 'frameworks', 'healthflow', 'files', 'report.md'), '# Report')

    const payload = await buildEvaluationCasePayload({
      projectRoot: tempRoot,
      benchmarkId: 'medagentboard',
      caseId: '0001',
      mode: 'dev',
    })

    expect(payload.question.id).toBe('medagentboard:1')
    expect(payload.question.candidates).toHaveLength(1)
    expect(payload.question.candidates[0]?.artifacts[0]?.relativePath).toMatch(/^\/__eval\/artifacts\//)
    expect(payload.question.candidates[0]?.reportPath).toMatch(/^\/__eval\/artifacts\//)
  })

  it('reports a diagnostic payload when the local evaluation root is missing', async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'healthflow-eval-'))
    tempRoots.push(tempRoot)

    const payload = await buildEvaluationManifestPayload({
      projectRoot: tempRoot,
    })

    expect(payload.mode).toBe('diagnostic')
    if (payload.mode !== 'diagnostic') {
      throw new Error('Expected diagnostic payload')
    }

    expect(payload.diagnostics.root).toBe(evaluationDataRootForProject(tempRoot))
    expect(payload.diagnostics.missing.some((item) => item.includes('Missing evaluation root'))).toBe(true)
  })

  it('uses static export paths when building a static bundle', async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'healthflow-static-'))
    tempRoots.push(tempRoot)

    const benchmarkRoot = path.join(tempRoot, 'evaluation-data', 'benchmarks', 'demo', 'cases', '0001')
    await mkdir(path.join(benchmarkRoot, 'reference', 'files'), { recursive: true })
    await mkdir(path.join(benchmarkRoot, 'frameworks', 'alpha', 'files'), { recursive: true })
    await writeFile(
      path.join(tempRoot, 'evaluation-data', 'benchmarks', 'demo', 'benchmark.json'),
      JSON.stringify({
        id: 'demo',
        label: 'Demo',
        description: 'Demo benchmark',
        frameworkOrder: ['alpha'],
      }),
    )
    await writeFile(
      path.join(benchmarkRoot, 'case.json'),
      JSON.stringify({
        id: 'demo:1',
        qid: '1',
        task: 'Example task',
        expectedOutputs: [],
      }),
    )
    await writeFile(
      path.join(benchmarkRoot, 'frameworks', 'alpha', 'manifest.json'),
      JSON.stringify({
        runId: 'demo-alpha',
        runLabel: 'Alpha',
        modelId: 'alpha',
      }),
    )
    await writeFile(path.join(benchmarkRoot, 'frameworks', 'alpha', 'answer.md'), 'Alpha answer')
    await writeFile(path.join(benchmarkRoot, 'frameworks', 'alpha', 'files', 'report.md'), '# Report')

    const bundle = await buildEvaluationSnapshotBundle({
      projectRoot: tempRoot,
      mode: 'static',
    })

    expect(bundle.payload.mode).toBe('live')
    if (bundle.payload.mode !== 'live') {
      throw new Error('Expected live payload')
    }

    expect(bundle.payload.snapshot.questions[0]?.candidates[0]?.artifacts[0]?.relativePath).toBe(
      'evaluation-assets/demo/0001/alpha/report.md',
    )
    expect(bundle.artifactCopies[0]?.relativePath).toBe('evaluation-assets/demo/0001/alpha/report.md')
  })

  it('exports a static evaluation bundle into a build output directory', async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'healthflow-export-'))
    tempRoots.push(tempRoot)

    const benchmarkRoot = path.join(tempRoot, 'evaluation-data', 'benchmarks', 'demo', 'cases', '0001')
    await mkdir(path.join(benchmarkRoot, 'reference', 'files'), { recursive: true })
    await mkdir(path.join(benchmarkRoot, 'frameworks', 'alpha', 'files'), { recursive: true })
    await writeFile(
      path.join(tempRoot, 'evaluation-data', 'benchmarks', 'demo', 'benchmark.json'),
      JSON.stringify({
        id: 'demo',
        label: 'Demo',
        description: 'Demo benchmark',
        frameworkOrder: ['alpha'],
      }),
    )
    await writeFile(
      path.join(benchmarkRoot, 'case.json'),
      JSON.stringify({
        id: 'demo:1',
        qid: '1',
        task: 'Example task',
        expectedOutputs: [],
      }),
    )
    await writeFile(
      path.join(benchmarkRoot, 'frameworks', 'alpha', 'manifest.json'),
      JSON.stringify({
        runId: 'demo-alpha',
        runLabel: 'Alpha',
        modelId: 'alpha',
      }),
    )
    await writeFile(path.join(benchmarkRoot, 'frameworks', 'alpha', 'answer.md'), 'Alpha answer')
    await writeFile(path.join(benchmarkRoot, 'frameworks', 'alpha', 'files', 'report.md'), '# Report')

    const outputRoot = path.join(tempRoot, 'dist')
    const result = await exportStaticEvaluationBundle({
      projectRoot: tempRoot,
      outputRoot,
    })

    expect(result.mode).toBe('live')
    expect(result.outputPath).toBe(path.join(outputRoot, 'data', 'evaluation.snapshot.json'))
    expect(result.payloadOutputPath).toBe(path.join(outputRoot, 'data', 'evaluation.payload.json'))
    expect(result.snapshot?.questions).toHaveLength(1)
    expect(result.artifactCount).toBe(1)
    expect(await readFile(path.join(outputRoot, 'evaluation-assets', 'demo', '0001', 'alpha', 'report.md'), 'utf-8')).toBe(
      '# Report',
    )
  })

  it('exports a diagnostic payload instead of failing when evaluation data is missing', async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'healthflow-export-diagnostic-'))
    tempRoots.push(tempRoot)

    const outputRoot = path.join(tempRoot, 'dist')
    const result = await exportStaticEvaluationBundle({
      projectRoot: tempRoot,
      outputRoot,
    })

    expect(result.mode).toBe('diagnostic')
    expect(result.outputPath).toBe(path.join(outputRoot, 'data', 'evaluation.payload.json'))
    expect(result.snapshot).toBeNull()
    expect(result.artifactCount).toBe(0)

    const payload = JSON.parse(await readFile(path.join(outputRoot, 'data', 'evaluation.payload.json'), 'utf-8')) as {
      mode: string
      diagnostics: { missing: string[] }
    }

    expect(payload.mode).toBe('diagnostic')
    expect(payload.diagnostics.missing.some((item) => item.includes('Missing evaluation root'))).toBe(true)
  })
})
