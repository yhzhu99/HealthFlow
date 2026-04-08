import { mkdtemp, mkdir, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { buildEvaluationSnapshotBundle, evaluationDataRootForProject } from '../dev/evaluation-data'

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
  it('builds a live dev payload from the committed evaluation-data tree', async () => {
    const bundle = await buildEvaluationSnapshotBundle({
      projectRoot: path.resolve(process.cwd()),
      mode: 'dev',
    })

    expect(bundle.payload.mode).toBe('live')
    if (bundle.payload.mode !== 'live') {
      throw new Error('Expected live payload')
    }

    expect(bundle.payload.snapshot.benchmarks.map((item) => item.id)).toEqual(['ehrflowbench', 'medagentboard'])
    expect(bundle.payload.snapshot.questions).toHaveLength(2)
    expect(bundle.payload.snapshot.questions[0]?.candidates.length).toBe(6)
    expect(bundle.payload.snapshot.questions[0]?.candidates[0]?.artifacts[0]?.relativePath).toMatch(/^\/__eval\/artifacts\//)
  })

  it('reports a diagnostic payload when the local evaluation root is missing', async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'healthflow-eval-'))
    tempRoots.push(tempRoot)

    const bundle = await buildEvaluationSnapshotBundle({
      projectRoot: tempRoot,
      mode: 'dev',
    })

    expect(bundle.payload.mode).toBe('diagnostic')
    if (bundle.payload.mode !== 'diagnostic') {
      throw new Error('Expected diagnostic payload')
    }

    expect(bundle.payload.diagnostics.root).toBe(evaluationDataRootForProject(tempRoot))
    expect(bundle.payload.diagnostics.missing.some((item) => item.includes('Missing evaluation root'))).toBe(true)
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
})
