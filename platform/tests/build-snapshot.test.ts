import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { buildSnapshot } from '../scripts/build-snapshot'

const tempRoots: string[] = []

afterEach(async () => {
  await Promise.allSettled(tempRoots.splice(0).map((target) => rm(target, { recursive: true, force: true })))
})

const writeJson = async (filePath: string, value: unknown) => {
  await mkdir(path.dirname(filePath), { recursive: true })
  await writeFile(filePath, JSON.stringify(value, null, 2))
}

describe('build snapshot', () => {
  it('exports a static snapshot from platform/evaluation-data', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'healthflow-platform-static-'))
    tempRoots.push(root)
    const benchmarkRoot = path.join(root, 'evaluation-data', 'benchmarks', 'demo', 'cases', '0001')

    await mkdir(path.join(benchmarkRoot, 'reference', 'files'), { recursive: true })
    await mkdir(path.join(benchmarkRoot, 'frameworks', 'alpha', 'files'), { recursive: true })

    await writeJson(path.join(root, 'evaluation-data', 'benchmarks', 'demo', 'benchmark.json'), {
      id: 'demo',
      label: 'Demo',
      description: 'Demo benchmark',
      frameworkOrder: ['alpha'],
    })
    await writeJson(path.join(benchmarkRoot, 'case.json'), {
      id: 'demo:1',
      qid: '1',
      task: 'Example task',
      taskBrief: 'Smoke test',
      expectedOutputs: [{ fileName: 'report.md', mediaType: 'text/markdown', referencePath: null }],
    })
    await writeFile(path.join(benchmarkRoot, 'reference', 'answer.md'), 'Reference answer')
    await writeFile(path.join(benchmarkRoot, 'reference', 'files', 'reference.md'), '# Reference')
    await writeJson(path.join(benchmarkRoot, 'frameworks', 'alpha', 'manifest.json'), {
      runId: 'demo-alpha',
      runLabel: 'Alpha',
      modelId: 'alpha',
      backend: 'codex',
      summary: 'Alpha summary',
      success: true,
    })
    await writeFile(path.join(benchmarkRoot, 'frameworks', 'alpha', 'answer.md'), 'Alpha answer')
    await writeFile(path.join(benchmarkRoot, 'frameworks', 'alpha', 'files', 'report.md'), '# Report')

    const { outputPath, snapshot, artifactCount } = await buildSnapshot({ projectRoot: root })
    const writtenSnapshot = JSON.parse(await readFile(outputPath, 'utf-8'))
    const exportedArtifact = await readFile(path.join(root, 'public', 'evaluation-assets', 'demo', '0001', 'alpha', 'report.md'), 'utf-8')

    expect(snapshot.questions).toHaveLength(1)
    expect(snapshot.questions[0]?.reference.mode).toBe('artifacts')
    expect(writtenSnapshot.questions[0].candidates[0].artifacts[0].relativePath).toBe(
      'evaluation-assets/demo/0001/alpha/report.md',
    )
    expect(exportedArtifact).toBe('# Report')
    expect(artifactCount).toBe(2)
  })

  it('fails with a diagnostic error when evaluation-data is missing', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'healthflow-platform-missing-'))
    tempRoots.push(root)

    await expect(buildSnapshot({ projectRoot: root })).rejects.toThrow(
      /Unable to export evaluation snapshot from platform\/evaluation-data/,
    )
  })
})
