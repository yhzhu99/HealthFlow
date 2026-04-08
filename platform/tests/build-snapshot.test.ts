import { mkdtempSync, writeFileSync } from 'node:fs'
import { mkdir, readFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { buildSnapshot } from '../scripts/build-snapshot'

const writeJson = (filePath: string, value: unknown) => {
  writeFileSync(filePath, JSON.stringify(value, null, 2) + '\n', 'utf-8')
}

describe('build snapshot', () => {
  it('builds a static snapshot and copies previewable artifacts', async () => {
    const root = mkdtempSync(path.join(os.tmpdir(), 'healthflow-platform-'))
    const projectRoot = path.join(root, 'platform')
    const contentRoot = path.join(projectRoot, 'content')
    const medProcessed = path.join(root, 'data', 'medagentboard', 'processed')
    const ehrProcessed = path.join(root, 'data', 'ehrflowbench', 'processed')
    const medRunRoot = path.join(root, 'benchmark_results', 'medagentboard', 'run-a')
    const ehrRunRoot = path.join(root, 'benchmark_results', 'ehrflowbench', 'run-b')

    await mkdir(contentRoot, { recursive: true })
    await mkdir(path.join(medProcessed, 'reference_answers', 'test', '1'), { recursive: true })
    await mkdir(path.join(ehrProcessed, 'reference_answers', 'test', '2'), { recursive: true })
    await mkdir(path.join(medRunRoot, '1', 'runtime'), { recursive: true })
    await mkdir(path.join(medRunRoot, '1', 'sandbox', 'tables'), { recursive: true })
    await mkdir(path.join(ehrRunRoot, '2', 'sandbox'), { recursive: true })

    writeFileSync(
      path.join(medProcessed, 'test.jsonl'),
      JSON.stringify({
        qid: 1,
        task: 'Analyze the local EHR slice and produce a report.',
        task_brief: 'MedAgentBoard smoke task',
        task_type: 'visualization',
        reference_answer: 'reference_answers/test/1/answer_manifest.json',
      }) + '\n',
      'utf-8',
    )
    writeFileSync(
      path.join(ehrProcessed, 'test.jsonl'),
      JSON.stringify({
        qid: 2,
        task: 'Write a full report backed by local EHR data.',
        task_brief: 'EHRFlowBench smoke task',
        task_type: 'report_generation',
        report_requirements: ['Introduction', 'Method', 'Results', 'Conclusion'],
        reference_answer: 'reference_answers/test/2/answer_manifest.json',
      }) + '\n',
      'utf-8',
    )

    writeJson(path.join(medProcessed, 'reference_answers', 'test', '1', 'answer_manifest.json'), {
      contract_version: 'medagentboard_llm_v2',
      required_outputs: [
        {
          file_name: 'reference.md',
          reference_path: 'reference_answers/test/1/reference.md',
          media_type: 'text/markdown',
        },
      ],
    })
    writeFileSync(path.join(medProcessed, 'reference_answers', 'test', '1', 'reference.md'), '# Reference\n\nMedAgentBoard answer.\n', 'utf-8')

    writeJson(path.join(ehrProcessed, 'reference_answers', 'test', '2', 'answer_manifest.json'), {
      contract_version: 'ehrflowbench_v1',
      required_outputs: [
        { file_name: 'report.md', media_type: 'text/markdown' },
        { file_name: 'metrics.json', media_type: 'application/json' },
      ],
    })

    writeJson(path.join(medRunRoot, '1', 'result.json'), {
      generated_answer: 'Generated answer for MedAgentBoard.',
      final_summary: 'MedAgentBoard summary.',
      success: true,
      score: 8.4,
      backend: 'opencode',
    })
    writeFileSync(path.join(medRunRoot, '1', 'runtime', 'report.md'), '# Runtime Report\n\nA runtime report.\n', 'utf-8')
    writeFileSync(path.join(medRunRoot, '1', 'sandbox', 'tables', 'metrics.csv'), 'metric,value\nauroc,0.81\n', 'utf-8')

    writeJson(path.join(ehrRunRoot, '2', 'result.json'), {
      generated_answer: 'Generated answer for EHRFlowBench.',
      final_summary: 'EHRFlowBench summary.',
      success: true,
      score: 9.1,
      backend: 'codex',
    })
    writeFileSync(path.join(ehrRunRoot, '2', 'sandbox', 'report.md'), '# Report\n\nA paper-style report.\n', 'utf-8')

    const configPath = path.join(contentRoot, 'evaluation.config.json')
    writeJson(configPath, {
      datasets: [
        {
          id: 'medagentboard',
          label: 'MedAgentBoard',
          description: 'MedAgentBoard dataset',
          taskSource: '../../data/medagentboard/processed/test.jsonl',
          referenceRoot: '../../data/medagentboard/processed',
          selectedQids: [1],
          artifactPreferences: ['report.md', '.csv'],
          runs: [
            {
              id: 'run-a',
              label: 'Run A',
              modelId: 'model-a',
              sourceResultsDir: '../../benchmark_results/medagentboard/run-a',
            },
          ],
        },
        {
          id: 'ehrflowbench',
          label: 'EHRFlowBench',
          description: 'EHRFlowBench dataset',
          taskSource: '../../data/ehrflowbench/processed/test.jsonl',
          referenceRoot: '../../data/ehrflowbench/processed',
          selectedQids: [2],
          artifactPreferences: ['report.md'],
          runs: [
            {
              id: 'run-b',
              label: 'Run B',
              modelId: 'model-b',
              sourceResultsDir: '../../benchmark_results/ehrflowbench/run-b',
            },
          ],
        },
      ],
    })

    const { outputPath, snapshot } = await buildSnapshot({ projectRoot, configPath })
    const writtenSnapshot = JSON.parse(await readFile(outputPath, 'utf-8'))

    expect(snapshot.questions).toHaveLength(2)
    expect(writtenSnapshot.questions[0].candidates[0].artifacts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: 'report.md' }),
        expect.objectContaining({ label: 'metrics.csv' }),
      ]),
    )
    expect(writtenSnapshot.questions[1].reference.mode).toBe('manifest')
    expect(writtenSnapshot.questions[1].reference.requiredOutputs).toEqual(
      expect.arrayContaining([expect.objectContaining({ fileName: 'report.md' })]),
    )
  })

  it('fails when a configured qid is missing from the task source', async () => {
    const root = mkdtempSync(path.join(os.tmpdir(), 'healthflow-platform-error-'))
    const projectRoot = path.join(root, 'platform')
    const contentRoot = path.join(projectRoot, 'content')
    const processedRoot = path.join(root, 'data', 'medagentboard', 'processed')

    await mkdir(contentRoot, { recursive: true })
    await mkdir(processedRoot, { recursive: true })
    writeFileSync(path.join(processedRoot, 'test.jsonl'), JSON.stringify({ qid: 1, task: 'Only one task', answer: 'A' }) + '\n', 'utf-8')

    const configPath = path.join(contentRoot, 'evaluation.config.json')
    writeJson(configPath, {
      datasets: [
        {
          id: 'medagentboard',
          label: 'MedAgentBoard',
          description: 'MedAgentBoard dataset',
          taskSource: '../../data/medagentboard/processed/test.jsonl',
          referenceRoot: '../../data/medagentboard/processed',
          selectedQids: [7],
          runs: [
            {
              id: 'run-a',
              label: 'Run A',
              modelId: 'model-a',
              sourceResultsDir: '../../benchmark_results/medagentboard/run-a',
            },
          ],
        },
      ],
    })

    await expect(buildSnapshot({ projectRoot, configPath })).rejects.toThrow('missing qid 7')
  })
})
