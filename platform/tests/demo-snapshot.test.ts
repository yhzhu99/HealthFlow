import { readFileSync } from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

describe('demo snapshot', () => {
  it('ships a committed mock snapshot with two demo questions and six framework baselines per benchmark', () => {
    const snapshotPath = path.resolve(process.cwd(), 'public/data/evaluation.snapshot.json')
    const snapshot = JSON.parse(readFileSync(snapshotPath, 'utf-8'))

    expect(snapshot.snapshotVersion).toBe('healthflow-demo-v3')
    expect(snapshot.questions).toHaveLength(2)
    expect(snapshot.runs).toHaveLength(12)
    expect(snapshot.benchmarks.map((item: { id: string }) => item.id)).toEqual(['medagentboard', 'ehrflowbench'])
    expect(snapshot.questions.map((item: { datasetId: string }) => item.datasetId)).toEqual([
      'medagentboard',
      'ehrflowbench',
    ])
    expect(snapshot.questions.map((item: { candidates: unknown[] }) => item.candidates.length)).toEqual([6, 6])
  })
})
