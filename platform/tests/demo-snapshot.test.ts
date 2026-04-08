import { describe, expect, it } from 'vitest'

import { demoEvaluationSnapshot } from '../src/content/demo-evaluation'

describe('demo snapshot', () => {
  it('ships a committed mock snapshot with two demo questions and six framework baselines per benchmark', () => {
    expect(demoEvaluationSnapshot.snapshotVersion).toBe('healthflow-demo-v4')
    expect(demoEvaluationSnapshot.questions).toHaveLength(2)
    expect(demoEvaluationSnapshot.runs).toHaveLength(12)
    expect(demoEvaluationSnapshot.benchmarks.map((item) => item.id)).toEqual(['medagentboard', 'ehrflowbench'])
    expect(demoEvaluationSnapshot.questions.map((item) => item.datasetId)).toEqual([
      'medagentboard',
      'ehrflowbench',
    ])
    expect(demoEvaluationSnapshot.questions.map((item) => item.candidates.length)).toEqual([6, 6])
    expect(demoEvaluationSnapshot.questions.flatMap((item) => item.candidates).every((candidate) => candidate.artifacts.length === 0)).toBe(true)
  })
})
