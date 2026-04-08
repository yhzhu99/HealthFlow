import { describe, expect, it } from 'vitest'

import { sampleEvaluationSnapshot } from '../src/content/sample-evaluation'

describe('sample snapshot', () => {
  it('ships a committed sample snapshot with two bundled cases and six blinded submissions per benchmark', () => {
    expect(sampleEvaluationSnapshot.snapshotVersion).toBe('healthflow-sample-v1')
    expect(sampleEvaluationSnapshot.questions).toHaveLength(2)
    expect(sampleEvaluationSnapshot.runs).toHaveLength(12)
    expect(sampleEvaluationSnapshot.benchmarks.map((item) => item.id)).toEqual(['medagentboard', 'ehrflowbench'])
    expect(sampleEvaluationSnapshot.questions.map((item) => item.datasetId)).toEqual([
      'medagentboard',
      'ehrflowbench',
    ])
    expect(sampleEvaluationSnapshot.questions.map((item) => item.candidates.length)).toEqual([6, 6])
    expect(sampleEvaluationSnapshot.questions.flatMap((item) => item.candidates).every((candidate) => candidate.artifacts.length === 0)).toBe(true)
  })
})
