import { describe, expect, it } from 'vitest'

import type { SnapshotCandidate } from '../src/domain/evaluation'
import {
  blindOrderCandidates,
  createReviewerState,
  DEFAULT_REVIEWER_ID,
  evaluationStorageKey,
  resolveReviewerId,
} from '../src/domain/evaluation'

const buildCandidate = (modelId: string): SnapshotCandidate => ({
  id: modelId,
  runId: modelId,
  runLabel: modelId,
  modelId,
  answerText: modelId,
  success: true,
  artifacts: [],
})

describe('evaluation utils', () => {
  it('uses a stable reviewer-scoped blind ordering', () => {
    const candidates = ['alpha', 'beta', 'gamma'].map(buildCandidate)
    const first = blindOrderCandidates(candidates, 'reviewer-a', 'medagentboard', '7')
    const second = blindOrderCandidates(candidates, 'reviewer-a', 'medagentboard', '7')

    expect(first.map((item) => item.candidate.modelId)).toEqual(second.map((item) => item.candidate.modelId))
    expect(first.map((item) => item.slot)).toEqual(['A', 'B', 'C'])
  })

  it('changes candidate order across reviewers', () => {
    const candidates = ['alpha', 'beta', 'gamma'].map(buildCandidate)
    const reviewerA = blindOrderCandidates(candidates, 'reviewer-a', 'medagentboard', '7')
    const reviewerB = blindOrderCandidates(candidates, 'reviewer-b', 'medagentboard', '7')

    expect(reviewerA.map((item) => item.candidate.modelId)).not.toEqual(reviewerB.map((item) => item.candidate.modelId))
  })

  it('uses the reviewer id in local storage keys', () => {
    expect(evaluationStorageKey(' reviewer-7 ')).toBe('healthflow:evaluation:reviewer-7')
  })

  it('falls back to the demo reviewer when no reviewer id is provided', () => {
    expect(resolveReviewerId('')).toBe(DEFAULT_REVIEWER_ID)
    expect(resolveReviewerId(undefined)).toBe(DEFAULT_REVIEWER_ID)
    expect(resolveReviewerId(' reviewer-9 ')).toBe('reviewer-9')
  })

  it('starts reviewer state with benchmark and framework memory buckets', () => {
    expect(createReviewerState(' reviewer-2 ')).toEqual({
      reviewerId: 'reviewer-2',
      activeBenchmarkId: null,
      activeRunIdByDataset: {},
      currentIndexByDataset: {},
      responses: {},
    })
  })
})
