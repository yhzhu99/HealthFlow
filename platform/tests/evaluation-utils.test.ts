import { describe, expect, it } from 'vitest'

import type { EvaluationSessionState, SnapshotCandidate } from '../src/domain/evaluation'
import {
  BLIND_SLOTS,
  blindOrderCandidates,
  buildBlindMapping,
  buildEvaluationExportPayload,
  createEvaluationSessionState,
  EVALUATION_SESSION_STORAGE_KEY,
  legacyReviewerStorageKey,
  restoreEvaluationSessionState,
  restoreEvaluationSessionStateWithLegacySupport,
} from '../src/domain/evaluation'

const buildCandidate = (modelId: string, runId = modelId): SnapshotCandidate => ({
  id: `${runId}:question`,
  runId,
  runLabel: runId,
  modelId,
  answerText: modelId,
  success: true,
  artifacts: [],
})

describe('evaluation utils', () => {
  it('uses a stable session-scoped blind ordering', () => {
    const candidates = ['alpha', 'beta', 'gamma'].map((item) => buildCandidate(item))
    const first = blindOrderCandidates(candidates, 'session-a', 'medagentboard', '7')
    const second = blindOrderCandidates(candidates, 'session-a', 'medagentboard', '7')

    expect(first.map((item) => item.candidate.modelId)).toEqual(second.map((item) => item.candidate.modelId))
    expect(first.map((item) => item.slot)).toEqual(BLIND_SLOTS.slice(0, 3))
  })

  it('changes candidate order across sessions', () => {
    const candidates = ['alpha', 'beta', 'gamma'].map((item) => buildCandidate(item))
    const sessionA = blindOrderCandidates(candidates, 'session-a', 'medagentboard', '7')
    const sessionB = blindOrderCandidates(candidates, 'session-b', 'medagentboard', '7')

    expect(sessionA.map((item) => item.candidate.modelId)).not.toEqual(sessionB.map((item) => item.candidate.modelId))
  })

  it('uses a single session storage key and preserves legacy reviewer keys for migration', () => {
    expect(EVALUATION_SESSION_STORAGE_KEY).toBe('healthflow:evaluation:session')
    expect(legacyReviewerStorageKey(' reviewer-7 ')).toBe('healthflow:evaluation:reviewer-7')
  })

  it('starts session state with browser-scoped memory buckets', () => {
    const state = createEvaluationSessionState(' session-2 ')

    expect(state).toEqual({
      sessionId: 'session-2',
      activeBenchmarkId: null,
      activeRunIdByDataset: {},
      activeCompareKeyByDataset: {},
      currentIndexByDataset: {},
      responses: {},
      drafts: {},
      lastParticipantName: '',
    })
  })

  it('restores session state from partial cached payloads', () => {
    expect(
      restoreEvaluationSessionState({
        sessionId: ' session-7 ',
        activeBenchmarkId: ' medagentboard ',
        activeRunIdByDataset: {
          medagentboard: ' medagentboard-healthflow ',
          bad: 7,
        },
        activeCompareKeyByDataset: {
          medagentboard: ' reference ',
        },
        currentIndexByDataset: {
          medagentboard: 2,
          negative: -1,
        },
        drafts: {
          'medagentboard:1': {
            choice: ' medagentboard-healthflow ',
            note: 'draft note',
            updatedAt: ' 2026-04-09T00:00:00.000Z ',
          },
        },
        responses: {
          'medagentboard:1': {
            datasetId: ' medagentboard ',
            qid: ' 1 ',
            choice: ' medagentboard-healthflow ',
            selectedBlindSlot: ' A ',
            selectedRunId: ' medagentboard-healthflow ',
            selectedRunLabel: ' HealthFlow ',
            selectedModelId: ' healthflow ',
            selectedBackend: ' codex ',
            blindMapping: [
              {
                slot: ' A ',
                runId: ' medagentboard-healthflow ',
                runLabel: ' HealthFlow ',
                modelId: ' healthflow ',
                backend: ' codex ',
              },
            ],
            note: 'saved note',
            updatedAt: ' 2026-04-09T00:00:00.000Z ',
          },
        },
        lastParticipantName: ' Alice ',
      }),
    ).toEqual({
      sessionId: 'session-7',
      activeBenchmarkId: 'medagentboard',
      activeRunIdByDataset: {
        medagentboard: 'medagentboard-healthflow',
      },
      activeCompareKeyByDataset: {
        medagentboard: 'reference',
      },
      currentIndexByDataset: {
        medagentboard: 2,
      },
      drafts: {
        'medagentboard:1': {
          choice: 'medagentboard-healthflow',
          note: 'draft note',
          updatedAt: '2026-04-09T00:00:00.000Z',
        },
      },
      responses: {
        'medagentboard:1': {
          questionId: 'medagentboard:1',
          datasetId: 'medagentboard',
          qid: '1',
          choice: 'medagentboard-healthflow',
          selectedBlindSlot: 'A',
          selectedRunId: 'medagentboard-healthflow',
          selectedRunLabel: 'HealthFlow',
          selectedModelId: 'healthflow',
          selectedBackend: 'codex',
          blindMapping: [
            {
              slot: 'A',
              runId: 'medagentboard-healthflow',
              runLabel: 'HealthFlow',
              modelId: 'healthflow',
              backend: 'codex',
            },
          ],
          note: 'saved note',
          updatedAt: '2026-04-09T00:00:00.000Z',
        },
      },
      lastParticipantName: 'Alice',
    })
  })

  it('migrates legacy reviewer state into the single-session shape', () => {
    expect(
      restoreEvaluationSessionStateWithLegacySupport({
        sessionValue: null,
        legacyReviewerId: ' reviewer-9 ',
        legacyReviewerValue: {
          reviewerId: ' reviewer-9 ',
          activeBenchmarkId: ' medagentboard ',
          activeRunIdByDataset: {
            medagentboard: ' medagentboard-healthflow ',
          },
          currentIndexByDataset: {
            medagentboard: 1,
          },
          responses: {
            'medagentboard:1': {
              datasetId: ' medagentboard ',
              qid: ' 1 ',
              choice: ' medagentboard-healthflow ',
              selectedSlot: ' B ',
              selectedModelId: ' healthflow ',
              slotMapping: {
                A: ' alpha ',
                B: ' healthflow ',
              },
              note: 'legacy note',
              updatedAt: ' 2026-04-09T00:00:00.000Z ',
            },
          },
        },
      }),
    ).toEqual({
      sessionId: 'reviewer-9',
      activeBenchmarkId: 'medagentboard',
      activeRunIdByDataset: {
        medagentboard: 'medagentboard-healthflow',
      },
      activeCompareKeyByDataset: {},
      currentIndexByDataset: {
        medagentboard: 1,
      },
      drafts: {},
      responses: {
        'medagentboard:1': {
          questionId: 'medagentboard:1',
          datasetId: 'medagentboard',
          qid: '1',
          choice: 'medagentboard-healthflow',
          selectedBlindSlot: 'B',
          selectedRunId: 'healthflow',
          selectedRunLabel: 'healthflow',
          selectedModelId: 'healthflow',
          selectedBackend: null,
          blindMapping: [
            {
              slot: 'A',
              runId: 'alpha',
              runLabel: 'alpha',
              modelId: 'alpha',
              backend: null,
            },
            {
              slot: 'B',
              runId: 'healthflow',
              runLabel: 'healthflow',
              modelId: 'healthflow',
              backend: null,
            },
          ],
          note: 'legacy note',
          updatedAt: '2026-04-09T00:00:00.000Z',
        },
      },
      lastParticipantName: '',
    })
  })

  it('builds de-anonymized export payloads and requires a participant name', () => {
    const blindMapping = buildBlindMapping([
      {
        slot: 'A',
        candidate: buildCandidate('alpha', 'medagentboard-alpha'),
      },
      {
        slot: 'B',
        candidate: buildCandidate('healthflow', 'medagentboard-healthflow'),
      },
    ])
    const sessionState: EvaluationSessionState = {
      ...createEvaluationSessionState('session-export'),
      responses: {
        'medagentboard:1': {
          questionId: 'medagentboard:1',
          datasetId: 'medagentboard',
          qid: '1',
          choice: 'medagentboard-healthflow',
          selectedBlindSlot: 'B',
          selectedRunId: 'medagentboard-healthflow',
          selectedRunLabel: 'HealthFlow',
          selectedModelId: 'healthflow',
          selectedBackend: 'codex',
          blindMapping,
          note: 'best artifact quality',
          updatedAt: '2026-04-09T00:00:00.000Z',
        },
      },
    }

    expect(() =>
      buildEvaluationExportPayload({
        participantName: '   ',
        snapshotVersion: 'healthflow-local-dev',
        sessionState,
        questions: [],
      }),
    ).toThrow(/Participant name is required/)

    expect(
      buildEvaluationExportPayload({
        participantName: ' Alice ',
        snapshotVersion: 'healthflow-local-dev',
        exportedAt: '2026-04-10T00:00:00.000Z',
        sessionState,
        questions: [
          {
            caseId: '0001',
            id: 'medagentboard:1',
            datasetId: 'medagentboard',
            datasetLabel: 'MedAgentBoard',
            qid: '1',
            candidateCount: 2,
          },
        ],
      }),
    ).toEqual({
      participant_name: 'Alice',
      session_id: 'session-export',
      snapshot_version: 'healthflow-local-dev',
      exported_at: '2026-04-10T00:00:00.000Z',
      responses: [
        {
          questionId: 'medagentboard:1',
          datasetId: 'medagentboard',
          qid: '1',
          choice: 'B',
          selectedBlindSlot: 'B',
          selectedRunId: 'medagentboard-healthflow',
          selectedRunLabel: 'HealthFlow',
          selectedModelId: 'healthflow',
          selectedBackend: 'codex',
          blindMapping,
          note: 'best artifact quality',
          updatedAt: '2026-04-09T00:00:00.000Z',
        },
      ],
    })
  })
})
