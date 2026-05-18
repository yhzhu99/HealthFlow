import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import {
  appDemoArtifacts,
  appDemoInputFile,
  appDemoPrompt,
  appDemoStages,
  coreAppDemoArtifactIds,
  liveRuntimeUrl,
  resolveAppDemoArtifact,
} from '../src/content/appDemo'

const readPlatformFile = (path: string) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8')

describe('app demo content', () => {
  it('points the live runtime action at the dedicated runtime hostname', () => {
    const runtimeUrl = new URL(liveRuntimeUrl)

    expect(runtimeUrl.origin).toBe('https://healthflow-app.medx-pku.com')
    expect(runtimeUrl.pathname).toBe('/')
  })

  it('keeps the demo prompt and input cohort available as static content', () => {
    expect(appDemoPrompt).toContain('ICU EHR cohort')
    expect(appDemoInputFile).toMatchObject({
      name: 'icu_mortality_cohort.csv',
      href: '/demo/icu_mortality_cohort.csv',
      rows: 18,
    })
    expect(appDemoInputFile.columns).toContain('mortality')
    expect(appDemoInputFile.columns).toContain('heart_rate')
  })

  it('models the full HealthFlow replay stages', () => {
    expect(appDemoStages.map((stage) => stage.id)).toEqual([
      'memory',
      'planner',
      'executor',
      'evaluator',
      'reflector',
    ])
    expect(appDemoStages.every((stage) => stage.status === 'completed')).toBe(true)
  })

  it('exposes core workspace artifacts with previewable types', () => {
    const coreArtifacts = coreAppDemoArtifactIds.map((artifactId) => resolveAppDemoArtifact(artifactId))

    expect(coreArtifacts.every(Boolean)).toBe(true)
    expect(coreArtifacts.map((artifact) => artifact?.kind)).toEqual(
      expect.arrayContaining(['markdown', 'json', 'table', 'chart']),
    )
    expect(resolveAppDemoArtifact('report')?.path).toBe('runtime/report.md')
    expect(appDemoArtifacts.some((artifact) => artifact.kind === 'log')).toBe(true)
  })

  it('ships Cloudflare Pages fallback and downloadable cohort assets', () => {
    const redirects = readPlatformFile('public/_redirects')
    const cohortCsv = readPlatformFile('public/demo/icu_mortality_cohort.csv')
    const cohortLines = cohortCsv.split(/\r?\n/).filter(Boolean)

    expect(redirects.trim()).toBe('/* /index.html 200')
    expect(cohortLines).toHaveLength(appDemoInputFile.rows + 1)
    expect(cohortLines[0]?.split(',')).toEqual(appDemoInputFile.columns)
  })
})
