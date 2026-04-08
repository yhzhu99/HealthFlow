import { cp, mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { buildEvaluationSnapshotBundle } from '../dev/evaluation-data'
import type { EvaluationSnapshot } from '../src/domain/evaluation'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

export interface BuildSnapshotOptions {
  projectRoot?: string
}

export interface BuildSnapshotResult {
  outputPath: string
  snapshot: EvaluationSnapshot
  artifactCount: number
}

const ensureLiveBundle = async (projectRoot: string) => {
  const bundle = await buildEvaluationSnapshotBundle({
    projectRoot,
    mode: 'static',
  })

  if (bundle.payload.mode !== 'live') {
    const diagnostics = bundle.payload.diagnostics
    const issues = [...diagnostics.missing, ...diagnostics.invalid, ...diagnostics.warnings]
    throw new Error(
      ['Unable to export evaluation snapshot from platform/evaluation-data.', ...issues].join('\n'),
    )
  }

  return bundle
}

export const buildSnapshot = async ({
  projectRoot = path.resolve(__dirname, '..'),
}: BuildSnapshotOptions = {}): Promise<BuildSnapshotResult> => {
  const bundle = await ensureLiveBundle(projectRoot)
  const snapshot = bundle.payload.snapshot as EvaluationSnapshot
  const publicRoot = path.join(projectRoot, 'public')
  const snapshotOutputPath = path.join(publicRoot, 'data', 'evaluation.snapshot.json')
  const artifactRoot = path.join(publicRoot, 'evaluation-assets')

  await rm(artifactRoot, { recursive: true, force: true })
  await mkdir(path.dirname(snapshotOutputPath), { recursive: true })
  await mkdir(artifactRoot, { recursive: true })

  for (const artifact of bundle.artifactCopies) {
    const destinationPath = path.join(publicRoot, artifact.relativePath)
    await mkdir(path.dirname(destinationPath), { recursive: true })
    await cp(artifact.sourcePath, destinationPath)
  }

  await writeFile(snapshotOutputPath, JSON.stringify(snapshot, null, 2))

  return {
    outputPath: snapshotOutputPath,
    snapshot,
    artifactCount: bundle.artifactCopies.length,
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  const result = await buildSnapshot()
  console.log(`Exported ${result.snapshot.questions.length} questions and ${result.artifactCount} artifacts to ${path.dirname(path.dirname(result.outputPath))}`)
}
