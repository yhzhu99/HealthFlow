import { readdir, readFile, stat } from 'node:fs/promises'
import path from 'node:path'

import type {
  BenchmarkId,
  DevEvaluationPayload,
  DiagnosticEvaluationDiagnostics,
  EvaluationSnapshot,
  LiveEvaluationDiagnostics,
  SnapshotArtifact,
  SnapshotArtifactKind,
  SnapshotCandidate,
  SnapshotExpectedOutput,
  SnapshotQuestion,
  SnapshotReference,
  SnapshotRun,
} from '../src/domain/evaluation'

type AssetMode = 'dev' | 'static'

interface BenchmarkFile {
  id?: BenchmarkId
  label?: string
  description?: string
  frameworkOrder?: string[]
}

interface CaseFile {
  id?: string
  qid?: string | number
  task?: string
  taskBrief?: string | null
  taskType?: string | null
  paperTitle?: string | null
  options?: Record<string, string> | null
  reportRequirements?: string[]
  expectedOutputs?: SnapshotExpectedOutput[]
}

interface FrameworkManifestFile {
  runId?: string
  runLabel?: string
  modelId?: string
  backend?: string | null
  summary?: string | null
  score?: number | null
  success?: boolean
}

interface ArtifactCopy {
  sourcePath: string
  relativePath: string
}

interface InternalArtifact {
  artifact: SnapshotArtifact
  sourcePath: string
}

export interface EvaluationSnapshotBundle {
  payload: DevEvaluationPayload
  artifactCopies: ArtifactCopy[]
}

const DEFAULT_SNAPSHOT_VERSION = 'healthflow-local-dev'
const EVALUATION_ROUTE_PREFIX = '/__eval'
const EVALUATION_ARTIFACT_ROUTE_PREFIX = `${EVALUATION_ROUTE_PREFIX}/artifacts`
const EVALUATION_MANIFEST_ROUTE = `${EVALUATION_ROUTE_PREFIX}/manifest`

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'])
const TEXT_EXTENSIONS = new Set(['.md', '.txt', '.log', '.rst'])
const STRUCTURED_EXTENSIONS = new Set(['.json', '.jsonl'])
const TABULAR_EXTENSIONS = new Set(['.csv', '.tsv'])

export const evaluationDataRootForProject = (projectRoot: string) => path.join(projectRoot, 'evaluation-data')
export const evaluationManifestRoute = () => EVALUATION_MANIFEST_ROUTE

export const evaluationArtifactRouteFromSegments = (segments: string[]) =>
  `${EVALUATION_ARTIFACT_ROUTE_PREFIX}/${segments.map(encodeURIComponent).join('/')}`

const sortDirectoryNames = (values: string[]) =>
  [...values].sort((left, right) =>
    left.localeCompare(right, undefined, {
      numeric: true,
      sensitivity: 'base',
    }),
  )

const relativeToPosix = (basePath: string, targetPath: string) => path.relative(basePath, targetPath).split(path.sep).join('/')

const mediaTypeForPath = (filePath: string) => {
  const extension = path.extname(filePath).toLowerCase()
  if (extension === '.md') return 'text/markdown'
  if (extension === '.pdf') return 'application/pdf'
  if (extension === '.csv') return 'text/csv'
  if (extension === '.tsv') return 'text/tab-separated-values'
  if (extension === '.json') return 'application/json'
  if (extension === '.jsonl') return 'application/x-ndjson'
  if (extension === '.png') return 'image/png'
  if (extension === '.jpg' || extension === '.jpeg') return 'image/jpeg'
  if (extension === '.gif') return 'image/gif'
  if (extension === '.svg') return 'image/svg+xml'
  if (extension === '.webp') return 'image/webp'
  return 'text/plain'
}

const artifactKindForPath = (filePath: string): SnapshotArtifactKind => {
  const extension = path.extname(filePath).toLowerCase()
  if (extension === '.md') return 'markdown'
  if (extension === '.pdf') return 'pdf'
  if (IMAGE_EXTENSIONS.has(extension)) return 'image'
  if (TABULAR_EXTENSIONS.has(extension)) return 'csv'
  if (STRUCTURED_EXTENSIONS.has(extension)) return 'json'
  if (TEXT_EXTENSIONS.has(extension)) return 'text'
  return 'download'
}

const previewPriority = (relativePath: string) => {
  const lowerPath = relativePath.toLowerCase()
  let score = 0

  if (lowerPath.endsWith('report.md')) score += 320
  if (lowerPath.endsWith('.pdf')) score += 280
  if (lowerPath.includes('/figures/') || lowerPath.includes('/images/')) score += 180
  if (lowerPath.endsWith('.png') || lowerPath.endsWith('.jpg') || lowerPath.endsWith('.svg')) score += 160
  if (lowerPath.endsWith('.csv')) score += 140
  if (lowerPath.endsWith('.json')) score += 100

  return score
}

const safeReadJson = async <T>(filePath: string): Promise<T> => JSON.parse(await readFile(filePath, 'utf-8')) as T

const readTextOrNull = async (filePath: string) => {
  try {
    const fileInfo = await stat(filePath)
    if (!fileInfo.isFile()) return null
    return await readFile(filePath, 'utf-8')
  } catch {
    return null
  }
}

const isDirectory = async (targetPath: string) => {
  try {
    return (await stat(targetPath)).isDirectory()
  } catch {
    return false
  }
}

const isFile = async (targetPath: string) => {
  try {
    return (await stat(targetPath)).isFile()
  } catch {
    return false
  }
}

const listDirectories = async (rootPath: string) => {
  const entries = await readdir(rootPath, { withFileTypes: true })
  return sortDirectoryNames(entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name))
}

const listFilesRecursive = async (rootPath: string): Promise<string[]> => {
  const entries = await readdir(rootPath, { withFileTypes: true })
  const files: string[] = []

  for (const entry of entries) {
    const resolvedPath = path.join(rootPath, entry.name)
    if (entry.isDirectory()) {
      files.push(...(await listFilesRecursive(resolvedPath)))
      continue
    }
    if (entry.isFile()) {
      files.push(resolvedPath)
    }
  }

  return files
}

const artifactRelativePath = (mode: AssetMode, routeSegments: string[]) =>
  mode === 'dev'
    ? evaluationArtifactRouteFromSegments(routeSegments)
    : path.posix.join('evaluation-assets', ...routeSegments)

const createArtifact = ({
  mode,
  sourcePath,
  label,
  originalPath,
  routeSegments,
}: {
  mode: AssetMode
  sourcePath: string
  label: string
  originalPath: string
  routeSegments: string[]
}): InternalArtifact => ({
  sourcePath,
  artifact: {
    id: routeSegments.join(':'),
    label,
    kind: artifactKindForPath(sourcePath),
    mediaType: mediaTypeForPath(sourcePath),
    relativePath: artifactRelativePath(mode, routeSegments),
    originalPath,
    previewPriority: previewPriority(originalPath),
  },
})

const loadArtifacts = async ({
  mode,
  sourceRoot,
  routeBaseSegments,
  originalPathPrefix,
}: {
  mode: AssetMode
  sourceRoot: string
  routeBaseSegments: string[]
  originalPathPrefix: string
}): Promise<InternalArtifact[]> => {
  if (!(await isDirectory(sourceRoot))) {
    return []
  }

  const files = await listFilesRecursive(sourceRoot)
  const artifacts = files.map((sourcePath) => {
    const relativePath = relativeToPosix(sourceRoot, sourcePath)
    return createArtifact({
      mode,
      sourcePath,
      label: path.basename(sourcePath),
      originalPath: path.posix.join(originalPathPrefix, relativePath),
      routeSegments: [...routeBaseSegments, ...relativePath.split('/')],
    })
  })

  artifacts.sort((left, right) => {
    if (left.artifact.previewPriority === right.artifact.previewPriority) {
      return left.artifact.originalPath.localeCompare(right.artifact.originalPath)
    }
    return right.artifact.previewPriority - left.artifact.previewPriority
  })

  return artifacts
}

const findReportPath = (artifacts: SnapshotArtifact[]) =>
  artifacts.find((artifact) => artifact.label === 'report.md')?.relativePath ??
  artifacts.find((artifact) => artifact.originalPath.endsWith('/report.md'))?.relativePath ??
  null

const summarizeDiagnostics = (
  root: string,
  warnings: string[],
  missing: string[] = [],
  invalid: string[] = [],
): DiagnosticEvaluationDiagnostics => ({
  root,
  warnings,
  missing,
  invalid,
})

export const buildEvaluationSnapshotBundle = async ({
  projectRoot,
  mode,
}: {
  projectRoot: string
  mode: AssetMode
}): Promise<EvaluationSnapshotBundle> => {
  const root = evaluationDataRootForProject(projectRoot)
  const benchmarksRoot = path.join(root, 'benchmarks')
  const missing: string[] = []
  const invalid: string[] = []
  const warnings: string[] = []
  const artifactCopies: ArtifactCopy[] = []

  if (!(await isDirectory(root))) {
    missing.push(`Missing evaluation root: ${root}`)
  }

  if (!(await isDirectory(benchmarksRoot))) {
    missing.push(`Missing benchmarks directory: ${benchmarksRoot}`)
  }

  if (missing.length > 0) {
    return {
      payload: {
        mode: 'diagnostic',
        snapshot: null,
        diagnostics: summarizeDiagnostics(root, warnings, missing, invalid),
      },
      artifactCopies,
    }
  }

  const benchmarkDirectories = await listDirectories(benchmarksRoot)
  if (benchmarkDirectories.length === 0) {
    missing.push(`No benchmark directories found under ${benchmarksRoot}`)
    return {
      payload: {
        mode: 'diagnostic',
        snapshot: null,
        diagnostics: summarizeDiagnostics(root, warnings, missing, invalid),
      },
      artifactCopies,
    }
  }

  const benchmarks = []
  const questions: SnapshotQuestion[] = []
  const runsByKey = new Map<string, SnapshotRun>()

  for (const benchmarkDirectory of benchmarkDirectories) {
    const benchmarkRoot = path.join(benchmarksRoot, benchmarkDirectory)
    const benchmarkConfigPath = path.join(benchmarkRoot, 'benchmark.json')
    if (!(await isFile(benchmarkConfigPath))) {
      invalid.push(`Missing benchmark.json for ${benchmarkDirectory}`)
      continue
    }

    let benchmarkConfig: BenchmarkFile
    try {
      benchmarkConfig = await safeReadJson<BenchmarkFile>(benchmarkConfigPath)
    } catch (caughtError) {
      invalid.push(`Invalid benchmark.json for ${benchmarkDirectory}: ${String(caughtError)}`)
      continue
    }

    const benchmarkId = String(benchmarkConfig.id ?? benchmarkDirectory)
    const benchmarkLabel = benchmarkConfig.label?.trim() || benchmarkId
    const frameworkOrder = new Map((benchmarkConfig.frameworkOrder ?? []).map((item, index) => [item, index]))
    const casesRoot = path.join(benchmarkRoot, 'cases')
    if (!(await isDirectory(casesRoot))) {
      invalid.push(`Missing cases directory for ${benchmarkId}`)
      continue
    }

    const caseDirectories = await listDirectories(casesRoot)
    if (caseDirectories.length === 0) {
      invalid.push(`No cases found for ${benchmarkId}`)
      continue
    }

    const benchmarkQuestions: SnapshotQuestion[] = []

    for (const caseDirectory of caseDirectories) {
      const caseRoot = path.join(casesRoot, caseDirectory)
      const caseConfigPath = path.join(caseRoot, 'case.json')
      if (!(await isFile(caseConfigPath))) {
        invalid.push(`Missing case.json for ${benchmarkId}/${caseDirectory}`)
        continue
      }

      let caseConfig: CaseFile
      try {
        caseConfig = await safeReadJson<CaseFile>(caseConfigPath)
      } catch (caughtError) {
        invalid.push(`Invalid case.json for ${benchmarkId}/${caseDirectory}: ${String(caughtError)}`)
        continue
      }

      const frameworksRoot = path.join(caseRoot, 'frameworks')
      if (!(await isDirectory(frameworksRoot))) {
        invalid.push(`Missing frameworks directory for ${benchmarkId}/${caseDirectory}`)
        continue
      }

      const frameworkDirectories = await listDirectories(frameworksRoot)
      if (frameworkDirectories.length === 0) {
        invalid.push(`No frameworks found for ${benchmarkId}/${caseDirectory}`)
        continue
      }

      const orderedFrameworkDirectories = [...frameworkDirectories].sort((left, right) => {
        const leftOrder = frameworkOrder.get(left) ?? Number.MAX_SAFE_INTEGER
        const rightOrder = frameworkOrder.get(right) ?? Number.MAX_SAFE_INTEGER
        if (leftOrder !== rightOrder) return leftOrder - rightOrder
        return left.localeCompare(right, undefined, { numeric: true, sensitivity: 'base' })
      })

      const referenceAnswerPath = path.join(caseRoot, 'reference', 'answer.md')
      const referenceAnswer = await readTextOrNull(referenceAnswerPath)
      const referenceArtifacts = await loadArtifacts({
        mode,
        sourceRoot: path.join(caseRoot, 'reference', 'files'),
        routeBaseSegments: [benchmarkId, caseDirectory, 'reference'],
        originalPathPrefix: 'reference/files',
      })
      artifactCopies.push(...referenceArtifacts.map(({ sourcePath, artifact }) => ({ sourcePath, relativePath: artifact.relativePath })))

      const candidates: SnapshotCandidate[] = []
      for (const frameworkDirectory of orderedFrameworkDirectories) {
        const frameworkRoot = path.join(frameworksRoot, frameworkDirectory)
        const frameworkManifestPath = path.join(frameworkRoot, 'manifest.json')
        if (!(await isFile(frameworkManifestPath))) {
          warnings.push(`Skipping ${benchmarkId}/${caseDirectory}/${frameworkDirectory}: missing manifest.json`)
          continue
        }

        let manifest: FrameworkManifestFile
        try {
          manifest = await safeReadJson<FrameworkManifestFile>(frameworkManifestPath)
        } catch (caughtError) {
          warnings.push(`Skipping ${benchmarkId}/${caseDirectory}/${frameworkDirectory}: invalid manifest.json (${String(caughtError)})`)
          continue
        }

        const runLabel = manifest.runLabel?.trim() || frameworkDirectory
        const runId = manifest.runId?.trim() || `${benchmarkId}-${frameworkDirectory}`
        const modelId = manifest.modelId?.trim() || frameworkDirectory
        const answerText =
          (await readTextOrNull(path.join(frameworkRoot, 'answer.md'))) ??
          manifest.summary?.trim() ??
          'No final answer was recorded.'
        const frameworkArtifacts = await loadArtifacts({
          mode,
          sourceRoot: path.join(frameworkRoot, 'files'),
          routeBaseSegments: [benchmarkId, caseDirectory, frameworkDirectory],
          originalPathPrefix: `frameworks/${frameworkDirectory}/files`,
        })
        artifactCopies.push(
          ...frameworkArtifacts.map(({ sourcePath, artifact }) => ({ sourcePath, relativePath: artifact.relativePath })),
        )

        const candidateArtifacts = frameworkArtifacts.map((item) => item.artifact)
        const candidate: SnapshotCandidate = {
          id: `${runId}:${String(caseConfig.qid ?? caseDirectory)}`,
          runId,
          runLabel,
          modelId,
          backend: manifest.backend ?? null,
          answerText,
          summary: manifest.summary ?? null,
          score: manifest.score ?? null,
          success: manifest.success ?? true,
          reportPath: findReportPath(candidateArtifacts),
          artifacts: candidateArtifacts,
        }
        candidates.push(candidate)

        const existingRun = runsByKey.get(`${benchmarkId}:${runId}`)
        if (!existingRun) {
          runsByKey.set(`${benchmarkId}:${runId}`, {
            id: runId,
            label: runLabel,
            modelId,
            datasetId: benchmarkId,
          })
        }
      }

      if (candidates.length === 0) {
        invalid.push(`No valid framework candidates found for ${benchmarkId}/${caseDirectory}`)
        continue
      }

      const referenceArtifactsForQuestion = referenceArtifacts.map((item) => item.artifact)
      const reference: SnapshotReference = {
        mode:
          referenceArtifactsForQuestion.length > 0
            ? 'artifacts'
            : referenceAnswer
              ? 'text'
              : 'none',
        text: referenceAnswer,
        note: null,
        artifacts: referenceArtifactsForQuestion,
        requiredOutputs: caseConfig.expectedOutputs ?? [],
      }

      benchmarkQuestions.push({
        id: caseConfig.id?.trim() || `${benchmarkId}:${String(caseConfig.qid ?? caseDirectory)}`,
        datasetId: benchmarkId,
        datasetLabel: benchmarkLabel,
        qid: String(caseConfig.qid ?? caseDirectory),
        task: caseConfig.task?.trim() || 'Task description missing.',
        taskBrief: caseConfig.taskBrief ?? null,
        taskType: caseConfig.taskType ?? null,
        paperTitle: caseConfig.paperTitle ?? null,
        options: caseConfig.options ?? null,
        reportRequirements: caseConfig.reportRequirements ?? [],
        expectedOutputs: caseConfig.expectedOutputs ?? [],
        reference,
        candidates,
      })
    }

    if (benchmarkQuestions.length === 0) {
      invalid.push(`No valid cases found for ${benchmarkId}`)
      continue
    }

    benchmarkQuestions.sort((left, right) =>
      left.qid.localeCompare(right.qid, undefined, {
        numeric: true,
        sensitivity: 'base',
      }),
    )
    questions.push(...benchmarkQuestions)
    benchmarks.push({
      id: benchmarkId,
      label: benchmarkLabel,
      description: benchmarkConfig.description?.trim() || `${benchmarkLabel} evaluation benchmark`,
      taskCount: benchmarkQuestions.length,
    })
  }

  if (questions.length === 0) {
    missing.push(`No valid evaluation cases were found under ${benchmarksRoot}`)
    return {
      payload: {
        mode: 'diagnostic',
        snapshot: null,
        diagnostics: summarizeDiagnostics(root, warnings, missing, invalid),
      },
      artifactCopies,
    }
  }

  const liveDiagnostics: LiveEvaluationDiagnostics = {
    root,
    warnings,
  }
  const snapshot: EvaluationSnapshot = {
    snapshotVersion: DEFAULT_SNAPSHOT_VERSION,
    generatedAt: new Date().toISOString(),
    site: {
      title: 'HealthFlow Platform',
      reviewerExportKeyPrefix: 'healthflow:evaluation',
    },
    benchmarks,
    runs: [...runsByKey.values()].sort((left, right) => {
      if (left.datasetId !== right.datasetId) {
        return left.datasetId.localeCompare(right.datasetId, undefined, { numeric: true, sensitivity: 'base' })
      }
      return left.label.localeCompare(right.label, undefined, { numeric: true, sensitivity: 'base' })
    }),
    questions,
  }

  if (missing.length > 0 || invalid.length > 0) {
    warnings.push(...missing, ...invalid)
  }

  return {
    payload: {
      mode: 'live',
      snapshot,
      diagnostics: liveDiagnostics,
    },
    artifactCopies,
  }
}
