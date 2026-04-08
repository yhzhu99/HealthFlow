import { cp, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import type {
  BenchmarkId,
  EvaluationSnapshot,
  SnapshotArtifact,
  SnapshotArtifactKind,
  SnapshotCandidate,
  SnapshotExpectedOutput,
  SnapshotQuestion,
  SnapshotReference,
  SnapshotRun,
} from '../src/domain/evaluation'

interface EvaluationRunConfig {
  id: string
  label: string
  modelId: string
  sourceResultsDir: string
}

interface EvaluationDatasetConfig {
  id: BenchmarkId
  label: string
  description: string
  taskSource: string
  referenceRoot: string
  split?: string | null
  selectedQids: Array<number | string>
  artifactPreferences?: string[]
  runs: EvaluationRunConfig[]
}

interface EvaluationConfig {
  snapshotVersion?: string
  site?: {
    title?: string
    reviewerExportKeyPrefix?: string
  }
  datasets: EvaluationDatasetConfig[]
}

interface BuildSnapshotOptions {
  projectRoot?: string
  configPath?: string
}

interface DatasetTaskRow {
  qid: string | number
  task: string
  task_brief?: string
  task_type?: string
  paper_title?: string
  options?: Record<string, string>
  report_requirements?: string[]
  reference_answer?: unknown
  answer?: unknown
}

interface ManifestRequiredOutput {
  file_name: string
  media_type?: string
  reference_path?: string
}

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const DEFAULT_PROJECT_ROOT = path.resolve(__dirname, '..')
const DEFAULT_CONFIG_PATH = path.resolve(DEFAULT_PROJECT_ROOT, 'content/evaluation.config.json')

const EXCLUDED_PREFIXES = ['runtime/attempts/', 'runtime/run/', 'runtime/turns/']
const EXCLUDED_EXACT = new Set([
  'result.json',
  'runtime/events.jsonl',
  'runtime/index.json',
  'runtime/run/costs.json',
  'runtime/run/final_evaluation.json',
  'runtime/run/summary.json',
  'runtime/run/trajectory.json',
])

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'])
const TEXT_EXTENSIONS = new Set(['.md', '.txt', '.log', '.rst'])
const STRUCTURED_EXTENSIONS = new Set(['.json', '.jsonl'])
const TABULAR_EXTENSIONS = new Set(['.csv', '.tsv'])

export const mediaTypeForPath = (filePath: string) => {
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

export const artifactKindForPath = (filePath: string): SnapshotArtifactKind => {
  const extension = path.extname(filePath).toLowerCase()
  if (extension === '.md') return 'markdown'
  if (extension === '.pdf') return 'pdf'
  if (IMAGE_EXTENSIONS.has(extension)) return 'image'
  if (TABULAR_EXTENSIONS.has(extension)) return 'csv'
  if (extension === '.json' || extension === '.jsonl') return 'json'
  if (TEXT_EXTENSIONS.has(extension)) return 'text'
  return 'download'
}

const isPreviewablePath = (filePath: string) => {
  const extension = path.extname(filePath).toLowerCase()
  return (
    extension === '.pdf' ||
    extension === '.md' ||
    IMAGE_EXTENSIONS.has(extension) ||
    TEXT_EXTENSIONS.has(extension) ||
    STRUCTURED_EXTENSIONS.has(extension) ||
    TABULAR_EXTENSIONS.has(extension)
  )
}

const normalizeQid = (value: string | number) => String(value).trim()

const sanitizeSegment = (value: string) => value.replace(/[^a-zA-Z0-9._-]+/g, '-')

const resolveFromConfig = (configPath: string, value: string) => {
  if (path.isAbsolute(value)) return value
  return path.resolve(path.dirname(configPath), value)
}

const relativeToPosix = (basePath: string, targetPath: string) => path.relative(basePath, targetPath).split(path.sep).join('/')

const parseJson = async <T>(filePath: string): Promise<T> => {
  const raw = await readFile(filePath, 'utf-8')
  return JSON.parse(raw) as T
}

const parseJsonl = async <T>(filePath: string): Promise<T[]> => {
  const raw = await readFile(filePath, 'utf-8')
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T)
}

const resolveReferencePath = async (referenceRoot: string, candidate: string) => {
  const attempts = [path.resolve(referenceRoot, candidate), path.resolve(path.dirname(referenceRoot), candidate)]
  for (const attempt of attempts) {
    try {
      const fileInfo = await stat(attempt)
      if (fileInfo.isFile()) {
        return attempt
      }
    } catch {
      continue
    }
  }
  return null
}

const listFilesRecursive = async (rootPath: string): Promise<string[]> => {
  const results: string[] = []
  const entries = await readdir(rootPath, { withFileTypes: true })

  for (const entry of entries) {
    const resolvedPath = path.join(rootPath, entry.name)
    if (entry.isDirectory()) {
      const nested = await listFilesRecursive(resolvedPath)
      results.push(...nested)
      continue
    }
    if (entry.isFile()) {
      results.push(resolvedPath)
    }
  }

  return results
}

const previewPriority = (relativePath: string, preferences: string[]) => {
  const lowerPath = relativePath.toLowerCase()
  let score = 0

  preferences.forEach((preference, index) => {
    if (lowerPath.includes(preference.toLowerCase())) {
      score += 200 - index * 10
    }
  })

  if (lowerPath.endsWith('report.md')) score += 320
  if (lowerPath.endsWith('.pdf')) score += 280
  if (lowerPath.includes('/figures/') || lowerPath.includes('/images/')) score += 180
  if (lowerPath.endsWith('.png') || lowerPath.endsWith('.jpg')) score += 160
  if (lowerPath.endsWith('.csv')) score += 140
  if (lowerPath.endsWith('.json')) score += 100

  return score
}

const copyPublicArtifact = async (
  sourcePath: string,
  {
    outputRoot,
    destinationRelativePath,
  }: {
    outputRoot: string
    destinationRelativePath: string
  },
): Promise<string> => {
  const destinationPath = path.join(outputRoot, destinationRelativePath)
  await mkdir(path.dirname(destinationPath), { recursive: true })
  await cp(sourcePath, destinationPath)
  return destinationRelativePath.split(path.sep).join('/')
}

const buildArtifact = async (
  sourcePath: string,
  {
    outputRoot,
    destinationRelativePath,
    label,
    originalPath,
    priority,
  }: {
    outputRoot: string
    destinationRelativePath: string
    label: string
    originalPath: string
    priority: number
  },
): Promise<SnapshotArtifact> => {
  const relativePath = await copyPublicArtifact(sourcePath, { outputRoot, destinationRelativePath })
  return {
    id: `${destinationRelativePath}:${label}`,
    label,
    kind: artifactKindForPath(sourcePath),
    mediaType: mediaTypeForPath(sourcePath),
    relativePath,
    originalPath,
    previewPriority: priority,
  }
}

const requiredOutputsFromManifest = (manifest: Record<string, unknown>): SnapshotExpectedOutput[] => {
  const requiredOutputs = manifest.required_outputs
  if (Array.isArray(requiredOutputs)) {
    return requiredOutputs.map((item) => {
      const typedItem = item as ManifestRequiredOutput
      return {
        fileName: typedItem.file_name,
        mediaType: typedItem.media_type ?? mediaTypeForPath(typedItem.file_name),
        referencePath: typedItem.reference_path ?? null,
      }
    })
  }

  const legacyOutputs = manifest.primary_outputs
  if (Array.isArray(legacyOutputs)) {
    return legacyOutputs.map((item) => {
      const fileName = path.basename(String(item))
      return {
        fileName,
        mediaType: mediaTypeForPath(fileName),
        referencePath: String(item),
      }
    })
  }

  return []
}

const buildReference = async (
  row: DatasetTaskRow,
  {
    datasetConfig,
    outputRoot,
  }: {
    datasetConfig: EvaluationDatasetConfig
    outputRoot: string
  },
): Promise<{ reference: SnapshotReference; expectedOutputs: SnapshotExpectedOutput[] }> => {
  const referenceValue = row.reference_answer ?? row.answer

  if (referenceValue == null) {
    return {
      reference: { mode: 'none', artifacts: [], requiredOutputs: [], note: 'No reference answer metadata was found.' },
      expectedOutputs: [],
    }
  }

  if (Array.isArray(referenceValue)) {
    const artifacts: SnapshotArtifact[] = []
    const textParts: string[] = []

    for (const item of referenceValue) {
      if (typeof item !== 'string') continue
      const resolvedPath = await resolveReferencePath(datasetConfig.referenceRoot, item)
      if (resolvedPath) {
        const artifact = await buildArtifact(resolvedPath, {
          outputRoot,
          destinationRelativePath: path.posix.join(
            'evaluation-assets',
            'reference',
            datasetConfig.id,
            normalizeQid(row.qid),
            item.split(path.sep).join('/'),
          ),
          label: path.basename(item),
          originalPath: item,
          priority: previewPriority(item, datasetConfig.artifactPreferences ?? []),
        })
        artifacts.push(artifact)
      } else {
        textParts.push(item)
      }
    }

    return {
      reference: {
        mode: artifacts.length > 0 ? 'artifacts' : 'text',
        text: textParts.join('\n\n') || null,
        artifacts,
        requiredOutputs: [],
      },
      expectedOutputs: artifacts.map((artifact) => ({
        fileName: artifact.label,
        mediaType: artifact.mediaType,
        referencePath: artifact.originalPath,
      })),
    }
  }

  if (typeof referenceValue === 'string') {
    const resolvedPath = await resolveReferencePath(datasetConfig.referenceRoot, referenceValue)
    if (resolvedPath && referenceValue.endsWith('.json')) {
      const manifest = await parseJson<Record<string, unknown>>(resolvedPath)
      const requiredOutputs = requiredOutputsFromManifest(manifest)
      const artifacts: SnapshotArtifact[] = []

      for (const output of requiredOutputs) {
        if (!output.referencePath) continue
        const manifestArtifactPath = await resolveReferencePath(datasetConfig.referenceRoot, output.referencePath)
        if (!manifestArtifactPath) continue
        artifacts.push(
          await buildArtifact(manifestArtifactPath, {
            outputRoot,
            destinationRelativePath: path.posix.join(
              'evaluation-assets',
              'reference',
              datasetConfig.id,
              normalizeQid(row.qid),
              output.referencePath.split(path.sep).join('/'),
            ),
            label: output.fileName,
            originalPath: output.referencePath,
            priority: previewPriority(output.referencePath, datasetConfig.artifactPreferences ?? []),
          }),
        )
      }

      return {
        reference: {
          mode: 'manifest',
          text: null,
          note:
            artifacts.length > 0
              ? 'Reference manifest and deliverable previews are available.'
              : 'Reference manifest is available, but no concrete reference deliverables were found locally.',
          artifacts,
          requiredOutputs,
        },
        expectedOutputs: requiredOutputs,
      }
    }

    if (resolvedPath && isPreviewablePath(resolvedPath)) {
      const artifact = await buildArtifact(resolvedPath, {
        outputRoot,
        destinationRelativePath: path.posix.join(
          'evaluation-assets',
          'reference',
          datasetConfig.id,
          normalizeQid(row.qid),
          referenceValue.split(path.sep).join('/'),
        ),
        label: path.basename(referenceValue),
        originalPath: referenceValue,
        priority: previewPriority(referenceValue, datasetConfig.artifactPreferences ?? []),
      })
      return {
        reference: {
          mode: 'artifacts',
          text: null,
          artifacts: [artifact],
          requiredOutputs: [],
        },
        expectedOutputs: [
          {
            fileName: artifact.label,
            mediaType: artifact.mediaType,
            referencePath: artifact.originalPath,
          },
        ],
      }
    }

    return {
      reference: {
        mode: 'text',
        text: referenceValue,
        artifacts: [],
        requiredOutputs: [],
      },
      expectedOutputs: [],
    }
  }

  return {
    reference: {
      mode: 'text',
      text: JSON.stringify(referenceValue, null, 2),
      artifacts: [],
      requiredOutputs: [],
    },
    expectedOutputs: [],
  }
}

const collectCandidateArtifacts = async (
  resultDir: string,
  {
    datasetConfig,
    run,
    qid,
    outputRoot,
  }: {
    datasetConfig: EvaluationDatasetConfig
    run: EvaluationRunConfig
    qid: string
    outputRoot: string
  },
): Promise<{ artifacts: SnapshotArtifact[]; reportPath: string | null }> => {
  const artifacts: SnapshotArtifact[] = []
  const files = await listFilesRecursive(resultDir)

  for (const filePath of files) {
    const relativePath = relativeToPosix(resultDir, filePath)
    if (!isPreviewablePath(relativePath)) continue
    if (EXCLUDED_EXACT.has(relativePath)) continue
    if (EXCLUDED_PREFIXES.some((prefix) => relativePath.startsWith(prefix))) continue

    artifacts.push(
      await buildArtifact(filePath, {
        outputRoot,
        destinationRelativePath: path.posix.join(
          'evaluation-assets',
          datasetConfig.id,
          sanitizeSegment(run.id),
          sanitizeSegment(qid),
          relativePath,
        ),
        label: path.basename(relativePath),
        originalPath: relativePath,
        priority: previewPriority(relativePath, datasetConfig.artifactPreferences ?? []),
      }),
    )
  }

  artifacts.sort((left, right) => {
    if (left.previewPriority === right.previewPriority) {
      return left.originalPath.localeCompare(right.originalPath)
    }
    return right.previewPriority - left.previewPriority
  })

  const reportArtifact =
    artifacts.find((artifact) => artifact.originalPath === 'runtime/report.md') ??
    artifacts.find((artifact) => artifact.originalPath.endsWith('report.md')) ??
    null

  return {
    artifacts,
    reportPath: reportArtifact?.relativePath ?? null,
  }
}

const buildQuestion = async (
  row: DatasetTaskRow,
  {
    datasetConfig,
    outputRoot,
    publicRuns,
  }: {
    datasetConfig: EvaluationDatasetConfig
    outputRoot: string
    publicRuns: SnapshotRun[]
  },
): Promise<SnapshotQuestion> => {
  const qid = normalizeQid(row.qid)
  const { reference, expectedOutputs } = await buildReference(row, {
    datasetConfig,
    outputRoot,
  })

  const candidates: SnapshotCandidate[] = []
  for (const run of datasetConfig.runs) {
    const resultDir = path.resolve(run.sourceResultsDir, qid)
    const resultPath = path.join(resultDir, 'result.json')

    let result: Record<string, unknown>
    try {
      result = await parseJson<Record<string, unknown>>(resultPath)
    } catch (error) {
      throw new Error(`Missing benchmark result for ${datasetConfig.id} qid ${qid} in run ${run.id}: ${String(error)}`)
    }

    const { artifacts, reportPath } = await collectCandidateArtifacts(resultDir, {
      datasetConfig,
      run,
      qid,
      outputRoot,
    })

    candidates.push({
      id: `${datasetConfig.id}:${run.id}:${qid}`,
      runId: run.id,
      runLabel: run.label,
      modelId: run.modelId,
      backend: typeof result.backend === 'string' ? result.backend : null,
      answerText:
        typeof result.generated_answer === 'string'
          ? result.generated_answer
          : typeof result.answer === 'string'
            ? result.answer
            : '',
      summary: typeof result.final_summary === 'string' ? result.final_summary : null,
      score:
        typeof result.score === 'number'
          ? result.score
          : typeof result.evaluation_score === 'number'
            ? result.evaluation_score
            : null,
      success: Boolean(result.success),
      reportPath,
      artifacts,
    })

    publicRuns.push({
      id: run.id,
      label: run.label,
      modelId: run.modelId,
      datasetId: datasetConfig.id,
    })
  }

  return {
    id: `${datasetConfig.id}:${qid}`,
    datasetId: datasetConfig.id,
    datasetLabel: datasetConfig.label,
    qid,
    task: row.task,
    taskBrief: row.task_brief ?? null,
    taskType: row.task_type ?? null,
    paperTitle: row.paper_title ?? null,
    options: row.options ?? null,
    reportRequirements: row.report_requirements ?? [],
    expectedOutputs,
    reference,
    candidates,
  }
}

const validateConfig = (config: EvaluationConfig) => {
  if (!Array.isArray(config.datasets) || config.datasets.length === 0) {
    throw new Error('Evaluation config must define at least one dataset.')
  }

  const datasetIds = new Set<string>()
  for (const dataset of config.datasets) {
    if (datasetIds.has(dataset.id)) {
      throw new Error(`Duplicate dataset id in evaluation config: ${dataset.id}`)
    }
    datasetIds.add(dataset.id)

    if (!dataset.taskSource || !dataset.referenceRoot) {
      throw new Error(`Dataset ${dataset.id} must define taskSource and referenceRoot.`)
    }
    if (!Array.isArray(dataset.selectedQids) || dataset.selectedQids.length === 0) {
      throw new Error(`Dataset ${dataset.id} must define at least one qid in selectedQids.`)
    }
    if (!Array.isArray(dataset.runs) || dataset.runs.length === 0) {
      throw new Error(`Dataset ${dataset.id} must define at least one run.`)
    }
  }
}

export const buildSnapshot = async ({ projectRoot = DEFAULT_PROJECT_ROOT, configPath = DEFAULT_CONFIG_PATH }: BuildSnapshotOptions = {}) => {
  const resolvedConfigPath = path.resolve(configPath)
  const config = await parseJson<EvaluationConfig>(resolvedConfigPath)
  validateConfig(config)

  const publicDataDir = path.join(projectRoot, 'public', 'data')
  const publicAssetsDir = path.join(projectRoot, 'public', 'evaluation-assets')
  const publicRoot = path.join(projectRoot, 'public')
  await rm(publicAssetsDir, { recursive: true, force: true })
  await mkdir(publicDataDir, { recursive: true })
  await mkdir(publicAssetsDir, { recursive: true })

  const runs: SnapshotRun[] = []
  const questions: SnapshotQuestion[] = []

  for (const dataset of config.datasets) {
    const taskSourcePath = resolveFromConfig(resolvedConfigPath, dataset.taskSource)
    const referenceRoot = resolveFromConfig(resolvedConfigPath, dataset.referenceRoot)
    const runConfigs = dataset.runs.map((run) => ({
      ...run,
      sourceResultsDir: resolveFromConfig(resolvedConfigPath, run.sourceResultsDir),
    }))

    const taskRows = await parseJsonl<DatasetTaskRow>(taskSourcePath)
    const taskMap = new Map(taskRows.map((row) => [normalizeQid(row.qid), row]))

    for (const rawQid of dataset.selectedQids) {
      const qid = normalizeQid(rawQid)
      const row = taskMap.get(qid)
      if (!row) {
        throw new Error(`Dataset ${dataset.id} is missing qid ${qid} in ${taskSourcePath}`)
      }
      questions.push(
        await buildQuestion(
          {
            ...row,
            qid,
          },
          {
            datasetConfig: {
              ...dataset,
              referenceRoot,
              taskSource: taskSourcePath,
              runs: runConfigs,
            },
            outputRoot: publicRoot,
            publicRuns: runs,
          },
        ),
      )
    }
  }

  const uniqueRuns = Array.from(new Map(runs.map((run) => [`${run.datasetId}:${run.id}`, run])).values())

  const snapshot: EvaluationSnapshot = {
    snapshotVersion: config.snapshotVersion ?? 'healthflow-eval-v1',
    generatedAt: new Date().toISOString(),
    site: {
      title: config.site?.title ?? 'HealthFlow Platform',
      reviewerExportKeyPrefix: config.site?.reviewerExportKeyPrefix ?? 'healthflow:evaluation',
    },
    benchmarks: config.datasets.map((dataset) => ({
      id: dataset.id,
      label: dataset.label,
      description: dataset.description,
      taskCount: dataset.selectedQids.length,
    })),
    runs: uniqueRuns,
    questions,
  }

  const outputPath = path.join(publicDataDir, 'evaluation.snapshot.json')
  await writeFile(outputPath, JSON.stringify(snapshot, null, 2) + '\n', 'utf-8')

  return {
    outputPath,
    questionCount: questions.length,
    runCount: uniqueRuns.length,
    snapshot,
  }
}

const main = async () => {
  const result = await buildSnapshot()
  console.log(`Snapshot written to ${result.outputPath}`)
  console.log(`Questions: ${result.questionCount}`)
  console.log(`Runs: ${result.runCount}`)
}

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : null
if (invokedPath === __filename) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error))
    process.exitCode = 1
  })
}
