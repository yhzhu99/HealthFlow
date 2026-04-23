export interface ParsedDelimitedTable {
  headers: string[]
  rows: string[][]
}

const detectDelimiter = (content: string) => {
  const firstLine = content.split(/\r?\n/, 1)[0] ?? ''
  const commaCount = (firstLine.match(/,/g) ?? []).length
  const tabCount = (firstLine.match(/\t/g) ?? []).length
  return tabCount > commaCount ? '\t' : ','
}

const parseDelimitedLine = (line: string, delimiter: string) => {
  const cells: string[] = []
  let current = ''
  let inQuotes = false

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index] ?? ''
    const nextCharacter = line[index + 1] ?? ''

    if (character === '"') {
      if (inQuotes && nextCharacter === '"') {
        current += '"'
        index += 1
        continue
      }

      inQuotes = !inQuotes
      continue
    }

    if (!inQuotes && character === delimiter) {
      cells.push(current.trim())
      current = ''
      continue
    }

    current += character
  }

  cells.push(current.trim())
  return cells
}

export const parseDelimitedText = (content: string): ParsedDelimitedTable => {
  const normalizedContent = content.replace(/\r\n/g, '\n').trim()
  if (!normalizedContent) {
    return { headers: [], rows: [] }
  }

  const delimiter = detectDelimiter(normalizedContent)
  const lines = normalizedContent
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)

  if (lines.length === 0) {
    return { headers: [], rows: [] }
  }

  const [headerLine, ...dataLines] = lines
  return {
    headers: parseDelimitedLine(headerLine ?? '', delimiter),
    rows: dataLines.map((line) => parseDelimitedLine(line, delimiter)),
  }
}
