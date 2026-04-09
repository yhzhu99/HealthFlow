import MarkdownIt from 'markdown-it'
import katex from 'katex'
// @ts-expect-error markdown-it-texmath does not ship TypeScript declarations.
import texmath from 'markdown-it-texmath'

import { toBasePath } from './assets'

const renderer = new MarkdownIt({
  breaks: true,
  html: false,
  linkify: true,
})
  .use(texmath, {
    engine: katex,
    delimiters: ['brackets', 'dollars', 'beg_end'],
    katexOptions: {
      output: 'html',
      throwOnError: false,
      strict: 'ignore',
      trust: false,
    },
  })

export interface RenderMarkdownOptions {
  assetBasePath?: string | null
}

const ABSOLUTE_RESOURCE_PATTERN = /^(?:[a-z]+:)?\/\//i
const SCHEME_RESOURCE_PATTERN = /^[a-z][a-z0-9+.-]*:/i

const isRelativeResourcePath = (value: string) => {
  if (!value) return false
  if (value.startsWith('#') || value.startsWith('/')) return false
  if (ABSOLUTE_RESOURCE_PATTERN.test(value) || SCHEME_RESOURCE_PATTERN.test(value)) return false
  return true
}

export const resolveMarkdownAssetUrl = (target: string, assetBasePath?: string | null) => {
  const normalizedBasePath = assetBasePath?.trim()
  if (!normalizedBasePath || !isRelativeResourcePath(target)) {
    return target
  }

  const baseUrl = new URL(toBasePath(normalizedBasePath), 'https://healthflow.local')
  const resolvedUrl = new URL(target, baseUrl)
  return `${resolvedUrl.pathname}${resolvedUrl.search}${resolvedUrl.hash}`
}

const rewriteHtmlAttributeUrls = (content: string, attribute: 'href' | 'src', assetBasePath?: string | null) => {
  if (!assetBasePath?.trim()) return content

  const pattern = new RegExp(`(${attribute}=")([^"]+)(")`, 'gi')
  return content.replace(pattern, (_match, prefix: string, value: string, suffix: string) => {
    const resolvedValue = resolveMarkdownAssetUrl(value, assetBasePath)
    return `${prefix}${resolvedValue}${suffix}`
  })
}

export const renderMarkdown = (content: string, options: RenderMarkdownOptions = {}) => {
  const rendered = renderer.render(content ?? '')
  const withResolvedLinks = rewriteHtmlAttributeUrls(rendered, 'href', options.assetBasePath)
  return rewriteHtmlAttributeUrls(withResolvedLinks, 'src', options.assetBasePath)
}
