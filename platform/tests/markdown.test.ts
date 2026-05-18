import { describe, expect, it } from 'vitest'

import { renderMarkdown, resolveMarkdownAssetUrl } from '../src/lib/markdown'

describe('markdown asset resolution', () => {
  it('resolves relative image paths against a candidate report path', () => {
    expect(
      resolveMarkdownAssetUrl(
        'figures/model.png',
        '/evaluation-data/benchmarks/demo/cases/0001/frameworks/alpha/files/report.md',
      ),
    ).toBe(
      '/evaluation-data/benchmarks/demo/cases/0001/frameworks/alpha/files/figures/model.png',
    )
  })

  it('resolves nested sandbox paths against the report location', () => {
    expect(
      resolveMarkdownAssetUrl(
        'sandbox/results/figures/feature.png',
        '/evaluation-data/benchmarks/demo/cases/0001/frameworks/alpha/files/report.md',
      ),
    ).toBe('/evaluation-data/benchmarks/demo/cases/0001/frameworks/alpha/files/sandbox/results/figures/feature.png')
  })

  it('leaves absolute and anchored urls unchanged', () => {
    expect(
      resolveMarkdownAssetUrl(
        'https://example.com/chart.png',
        '/evaluation-data/benchmarks/demo/cases/0001/frameworks/alpha/files/report.md',
      ),
    ).toBe('https://example.com/chart.png')
    expect(resolveMarkdownAssetUrl('#section-2', '/evaluation-data/benchmarks/demo/cases/0001/frameworks/alpha/files/report.md')).toBe(
      '#section-2',
    )
  })

  it('rewrites relative markdown image and link urls in rendered html', () => {
    const rendered = renderMarkdown('![plot](figures/model.png)\n\n[figure](tables/metrics.csv)', {
      assetBasePath: '/evaluation-data/benchmarks/demo/cases/0001/frameworks/alpha/files/report.md',
    })

    expect(rendered).toContain(
      'src="/evaluation-data/benchmarks/demo/cases/0001/frameworks/alpha/files/figures/model.png"',
    )
    expect(rendered).toContain(
      'href="/evaluation-data/benchmarks/demo/cases/0001/frameworks/alpha/files/tables/metrics.csv"',
    )
  })

  it('renders bracket-delimited formulas with katex output', () => {
    const rendered = renderMarkdown('The decay is \\(\\beta_n\\).\n\n\\[\nz_n = x_n + y_n\n\\]')

    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('β')
  })

  it('renders dollar-delimited formulas with katex output', () => {
    const rendered = renderMarkdown('The score is $x^2 + y^2$.')

    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('x')
  })
})
