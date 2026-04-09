import MarkdownIt from 'markdown-it'

const renderer = new MarkdownIt({
  breaks: true,
  html: false,
  linkify: true,
})

export const renderMarkdown = (content: string) => renderer.render(content ?? '')
