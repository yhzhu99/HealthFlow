const baseUrl = import.meta.env.BASE_URL || '/'

export const toBasePath = (path: string) => {
  if (!path) return baseUrl
  if (/^(?:[a-z]+:)?\/\//i.test(path)) return path

  const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`
  const normalizedPath = path.startsWith('/') ? path.slice(1) : path
  return `${normalizedBase}${normalizedPath}`
}
