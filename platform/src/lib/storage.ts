export const readJson = <T>(key: string, fallbackValue: T): T => {
  try {
    const rawValue = window.localStorage.getItem(key)
    if (rawValue == null) {
      return fallbackValue
    }
    return JSON.parse(rawValue) as T
  } catch {
    return fallbackValue
  }
}

export const writeJson = (key: string, value: unknown) => {
  try {
    window.localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Ignore storage quota and privacy-mode failures.
  }
}
