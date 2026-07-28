"use client"

/**
 * Auto-translation system.
 *
 * <T>Any hardcoded English text here</T>
 *
 * - Shows English text immediately (no flicker)
 * - Translates async on language change
 * - Caches results in localStorage (per language)
 * - Batches all requests on a page into ONE API call per language change
 */

import {
  createContext,
  useContext,
  useEffect,
  useState,
  useRef,
  type ReactNode,
  type ElementType,
  type ComponentPropsWithoutRef,
} from "react"
import { useLanguage } from "@/lib/language-context"

// ── Cache helpers ─────────────────────────────────────────────────────────────

const CACHE_KEY = (lang: string) => `finedge_trans_${lang}`

function loadCache(lang: string): Record<string, string> {
  if (typeof window === "undefined") return {}
  try {
    return JSON.parse(localStorage.getItem(CACHE_KEY(lang)) || "{}")
  } catch {
    return {}
  }
}

function saveCache(lang: string, cache: Record<string, string>) {
  if (typeof window === "undefined") return
  localStorage.setItem(CACHE_KEY(lang), JSON.stringify(cache))
}

// ── Global batcher ────────────────────────────────────────────────────────────
// Collects all pending texts for the current frame, then fires one batch request.

type Subscriber = (translated: string) => void

interface PendingEntry {
  text: string
  subscribers: Subscriber[]
}

const pending = new Map<string, PendingEntry>()  // key = english text
let batchTimer: ReturnType<typeof setTimeout> | null = null
let currentLanguage = "en"
let activeCache: Record<string, string> = {}

function flushBatch() {
  batchTimer = null
  const lang = currentLanguage
  if (lang === "en") return

  const uncached: string[] = []
  for (const [text] of pending) {
    if (!activeCache[text]) uncached.push(text)
  }

  if (uncached.length === 0) {
    // All cached — just notify subscribers
    for (const [text, entry] of pending) {
      entry.subscribers.forEach(fn => fn(activeCache[text] || text))
    }
    pending.clear()
    return
  }

  const attemptFetch = (attempt: number) => {
    fetch("/api/translate/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: uncached, target_language: lang }),
    })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(data => {
        const translations: string[] = data.translations || []
        uncached.forEach((text, i) => {
          activeCache[text] = translations[i] || text
        })
        saveCache(lang, activeCache)
        for (const [text, entry] of pending) {
          entry.subscribers.forEach(fn => fn(activeCache[text] || text))
        }
        pending.clear()
      })
      .catch(() => {
        if (attempt < 2) {
          // Retry up to 2 times with backoff (500ms, 1500ms)
          setTimeout(() => attemptFetch(attempt + 1), 500 * (attempt + 1))
        } else {
          // Give up — fall back to English
          for (const [text, entry] of pending) {
            entry.subscribers.forEach(fn => fn(text))
          }
          pending.clear()
        }
      })
  }
  attemptFetch(0)
}

function requestTranslation(text: string, lang: string, callback: Subscriber) {
  if (lang === "en") {
    callback(text)
    return
  }

  // Check cache
  const cached = activeCache[text]
  if (cached) {
    callback(cached)
    return
  }

  // Enqueue
  if (!pending.has(text)) {
    pending.set(text, { text, subscribers: [] })
  }
  pending.get(text)!.subscribers.push(callback)

  if (!batchTimer) {
    batchTimer = setTimeout(flushBatch, 50)  // 50ms debounce — collect all on page
  }
}

// ── Language change coordinator ───────────────────────────────────────────────

type LangChangeListener = (lang: string) => void
const langListeners = new Set<LangChangeListener>()

export function notifyLanguageChange(lang: string) {
  currentLanguage = lang
  activeCache = loadCache(lang)
  langListeners.forEach(fn => fn(lang))
}

// ── <T> component ─────────────────────────────────────────────────────────────

interface TProps {
  children: string
  as?: ElementType
  className?: string
  [key: string]: any
}

export function T({ children, as: Tag = "span", className, ...rest }: TProps) {
  const { language } = useLanguage()
  const [translated, setTranslated] = useState(children)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])

  useEffect(() => {
    if (language === "en") {
      setTranslated(children)
      return
    }
    activeCache = loadCache(language)
    currentLanguage = language
    requestTranslation(children, language, (result) => {
      if (mountedRef.current) setTranslated(result)
    })
  }, [children, language])

  // Re-translate when global language changes
  useEffect(() => {
    const handler = (lang: string) => {
      if (lang === "en") {
        setTranslated(children)
        return
      }
      activeCache = loadCache(lang)
      requestTranslation(children, lang, (result) => {
        if (mountedRef.current) setTranslated(result)
      })
    }
    langListeners.add(handler)
    return () => { langListeners.delete(handler) }
  }, [children])

  const Component = Tag as ElementType<ComponentPropsWithoutRef<"span">>
  return <Component className={className} {...rest}>{translated}</Component>
}

// ── useTrans hook (for attributes like placeholder, title, aria-label) ────────

export function useTrans(text: string): string {
  const { language } = useLanguage()
  const [translated, setTranslated] = useState(text)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])

  useEffect(() => {
    if (language === "en") {
      setTranslated(text)
      return
    }
    activeCache = loadCache(language)
    currentLanguage = language
    requestTranslation(text, language, (result) => {
      if (mountedRef.current) setTranslated(result)
    })
  }, [text, language])

  useEffect(() => {
    const handler = (lang: string) => {
      if (lang === "en") { setTranslated(text); return }
      activeCache = loadCache(lang)
      requestTranslation(text, lang, r => {
        if (mountedRef.current) setTranslated(r)
      })
    }
    langListeners.add(handler)
    return () => { langListeners.delete(handler) }
  }, [text])

  return translated
}

/** Plain text from the API or dynamic data — translated to the active UI language. */
export function DynamicText({
  text,
  className,
  as: Tag = "span",
}: {
  text: string
  className?: string
  as?: ElementType
}) {
  const translated = useTrans(text)
  const Component = Tag as ElementType<ComponentPropsWithoutRef<"span">>
  return <Component className={className}>{translated}</Component>
}
