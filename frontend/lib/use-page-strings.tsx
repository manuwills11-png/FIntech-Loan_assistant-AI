"use client"

/**
 * usePageStrings — translate a record of English strings as one batch.
 *
 * Usage:
 *   const s = usePageStrings({
 *     title: "My Page Title",
 *     desc:  "Some description here",
 *   })
 *   // then: <h1>{s.title}</h1>  <p>{s.desc}</p>
 *
 * Results are cached in localStorage per language.
 */

import { useState, useEffect, useRef, useMemo } from "react"
import { useLanguage } from "@/lib/language-context"

const CACHE_PREFIX = "finedge_pstr_"

function cacheKey(lang: string, pageKey: string) {
  return `${CACHE_PREFIX}${lang}_${pageKey}`
}

export function usePageStrings<T extends Record<string, string>>(
  strings: T,
  pageKey: string,  // unique identifier for this page's string set
): T {
  const { language } = useLanguage()
  const [translated, setTranslated] = useState<T>(strings)
  const mountedRef = useRef(true)
  const stringsRef = useRef(strings)
  stringsRef.current = strings
  const stringsSig = useMemo(() => JSON.stringify(strings), [strings])

  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])

  useEffect(() => {
    const s = stringsRef.current
    if (language === "en") {
      setTranslated(s)
      return
    }

    const ck = cacheKey(language, pageKey)
    try {
      const raw = localStorage.getItem(ck)
      if (raw) {
        const parsed = JSON.parse(raw) as { sig?: string; data?: T } | T | null
        if (parsed && typeof parsed === "object" && "sig" in parsed && "data" in parsed) {
          const wrapped = parsed as { sig: string; data: T }
          if (wrapped.sig === stringsSig && wrapped.data) {
            setTranslated(wrapped.data)
            return
          }
        }
      }
    } catch {}

    const keys = Object.keys(s) as (keyof T)[]
    const texts = keys.map(k => s[k])

    const attemptFetch = (attempt: number) => {
      fetch("/api/translate/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts, target_language: language }),
      })
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`)
          return r.json()
        })
        .then(data => {
          const result = { ...s }
          keys.forEach((k, i) => {
            result[k] = (data.translations?.[i] || s[k]) as T[keyof T]
          })
          try {
            localStorage.setItem(ck, JSON.stringify({ sig: stringsSig, data: result }))
          } catch {}
          if (mountedRef.current) setTranslated(result)
        })
        .catch(() => {
          if (attempt < 2) {
            setTimeout(() => attemptFetch(attempt + 1), 500 * (attempt + 1))
          } else {
            if (mountedRef.current) setTranslated(s)
          }
        })
    }
    attemptFetch(0)
  }, [language, pageKey, stringsSig])

  return translated
}
