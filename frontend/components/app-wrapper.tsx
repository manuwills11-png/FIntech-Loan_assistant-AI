"use client"

import { type ReactNode, type ReactElement } from "react"
import { LanguageProvider } from "@/lib/language-context"
import { AppStoreProvider } from "@/lib/app-store"
import { Navigation } from "@/components/navigation"
import { T } from "@/lib/auto-translate"
import { cn } from "@/lib/utils"

interface AppWrapperProps {
  children: ReactNode
}

export function AppWrapper({ children }: AppWrapperProps) {
  return (
    <AppStoreProvider>
    <LanguageProvider>
      <div className="min-h-screen bg-background">
        <Navigation />
        <main className="pb-20 md:ml-64 md:pb-8">
          <div className="mx-auto max-w-5xl p-4 md:p-8">
            {children}
          </div>
        </main>

        {/* Mobile Bottom Navigation */}
        <MobileBottomNav />
      </div>
    </LanguageProvider>
    </AppStoreProvider>
  )
}

function MobileBottomNav() {
  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 flex items-center justify-around border-t border-border bg-card px-2 py-3 md:hidden">
      <NavItem href="/" icon="home" label={<T>Home</T>} />
      <NavItem href="/loan-risk" icon="calculator" label={<T>Risk</T>} />
      <NavItem href="/chat" icon="chat" label={<T>Chat</T>} isPrimary />
      <NavItem href="/documents" icon="file" label={<T>Docs</T>} />
      <NavItem href="/roadmap" icon="map" label={<T>Plan</T>} />
    </nav>
  )
}

function NavItem({
  href,
  icon,
  label,
  isPrimary = false,
}: {
  href: string
  icon: string
  label: ReactNode
  isPrimary?: boolean
}) {
  const iconMap: Record<string, ReactElement> = {
    home: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    ),
    calculator: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
      </svg>
    ),
    chat: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
      </svg>
    ),
    file: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    map: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
      </svg>
    ),
  }

  return (
    <a
      href={href}
      className={cn(
        "flex flex-col items-center gap-1 transition-colors",
        isPrimary
          ? "relative -mt-6 rounded-full bg-primary p-4 text-primary-foreground shadow-lg"
          : "text-muted-foreground hover:text-foreground"
      )}
    >
      {iconMap[icon]}
      {!isPrimary && <span className="text-xs">{label}</span>}
    </a>
  )
}
