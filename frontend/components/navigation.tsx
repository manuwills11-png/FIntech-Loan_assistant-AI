"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { useLanguage, languageNames, type Language } from "@/lib/language-context"
import { useTrans } from "@/lib/auto-translate"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Home,
  MessageCircle,
  FileText,
  Calculator,
  Map,
  Bell,
  Globe,
  Menu,
  X,
  ChevronDown,
  Building2,
  Coins,
  Landmark,
  Handshake,
} from "lucide-react"

const navItems = [
  { href: "/", icon: Home, labelKey: "dashboard" },
  { href: "/loan-risk", icon: Calculator, labelKey: "checkLoanRisk" },
  { href: "/chat", icon: MessageCircle, labelKey: "chat" },
  { href: "/bank-rates", icon: Building2, labelKey: "bankRates" },
  { href: "/gold-loan", icon: Coins, labelKey: "goldLoan" },
  { href: "/govt-schemes", icon: Landmark, labelKey: "govtSchemes" },
  { href: "/negotiate", icon: Handshake, labelKey: "negotiate" },
  { href: "/documents", icon: FileText, labelKey: "documents" },
  { href: "/simulator", icon: Calculator, labelKey: "simulator" },
  { href: "/roadmap", icon: Map, labelKey: "roadmap" },
  { href: "/reminders", icon: Bell, labelKey: "reminders" },
]

export function Navigation() {
  const [isOpen, setIsOpen] = useState(false)
  const pathname = usePathname()
  const { language, setLanguage, t } = useLanguage()

  return (
    <>
      {/* Mobile Header */}
      <header className="sticky top-0 z-50 flex items-center justify-between border-b border-border bg-card px-4 py-3 md:hidden">
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <span className="text-sm font-bold text-primary-foreground">F</span>
          </div>
          <span className="text-lg font-semibold text-foreground">FinAI</span>
        </Link>
        <div className="flex items-center gap-2">
          <LanguageSelector language={language} setLanguage={setLanguage} t={t} />
          <MenuToggleButton isOpen={isOpen} onClick={() => setIsOpen(!isOpen)} />
        </div>
      </header>

      {/* Mobile Navigation Overlay */}
      {isOpen && (
        <div className="fixed inset-0 top-14 z-40 bg-background md:hidden">
          <nav className="flex flex-col gap-2 p-4">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setIsOpen(false)}
                  className={cn(
                    "flex items-center gap-3 rounded-xl px-4 py-3 text-base font-medium transition-colors",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  <Icon className="h-5 w-5" />
                  {t(item.labelKey)}
                </Link>
              )
            })}
          </nav>
        </div>
      )}

      {/* Desktop Sidebar */}
      <aside className="fixed left-0 top-0 hidden h-screen w-64 flex-col border-r border-border bg-card p-4 md:flex">
        <Link href="/" className="mb-8 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
            <span className="text-lg font-bold text-primary-foreground">F</span>
          </div>
          <span className="text-xl font-semibold text-foreground">FinAI</span>
        </Link>

        <nav className="flex flex-1 flex-col gap-1">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.href
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
              >
                <Icon className="h-5 w-5" />
                {t(item.labelKey)}
              </Link>
            )
          })}
        </nav>

        <div className="border-t border-border pt-4">
          <LanguageSelector language={language} setLanguage={setLanguage} t={t} />
        </div>
      </aside>
    </>
  )
}

function MenuToggleButton({ isOpen, onClick }: { isOpen: boolean; onClick: () => void }) {
  const ariaLabel = useTrans("Toggle menu")
  return (
    <Button variant="ghost" size="icon" onClick={onClick} aria-label={ariaLabel}>
      {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
    </Button>
  )
}

function LanguageSelector({
  language,
  setLanguage,
  t,
}: {
  language: Language
  setLanguage: (lang: Language) => void
  t: (key: string) => string
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" className="flex items-center gap-2">
          <Globe className="h-4 w-4" />
          <span className="hidden sm:inline">{languageNames[language]}</span>
          <ChevronDown className="h-3 w-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {(Object.keys(languageNames) as Language[]).map((lang) => (
          <DropdownMenuItem
            key={lang}
            onClick={() => setLanguage(lang)}
            className={cn(language === lang && "bg-muted")}
          >
            {languageNames[lang]}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
