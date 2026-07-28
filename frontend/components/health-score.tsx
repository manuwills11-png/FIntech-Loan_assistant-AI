"use client"

import { cn } from "@/lib/utils"
import { useLanguage } from "@/lib/language-context"

interface HealthScoreProps {
  score: number
  className?: string
}

export function HealthScore({ score, className }: HealthScoreProps) {
  const { t } = useLanguage()
  
  const getScoreColor = () => {
    if (score >= 70) return "from-success to-accent"
    if (score >= 40) return "from-warning to-chart-4"
    return "from-destructive to-chart-5"
  }

  const getScoreLabel = () => {
    if (score >= 70) return "Excellent"
    if (score >= 40) return "Good"
    return "Needs Improvement"
  }

  const circumference = 2 * Math.PI * 45
  const strokeDashoffset = circumference - (score / 100) * circumference

  return (
    <div className={cn("flex flex-col items-center gap-3", className)}>
      <div className="relative h-32 w-32">
        <svg className="h-full w-full -rotate-90" viewBox="0 0 100 100">
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            className="text-muted"
          />
          {/* Progress circle */}
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="url(#scoreGradient)"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-1000 ease-out"
          />
          <defs>
            <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="oklch(0.55 0.18 230)" />
              <stop offset="100%" stopColor="oklch(0.65 0.14 160)" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold text-foreground">{score}</span>
          <span className="text-xs text-muted-foreground">/100</span>
        </div>
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-foreground">{t("financialHealth")}</p>
        <p className="text-xs text-muted-foreground">{getScoreLabel()}</p>
      </div>
    </div>
  )
}
