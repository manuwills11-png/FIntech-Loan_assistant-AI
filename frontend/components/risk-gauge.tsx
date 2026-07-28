"use client"

import { cn } from "@/lib/utils"
import { useLanguage } from "@/lib/language-context"

interface RiskGaugeProps {
  score: number
  className?: string
}

export function RiskGauge({ score, className }: RiskGaugeProps) {
  const { t } = useLanguage()
  
  const getRiskLevel = () => {
    if (score <= 30) return { level: "lowRisk", color: "text-success" }
    if (score <= 60) return { level: "mediumRisk", color: "text-warning" }
    return { level: "highRisk", color: "text-destructive" }
  }

  const risk = getRiskLevel()
  const rotation = (score / 100) * 180 - 90

  return (
    <div className={cn("flex flex-col items-center gap-4", className)}>
      <div className="relative h-32 w-64 overflow-hidden">
        {/* Background Arc */}
        <div className="absolute inset-0">
          <svg viewBox="0 0 200 100" className="h-full w-full">
            {/* Background gradient arc */}
            <defs>
              <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="oklch(0.65 0.14 160)" />
                <stop offset="50%" stopColor="oklch(0.75 0.15 80)" />
                <stop offset="100%" stopColor="oklch(0.577 0.245 27.325)" />
              </linearGradient>
            </defs>
            <path
              d="M 10 100 A 90 90 0 0 1 190 100"
              fill="none"
              stroke="oklch(0.9 0.01 240)"
              strokeWidth="16"
              strokeLinecap="round"
            />
            <path
              d="M 10 100 A 90 90 0 0 1 190 100"
              fill="none"
              stroke="url(#gaugeGradient)"
              strokeWidth="16"
              strokeLinecap="round"
              strokeDasharray={`${(score / 100) * 283} 283`}
            />
          </svg>
        </div>

        {/* Needle */}
        <div
          className="absolute bottom-0 left-1/2 h-24 w-1 origin-bottom -translate-x-1/2 transition-transform duration-700 ease-out"
          style={{ transform: `translateX(-50%) rotate(${rotation}deg)` }}
        >
          <div className="h-full w-full rounded-full bg-foreground" />
          <div className="absolute -bottom-2 -left-2 h-5 w-5 rounded-full bg-foreground" />
        </div>
      </div>

      <div className="text-center">
        <div className={cn("text-4xl font-bold", risk.color)}>{score}</div>
        <div className={cn("text-lg font-medium", risk.color)}>{t(risk.level)}</div>
      </div>
    </div>
  )
}
