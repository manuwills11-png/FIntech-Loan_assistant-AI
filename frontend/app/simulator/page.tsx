"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { useLanguage } from "@/lib/language-context"
import { usePageStrings } from "@/lib/use-page-strings"
import { AppWrapper } from "@/components/app-wrapper"
import { RiskGauge } from "@/components/risk-gauge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Sliders, TrendingUp, TrendingDown, Minus, ArrowRight, Info } from "lucide-react"

export default function SimulatorPage() {
  return (
    <AppWrapper>
      <CreditSimulator />
    </AppWrapper>
  )
}

function CreditSimulator() {
  const { t } = useLanguage()
  const ps = usePageStrings({
    desc: "Drag the sliders to simulate \"what if\" scenarios — see exactly how income, EMI, and debt affect your loan approval chances",
    adjustParams: "Adjust Parameters",
    moveSliders: "Move sliders to simulate different scenarios",
    safe: "Safe",
    moderate: "Moderate",
    highRisk: "High Risk",
    liveUpdating: "Live-updating based on your inputs",
    foirLabel: "FOIR (EMI to Income)",
    foirSafe: "✓ Within bank limit (≤ 40%)",
    foirBorderline: "⚠ Borderline — banks cap at 50%",
    foirExceeds: "✗ Exceeds bank FOIR limit of 50%",
    savingsBuffer: "Monthly Savings Buffer",
    healthyBuffer: "healthy buffer",
    thinMargin: "thin margin",
    negativeBuffer: "⚠ Negative — spending exceeds income",
    debtToIncome: "Debt to Annual Income",
    banksPrefer: "Banks prefer below 50% · Yours:",
    approxMaxEmi: "Approx. Max New EMI",
    foirCap: "Based on 50% FOIR cap · Conservatively at 40%:",
    whatThisMeans: "What this means for your loan application",
    foirSafeInsight: "is within the safe zone. Banks will comfortably approve a loan with EMI up to",
    foirBorderlineInsight: "is borderline. Some banks may approve with stricter terms. Try increasing income or reducing existing EMIs.",
    foirHighInsight: "exceeds most banks' 50% cap. Reduce existing debt or increase income before applying.",
    runAssessment: "Run Full Eligibility Assessment",
  }, "simulator")
  const [income, setIncome] = useState(50000)
  const [emi, setEmi] = useState(15000)
  const [expenses, setExpenses] = useState(20000)
  const [existingLoans, setExistingLoans] = useState(100000)
  const [riskScore, setRiskScore] = useState(0)

  useEffect(() => {
    // Calculate risk score based on inputs
    const emiRatio = income > 0 ? (emi / income) * 100 : 100
    const expenseRatio = income > 0 ? (expenses / income) * 100 : 100
    const loanBurden = income > 0 ? (existingLoans / (income * 12)) * 100 : 100

    let score = 100 - (emiRatio * 0.4 + expenseRatio * 0.3 + loanBurden * 0.1)
    score = Math.max(0, Math.min(100, score))

    setRiskScore(Math.round(100 - score))
  }, [income, emi, expenses, existingLoans])

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
      maximumFractionDigits: 0,
    }).format(value)
  }

  const getEmiStatus = () => {
    const ratio = (emi / income) * 100
    if (ratio <= 30) return { label: ps.safe, color: "bg-success text-success-foreground", icon: TrendingDown }
    if (ratio <= 50) return { label: ps.moderate, color: "bg-warning text-warning-foreground", icon: Minus }
    return { label: ps.highRisk, color: "bg-destructive text-destructive-foreground", icon: TrendingUp }
  }

  const getSavingsRate = () => {
    const savings = income - expenses - emi
    const rate = (savings / income) * 100
    return { amount: savings, rate: rate.toFixed(1) }
  }

  const emiStatus = getEmiStatus()
  const savings = getSavingsRate()
  const EmiIcon = emiStatus.icon

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-balance text-3xl font-bold tracking-tight text-foreground">
          {t("creditSimulator")}
        </h1>
        <p className="mt-2 text-muted-foreground">{ps.desc}</p>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Sliders */}
        <Card className="border-none shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sliders className="h-5 w-5 text-primary" />
              {ps.adjustParams}
            </CardTitle>
            <CardDescription>{ps.moveSliders}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-8">
            {/* Income Slider */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label className="text-base font-medium">{t("income")}</Label>
                <span className="text-lg font-semibold text-foreground">
                  {formatCurrency(income)}
                </span>
              </div>
              <Slider
                value={[income]}
                onValueChange={([value]) => setIncome(value)}
                min={10000}
                max={500000}
                step={5000}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{formatCurrency(10000)}</span>
                <span>{formatCurrency(500000)}</span>
              </div>
            </div>

            {/* EMI Slider */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Label className="text-base font-medium">{t("emi")}</Label>
                  <Badge className={emiStatus.color}>
                    <EmiIcon className="mr-1 h-3 w-3" />
                    {emiStatus.label}
                  </Badge>
                </div>
                <span className="text-lg font-semibold text-foreground">
                  {formatCurrency(emi)}
                </span>
              </div>
              <Slider
                value={[emi]}
                onValueChange={([value]) => setEmi(value)}
                min={0}
                max={Math.min(income * 0.8, 200000)}
                step={1000}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{formatCurrency(0)}</span>
                <span>{formatCurrency(Math.min(income * 0.8, 200000))}</span>
              </div>
            </div>

            {/* Expenses Slider */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label className="text-base font-medium">{t("expenses")}</Label>
                <span className="text-lg font-semibold text-foreground">
                  {formatCurrency(expenses)}
                </span>
              </div>
              <Slider
                value={[expenses]}
                onValueChange={([value]) => setExpenses(value)}
                min={5000}
                max={Math.min(income * 0.9, 300000)}
                step={1000}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{formatCurrency(5000)}</span>
                <span>{formatCurrency(Math.min(income * 0.9, 300000))}</span>
              </div>
            </div>

            {/* Existing Loans Slider */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label className="text-base font-medium">{t("existingLoans")}</Label>
                <span className="text-lg font-semibold text-foreground">
                  {formatCurrency(existingLoans)}
                </span>
              </div>
              <Slider
                value={[existingLoans]}
                onValueChange={([value]) => setExistingLoans(value)}
                min={0}
                max={2000000}
                step={10000}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{formatCurrency(0)}</span>
                <span>{formatCurrency(2000000)}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        <div className="space-y-6">
          <Card className="border-none shadow-lg">
            <CardHeader>
              <CardTitle>{t("riskScore")}</CardTitle>
              <CardDescription>{ps.liveUpdating}</CardDescription>
            </CardHeader>
            <CardContent className="flex justify-center py-4">
              <RiskGauge score={riskScore} />
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <div className="grid gap-4 sm:grid-cols-2">
            {/* FOIR */}
            <Card className="border-none shadow-md">
              <CardContent className="p-4">
                <p className="text-sm text-muted-foreground">{ps.foirLabel}</p>
                <p className={cn("text-2xl font-bold",
                  (emi / income) <= 0.4 ? "text-success" :
                  (emi / income) <= 0.5 ? "text-warning" : "text-destructive"
                )}>
                  {((emi / income) * 100).toFixed(1)}%
                </p>
                <p className={cn("text-xs font-medium mt-0.5",
                  (emi / income) <= 0.4 ? "text-success" :
                  (emi / income) <= 0.5 ? "text-warning" : "text-destructive"
                )}>
                  {(emi / income) <= 0.4 ? ps.foirSafe
                    : (emi / income) <= 0.5 ? ps.foirBorderline
                    : ps.foirExceeds}
                </p>
              </CardContent>
            </Card>

            <Card className="border-none shadow-md">
              <CardContent className="p-4">
                <p className="text-sm text-muted-foreground">{ps.savingsBuffer}</p>
                <p className={cn("text-2xl font-bold", savings.amount >= 0 ? "text-success" : "text-destructive")}>
                  {formatCurrency(Math.abs(savings.amount))}
                </p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {savings.amount >= 0
                    ? `${savings.rate}% of income — ${parseFloat(savings.rate) >= 20 ? ps.healthyBuffer : ps.thinMargin}`
                    : ps.negativeBuffer}
                </p>
              </CardContent>
            </Card>

            <Card className="border-none shadow-md">
              <CardContent className="p-4">
                <p className="text-sm text-muted-foreground">{ps.debtToIncome}</p>
                <p className="text-2xl font-bold text-foreground">
                  {((existingLoans / (income * 12)) * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {ps.banksPrefer} {((existingLoans / (income * 12)) * 100).toFixed(0)}%
                </p>
              </CardContent>
            </Card>

            {/* Max eligible loan */}
            <Card className="border-none shadow-md">
              <CardContent className="p-4">
                <p className="text-sm text-muted-foreground">{ps.approxMaxEmi}</p>
                <p className="text-2xl font-bold text-primary">
                  {formatCurrency(Math.max(0, income * 0.5 - emi))}
                </p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {ps.foirCap} {formatCurrency(Math.max(0, income * 0.4 - emi))}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Insight box */}
          <div className="rounded-xl border border-primary/20 bg-primary/5 p-4">
            <div className="flex items-start gap-2">
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
              <div className="space-y-1">
                <p className="text-sm font-semibold text-foreground">{ps.whatThisMeans}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {(emi / income) <= 0.4
                    ? `Your FOIR of ${((emi / income) * 100).toFixed(0)}% ${ps.foirSafeInsight} ${formatCurrency(Math.max(0, income * 0.4 - emi))} more.`
                    : (emi / income) <= 0.5
                    ? `Your FOIR of ${((emi / income) * 100).toFixed(0)}% ${ps.foirBorderlineInsight}`
                    : `Your FOIR of ${((emi / income) * 100).toFixed(0)}% ${ps.foirHighInsight}`
                  }
                </p>
              </div>
            </div>
          </div>

          <Link href="/loan-risk">
            <Button className="w-full gap-2" variant="outline">
              {ps.runAssessment} <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
