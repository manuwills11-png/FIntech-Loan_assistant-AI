"use client"

import { useState } from "react"
import Link from "next/link"
import { useLanguage } from "@/lib/language-context"
import { usePageStrings } from "@/lib/use-page-strings"
import { T } from "@/lib/auto-translate"
import { AppWrapper } from "@/components/app-wrapper"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Spinner } from "@/components/ui/spinner"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import {
  Calendar, Lightbulb, TrendingUp, Calculator,
  Bell, RefreshCw, CheckCircle2, ArrowRight, IndianRupee,
} from "lucide-react"
import { generateRoadmap, toBackendLang, type RoadmapOutput } from "@/lib/api"
import { useAppStore } from "@/lib/app-store"

export default function RoadmapPage() {
  return <AppWrapper><FinancialRoadmap /></AppWrapper>
}

// ── Budget bar ────────────────────────────────────────────────────────────────

function BudgetBar({ income, expenses, currentEmi, newEmi }: {
  income: number; expenses: number; currentEmi: number; newEmi: number
}) {
  if (!income) return null
  const expPct    = Math.min(100, (expenses    / income) * 100)
  const currPct   = Math.min(100, (currentEmi  / income) * 100)
  const newPct    = Math.min(100, (newEmi      / income) * 100)
  const savePct   = Math.max(0,   100 - expPct - currPct - newPct)
  const totalObligations = expenses + currentEmi + newEmi
  const savings   = income - totalObligations

  return (
    <div className="space-y-3">
      <T as="p" className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Monthly Budget Allocation</T>
      {/* Stacked bar */}
      <div className="flex h-6 w-full overflow-hidden rounded-full">
        <div className="bg-destructive/70 transition-all" style={{ width: `${expPct}%` }} title="Expenses" />
        <div className="bg-warning/70 transition-all"    style={{ width: `${currPct}%` }} title="Existing EMIs" />
        <div className="bg-primary transition-all"      style={{ width: `${newPct}%` }}  title="New EMI" />
        <div className="bg-success/60 transition-all"   style={{ width: `${savePct}%` }} title="Savings" />
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {[
          { label: "Living Expenses", pct: expPct,  color: "bg-destructive/70", amount: expenses    },
          { label: "Existing EMIs",   pct: currPct, color: "bg-warning/70",     amount: currentEmi  },
          { label: "New Loan EMI",    pct: newPct,  color: "bg-primary",        amount: newEmi      },
          { label: "Net Savings",     pct: savePct, color: "bg-success/60",     amount: savings     },
        ].map(({ label, pct, color, amount }) => (
          <div key={label} className="flex items-center gap-2">
            <div className={cn("h-2.5 w-2.5 shrink-0 rounded-full", color)} />
            <div className="min-w-0">
              <T as="p" className="text-xs text-muted-foreground truncate">{label}</T>
              <p className={cn("text-sm font-semibold", amount < 0 ? "text-destructive" : "text-foreground")}>
                {amount < 0 ? "−" : ""}₹{Math.abs(amount).toLocaleString("en-IN")}
              </p>
              <p className="text-xs text-muted-foreground">{pct.toFixed(0)}%</p>
            </div>
          </div>
        ))}
      </div>
      {savings < 0 && (
        <p className="rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive font-medium">
          ⚠ Your obligations exceed income by ₹{Math.abs(savings).toLocaleString("en-IN")}. Banks will likely reject this loan or require a co-applicant.
        </p>
      )}
      {savings >= 0 && savings / income < 0.1 && (
        <p className="rounded-lg bg-warning/10 px-3 py-2 text-xs text-warning font-medium">
          ⚠ Savings buffer is only {((savings / income) * 100).toFixed(0)}% of income. Consider a smaller loan or longer tenure.
        </p>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

function FinancialRoadmap() {
  const { t, language } = useLanguage()
  const ps = usePageStrings({
    step4: "Step 4 of 4",
    desc: "A personalised month-by-month repayment schedule — so you know exactly what to expect",
    yourDetails: "Your Financial Details",
    enterDetails: "Enter Your Financial Details",
    close: "Close",
    edit: "Edit",
    incomeHint: "Net take-home salary every month",
    expensesHint: "Living costs excluding existing EMIs",
    existingLoansHint: "Total outstanding principal on current loans",
    emiHint: "Total of all current EMIs you pay monthly",
    newLoanAmount: "New Loan Amount",
    newLoanHint: "Amount you want to borrow",
    tenureLabel: "Loan Tenure (months)",
    tenureHint: "Longer tenure = lower EMI, higher total interest",
    generating: "Generating your roadmap…",
    repaymentSummary: "Your Repayment Summary",
    tenureHeader: "Tenure",
    monthByMonth: "Month-by-month schedule — principal, interest & balance",
    opening: "Opening",
    closing: "Closing",
    cleared: "CLEARED ✓",
    cutExpenses: "Cut Expenses",
    growIncome: "Grow Income",
    neverMiss: "Never miss an EMI",
    neverMissDesc: "Set WhatsApp reminders for your EMI due date — late payments hurt your CIBIL score.",
    setReminder: "Set EMI Reminder",
    incomeLabel: "Income:",
    loanLabel: "Loan:",
    tenureSummary: "Tenure:",
    months: "months",
  }, "roadmap")
  const { roadmap: storedRoadmap, setRoadmap: saveRoadmap } = useAppStore()
  const [isGenerating, setIsGenerating] = useState(false)
  const [roadmap, setRoadmap]   = useState<RoadmapOutput | null>(storedRoadmap.result)
  const [error, setError]       = useState<string | null>(null)
  const [formData, setFormData] = useState(storedRoadmap.formData)
  const [formOpen, setFormOpen] = useState(!storedRoadmap.result)

  const handleInputChange = (field: string, value: string) => {
    const updated = { ...formData, [field]: value }
    setFormData(updated)
    saveRoadmap({ formData: updated, result: null })
    setRoadmap(null)
  }

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsGenerating(true)
    setError(null)
    setRoadmap(null)
    try {
      const data = await generateRoadmap({
        monthly_income:         parseFloat(formData.income),
        monthly_expenses:       parseFloat(formData.expenses),
        existing_loans:         parseFloat(formData.existingLoans) || 0,
        emi_amount:             parseFloat(formData.emi) || 0,
        loan_amount_requested:  parseFloat(formData.loanAmount),
        loan_tenure_months:     parseInt(formData.tenure),
        language:               toBackendLang(language),
      })
      setRoadmap(data)
      saveRoadmap({ formData, result: data })
      setFormOpen(false)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to generate roadmap")
    } finally {
      setIsGenerating(false)
    }
  }

  const fmt = (v: number) =>
    new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(v)

  const income      = parseFloat(formData.income) || 0
  const expenses    = parseFloat(formData.expenses) || 0
  const currentEmi  = parseFloat(formData.emi) || 0
  const newEmi      = roadmap?.suggested_emi ?? 0

  return (
    <div className="space-y-8">

      {/* Header */}
      <div>
        <p className="mb-1 text-xs font-semibold uppercase tracking-widest text-muted-foreground">{ps.step4}</p>
        <h1 className="text-balance text-3xl font-bold tracking-tight text-foreground">
          {t("financialRoadmap")}
        </h1>
        <p className="mt-2 text-muted-foreground">{ps.desc}</p>
      </div>

      {/* ── Input form (collapsible once results are shown) ── */}
      <Card className="border-none shadow-lg">
        <CardHeader className="cursor-pointer" onClick={() => setFormOpen(v => !v)}>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Calculator className="h-5 w-5 text-primary" />
              {roadmap ? ps.yourDetails : ps.enterDetails}
            </CardTitle>
            {roadmap && (
              <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground">
                <RefreshCw className="h-3.5 w-3.5" />
                {formOpen ? ps.close : ps.edit}
              </Button>
            )}
          </div>
          {!formOpen && roadmap && (
            <div className="mt-1 flex flex-wrap gap-4 text-sm">
              {formData.income      && <span className="text-muted-foreground">{ps.incomeLabel} <strong className="text-foreground">₹{parseInt(formData.income).toLocaleString("en-IN")}</strong></span>}
              {formData.loanAmount  && <span className="text-muted-foreground">{ps.loanLabel} <strong className="text-foreground">₹{parseInt(formData.loanAmount).toLocaleString("en-IN")}</strong></span>}
              {formData.tenure      && <span className="text-muted-foreground">{ps.tenureSummary} <strong className="text-foreground">{formData.tenure} {ps.months}</strong></span>}
            </div>
          )}
        </CardHeader>

        {formOpen && (
          <CardContent>
            <form onSubmit={handleGenerate} className="grid gap-5 sm:grid-cols-2">
              {[
                { id: "income",        label: t("income"),             placeholder: "50000", hint: ps.incomeHint },
                { id: "expenses",      label: t("expenses"),           placeholder: "25000", hint: ps.expensesHint },
                { id: "existingLoans", label: t("existingLoans"),      placeholder: "100000", hint: ps.existingLoansHint },
                { id: "emi",           label: t("emi"),                placeholder: "10000", hint: ps.emiHint },
                { id: "loanAmount",    label: ps.newLoanAmount,        placeholder: "200000", hint: ps.newLoanHint },
              ].map(({ id, label, placeholder, hint }) => (
                <div key={id} className="space-y-1.5">
                  <Label htmlFor={id} className="text-base font-medium">{label}</Label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">₹</span>
                    <Input
                      id={id} type="number" placeholder={placeholder}
                      value={formData[id as keyof typeof formData]}
                      onChange={e => handleInputChange(id, e.target.value)}
                      className="h-12 pl-8 text-lg"
                      required={id !== "existingLoans" && id !== "emi"}
                      min={0}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">{hint}</p>
                </div>
              ))}

              <div className="space-y-1.5">
                <Label htmlFor="tenure" className="text-base font-medium">{ps.tenureLabel}</Label>
                <Input id="tenure" type="number" placeholder="60"
                  value={formData.tenure} onChange={e => handleInputChange("tenure", e.target.value)}
                  className="h-12 text-lg" required min={1} />
                <p className="text-xs text-muted-foreground">{ps.tenureHint}</p>
              </div>

              {error && (
                <div className="col-span-full rounded-lg bg-destructive/10 px-4 py-2 text-sm text-destructive">{error}</div>
              )}

              <div className="col-span-full">
                <Button type="submit" className="h-14 w-full text-lg font-semibold" disabled={isGenerating}>
                  {isGenerating
                    ? <><Spinner className="mr-2 h-5 w-5" /> {ps.generating}</>
                    : t("generatePlan")
                  }
                </Button>
              </div>
            </form>
          </CardContent>
        )}
      </Card>

      {/* ── Results ── */}
      {roadmap && (
        <>
          {/* Summary hero */}
          <Card className="border-none bg-primary text-primary-foreground shadow-lg">
            <CardContent className="p-6">
              <h3 className="mb-2 text-lg font-semibold">{ps.repaymentSummary}</h3>
              <p className="mb-5 text-sm leading-relaxed text-primary-foreground/90">{roadmap.summary}</p>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <div>
                  <p className="text-xs text-primary-foreground/70">{t("monthlyEmi")}</p>
                  <p className="text-2xl font-bold">{fmt(roadmap.suggested_emi)}</p>
                </div>
                <div>
                  <p className="text-xs text-primary-foreground/70">{t("totalInterest")}</p>
                  <p className="text-2xl font-bold">{fmt(roadmap.total_interest_payable)}</p>
                </div>
                <div>
                  <p className="text-xs text-primary-foreground/70">{t("totalPayable")}</p>
                  <p className="text-2xl font-bold">
                    {fmt(parseFloat(formData.loanAmount) + roadmap.total_interest_payable)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-primary-foreground/70">{ps.tenureHeader}</p>
                  <p className="text-2xl font-bold">{roadmap.repayment_plan.length} months</p>
                  <p className="text-xs text-primary-foreground/60">
                    {Math.floor(roadmap.repayment_plan.length / 12)} yr{Math.floor(roadmap.repayment_plan.length / 12) !== 1 ? "s" : ""}
                    {roadmap.repayment_plan.length % 12 > 0 ? ` ${roadmap.repayment_plan.length % 12} mo` : ""}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Budget allocation */}
          {income > 0 && newEmi > 0 && (
            <Card className="border-none shadow-sm">
              <CardContent className="p-5">
                <BudgetBar
                  income={income}
                  expenses={expenses}
                  currentEmi={currentEmi}
                  newEmi={newEmi}
                />
              </CardContent>
            </Card>
          )}

          <div className="grid gap-8 lg:grid-cols-3">

            {/* Repayment Timeline */}
            <div className="lg:col-span-2">
              <Card className="border-none shadow-lg">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Calendar className="h-5 w-5 text-primary" />
                    {t("repaymentPlan")}
                  </CardTitle>
                  <CardDescription>{ps.monthByMonth}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="max-h-96 overflow-y-auto space-y-1.5 pr-1">
                    {roadmap.repayment_plan.map((item) => {
                      const isLast = item.closing_balance <= 0
                      return (
                        <div key={item.month}
                          className={cn(
                            "grid grid-cols-4 items-center gap-2 rounded-lg px-3 py-2.5 text-sm",
                            isLast ? "bg-success/10 border border-success/20" : "bg-muted"
                          )}
                        >
                          <div className="flex items-center gap-2">
                            {isLast
                              ? <CheckCircle2 className="h-4 w-4 shrink-0 text-success" />
                              : <Badge variant="secondary" className="text-xs">Mo {item.month}</Badge>
                            }
                          </div>
                          <div className="text-center">
                            <p className="text-xs text-muted-foreground">{ps.opening}</p>
                            <p className="font-medium">{fmt(item.opening_balance)}</p>
                          </div>
                          <div className="text-center">
                            <p className="text-xs text-muted-foreground">{t("emi")}</p>
                            <p className="font-medium text-destructive">−{fmt(item.emi_payment)}</p>
                          </div>
                          <div className="text-center">
                            <p className="text-xs text-muted-foreground">{ps.closing}</p>
                            <p className={cn("font-medium", isLast ? "text-success font-bold" : "text-foreground")}>
                              {isLast ? ps.cleared : fmt(Math.max(0, item.closing_balance))}
                            </p>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Tips sidebar */}
            <div className="space-y-6">
              {roadmap.expense_reduction_tips.length > 0 && (
                <Card className="border-none shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Lightbulb className="h-5 w-5 text-warning" />
                      {ps.cutExpenses}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2.5">
                    {roadmap.expense_reduction_tips.map((tip, i) => (
                      <div key={i} className="flex gap-2 rounded-lg bg-muted/50 px-3 py-2.5">
                        <span className="mt-0.5 shrink-0 text-warning">•</span>
                        <p className="text-sm text-foreground leading-relaxed">{tip}</p>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              {roadmap.income_improvement_tips.length > 0 && (
                <Card className="border-none shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5 text-success" />
                      {ps.growIncome}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2.5">
                    {roadmap.income_improvement_tips.map((tip, i) => (
                      <div key={i} className="flex gap-2 rounded-lg bg-muted/50 px-3 py-2.5">
                        <span className="mt-0.5 shrink-0 text-success">•</span>
                        <p className="text-sm text-foreground leading-relaxed">{tip}</p>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* Reminder CTA */}
              <Card className="border-none bg-primary/5 shadow-sm">
                <CardContent className="p-4">
                  <Bell className="mb-2 h-5 w-5 text-primary" />
                  <p className="text-sm font-semibold text-foreground">{ps.neverMiss}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{ps.neverMissDesc}</p>
                  <Link href="/reminders">
                    <Button variant="outline" size="sm" className="mt-3 w-full gap-2">
                      {ps.setReminder} <ArrowRight className="h-3 w-3" />
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
