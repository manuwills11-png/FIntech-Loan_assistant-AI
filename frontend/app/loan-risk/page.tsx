"use client"

import { useState, useEffect, useRef } from "react"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { useLanguage } from "@/lib/language-context"
import { AppWrapper } from "@/components/app-wrapper"
import { RiskGauge } from "@/components/risk-gauge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Spinner } from "@/components/ui/spinner"
import { cn } from "@/lib/utils"
import {
  CheckCircle2, AlertTriangle, XCircle, Calculator,
  TrendingUp, Lightbulb, ChevronDown, ChevronUp,
  Building2, Map, FileText, ArrowRight, Info,
} from "lucide-react"
import { predictRisk, toBackendLang, type LoanRiskOutput, type FactorScore } from "@/lib/api"
import { useAppStore } from "@/lib/app-store"
import { T, useTrans, DynamicText } from "@/lib/auto-translate"

export default function LoanRiskPage() {
  return <AppWrapper><LoanRiskForm /></AppWrapper>
}

// ── Factor bar ────────────────────────────────────────────────────────────────

function FactorBar({ factor }: { factor: FactorScore }) {
  const colorClass  = factor.status === "good" ? "bg-success"     : factor.status === "fair" ? "bg-warning"     : "bg-destructive"
  const textClass   = factor.status === "good" ? "text-success"   : factor.status === "fair" ? "text-warning"   : "text-destructive"
  const badgeClass  = factor.status === "good" ? "bg-success/10 text-success" : factor.status === "fair" ? "bg-warning/10 text-warning" : "bg-destructive/10 text-destructive"
  const label = useTrans(factor.label)
  const statusLabel = useTrans(factor.status)
  const weightLine = useTrans(`${factor.weight}% of overall score`)

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-foreground">{label}</span>
        <div className="flex items-center gap-2">
          <span className={cn("rounded-full px-2 py-0.5 text-xs font-semibold capitalize", badgeClass)}>
            {statusLabel}
          </span>
          <span className={cn("w-8 text-right font-bold", textClass)}>{factor.score.toFixed(0)}</span>
        </div>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full transition-all duration-700", colorClass)} style={{ width: `${factor.score}%` }} />
      </div>
      <p className="text-xs text-muted-foreground">{weightLine}</p>
    </div>
  )
}

function TranslatedOption({ value, children }: { value: string; children: string }) {
  const label = useTrans(children)
  return <option value={value}>{label}</option>
}

// ── Helper text component ─────────────────────────────────────────────────────

function FieldHint({ children }: { children: React.ReactNode }) {
  return (
    <p className="flex items-start gap-1 text-xs text-muted-foreground">
      <Info className="mt-0.5 h-3 w-3 shrink-0 opacity-60" />
      {children}
    </p>
  )
}

// ── Section divider ───────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="border-t border-border pt-5">
      <p className="mb-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">{children}</p>
    </div>
  )
}

// ── Main form ─────────────────────────────────────────────────────────────────

function LoanRiskForm() {
  const { t, language } = useLanguage()
  const { loanRisk, setLoanRisk } = useAppStore()
  const searchParams  = useSearchParams()
  const [isCalculating, setIsCalculating] = useState(false)
  const [result, setResult]   = useState<LoanRiskOutput | null>(loanRisk.result)
  const [error, setError]     = useState<string | null>(null)
  const submitRef = useRef<HTMLButtonElement>(null)

  const urlFormData = {
    income:         searchParams.get("income")         ?? loanRisk.formData.income,
    expenses:       searchParams.get("expenses")       ?? loanRisk.formData.expenses,
    existingLoans:  searchParams.get("existingLoans")  ?? loanRisk.formData.existingLoans,
    emi:            searchParams.get("emi")            ?? loanRisk.formData.emi,
    loanAmount:     searchParams.get("loanAmount")     ?? loanRisk.formData.loanAmount,
    tenure:         searchParams.get("tenure")         ?? loanRisk.formData.tenure,
    cibilScore:     searchParams.get("cibilScore")     ?? loanRisk.formData.cibilScore    ?? "",
    age:            searchParams.get("age")            ?? loanRisk.formData.age           ?? "",
    loanPurpose:    searchParams.get("loanPurpose")    ?? loanRisk.formData.loanPurpose   ?? "",
    employmentType: searchParams.get("employmentType") ?? loanRisk.formData.employmentType ?? "salaried",
    stabilityYears: searchParams.get("stabilityYears") ?? loanRisk.formData.stabilityYears ?? "",
    coApplicantIncome:         searchParams.get("coApplicantIncome")         ?? (loanRisk.formData as any).coApplicantIncome         ?? "",
    coApplicantEmploymentType: searchParams.get("coApplicantEmploymentType") ?? (loanRisk.formData as any).coApplicantEmploymentType ?? "",
    coApplicantCibilScore:     searchParams.get("coApplicantCibilScore")     ?? (loanRisk.formData as any).coApplicantCibilScore     ?? "",
  }
  const [formData, setFormData]       = useState(urlFormData)
  const [showAllFactors, setShowAllFactors] = useState(false)

  useEffect(() => {
    if (searchParams.get("autosubmit") === "1" && formData.income && formData.loanAmount && formData.tenure) {
      setTimeout(() => submitRef.current?.click(), 300)
    }
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsCalculating(true)
    setError(null)
    setResult(null)
    try {
      const data = await predictRisk({
        monthly_income:              parseFloat(formData.income),
        monthly_expenses:            parseFloat(formData.expenses),
        existing_loans:              parseFloat(formData.existingLoans) || 0,
        emi_amount:                  parseFloat(formData.emi) || 0,
        loan_amount_requested:       parseFloat(formData.loanAmount),
        loan_tenure_months:          parseInt(formData.tenure),
        employment_type:             formData.employmentType || "salaried",
        language:                    toBackendLang(language),
        cibil_score:                 formData.cibilScore ? parseInt(formData.cibilScore) : undefined,
        age:                         formData.age ? parseInt(formData.age) : undefined,
        loan_purpose:                formData.loanPurpose || undefined,
        employment_stability_years:  formData.stabilityYears ? parseFloat(formData.stabilityYears) : undefined,
        co_applicant_income:         formData.coApplicantIncome ? parseFloat(formData.coApplicantIncome) : undefined,
        co_applicant_employment_type: formData.coApplicantEmploymentType || undefined,
        co_applicant_cibil_score:    formData.coApplicantCibilScore ? parseInt(formData.coApplicantCibilScore) : undefined,
      } as any)
      setResult(data)
      setLoanRisk({ formData, result: data })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong")
    } finally {
      setIsCalculating(false)
    }
  }

  const handleInputChange = (field: string, value: string) => {
    const updated = { ...formData, [field]: value }
    setFormData(updated)
    setLoanRisk({ formData: updated, result: null })
    setResult(null)
  }

  const getRiskIcon = (size = "h-5 w-5") => {
    if (!result) return null
    if (result.risk_category === "Low")    return <CheckCircle2  className={cn(size, "shrink-0 text-success")}     />
    if (result.risk_category === "Medium") return <AlertTriangle className={cn(size, "shrink-0 text-warning")}     />
    return                                        <XCircle       className={cn(size, "shrink-0 text-destructive")} />
  }

  const visibleFactors = result?.factor_breakdown
    ? showAllFactors
      ? result.factor_breakdown
      : [...result.factor_breakdown].sort((a, b) => b.score - a.score).slice(0, 3)
    : []

  // FOIR = (current EMI + new proposed EMI) / income
  const income = parseFloat(formData.income) || 0
  const currentEmi = parseFloat(formData.emi) || 0
  const foirPct = income > 0 && result
    ? ((result.emi_to_income_ratio) * 100).toFixed(1)
    : null

  // For next steps URL
  const bankRatesUrl = `/bank-rates?cibilScore=${formData.cibilScore}&loanAmount=${formData.loanAmount}&loanPurpose=${formData.loanPurpose}&tenure=${formData.tenure}`

  return (
    <div className="space-y-8">

      {/* Page header */}
      <div>
        <T as="p" className="mb-1 text-xs font-semibold uppercase tracking-widest text-muted-foreground">Step 1 of 4</T>
        <h1 className="text-balance text-3xl font-bold tracking-tight text-foreground">
          {t("loanRiskForm")}
        </h1>
        <T as="p" className="mt-2 text-muted-foreground">Fill in your financial details — we'll calculate your risk score exactly the way banks do.</T>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">

        {/* ── Input form ── */}
        <Card className="border-none shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calculator className="h-5 w-5 text-primary" />
              <T>Your Financial Details</T>
            </CardTitle>
            <CardDescription>
              <T>All figures should be your current monthly numbers — be accurate for a reliable result</T>
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-5">

              {/* ── Section 1: Income & Obligations ── */}
              <div>
                <T as="p" className="mb-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Income & Obligations</T>
                <div className="space-y-5">
                  <div className="space-y-2">
                    <Label htmlFor="income" className="text-base font-medium">{t("income")}</Label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">₹</span>
                      <Input id="income" type="number" placeholder="50000"
                        value={formData.income} onChange={e => handleInputChange("income", e.target.value)}
                        className="h-12 pl-8 text-lg" required />
                    </div>
                    <FieldHint><T>Net take-home salary or average monthly business income after tax</T></FieldHint>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="expenses" className="text-base font-medium">{t("expenses")}</Label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">₹</span>
                      <Input id="expenses" type="number" placeholder="25000"
                        value={formData.expenses} onChange={e => handleInputChange("expenses", e.target.value)}
                        className="h-12 pl-8 text-lg" required />
                    </div>
                    <FieldHint><T>Monthly living costs (rent, food, utilities, transport) — excluding existing EMIs</T></FieldHint>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="emi" className="text-base font-medium">{t("emi")}</Label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">₹</span>
                      <Input id="emi" type="number" placeholder="10000"
                        value={formData.emi} onChange={e => handleInputChange("emi", e.target.value)}
                        className="h-12 pl-8 text-lg" />
                    </div>
                    <FieldHint><T>Total of all existing loan EMIs you are currently paying every month</T></FieldHint>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="existingLoans" className="text-base font-medium">{t("existingLoans")}</Label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">₹</span>
                      <Input id="existingLoans" type="number" placeholder="100000"
                        value={formData.existingLoans} onChange={e => handleInputChange("existingLoans", e.target.value)}
                        className="h-12 pl-8 text-lg" />
                    </div>
                    <FieldHint><T>Total outstanding principal remaining across all current loans (not monthly EMI)</T></FieldHint>
                  </div>
                </div>
              </div>

              {/* ── Section 2: Loan Details ── */}
              <SectionLabel>{t("sectionLoanDetails")}</SectionLabel>
              <div className="space-y-5">
                <div className="space-y-2">
                  <Label htmlFor="loanAmount" className="text-base font-medium">{t("loanAmount")}</Label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">₹</span>
                    <Input id="loanAmount" type="number" placeholder="500000"
                      value={formData.loanAmount} onChange={e => handleInputChange("loanAmount", e.target.value)}
                      className="h-12 pl-8 text-lg" required />
                  </div>
                  <FieldHint><T>Banks typically lend up to 60x your monthly income for personal loans</T></FieldHint>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="tenure" className="text-base font-medium">{t("tenure")}</Label>
                  <Input id="tenure" type="number" placeholder="60"
                    value={formData.tenure} onChange={e => handleInputChange("tenure", e.target.value)}
                    className="h-12 text-lg" required min={1} />
                  <FieldHint><T>Longer tenure = lower EMI but higher total interest paid overall</T></FieldHint>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="loanPurpose" className="text-base font-medium">{t("loanPurpose")}</Label>
                  <select id="loanPurpose" value={formData.loanPurpose}
                    onChange={e => handleInputChange("loanPurpose", e.target.value)}
                    className="h-12 w-full rounded-md border border-input bg-background px-3 text-base focus:outline-none focus:ring-2 focus:ring-ring">
                    <TranslatedOption value="">Select purpose…</TranslatedOption>
                    <TranslatedOption value="home">Home Loan</TranslatedOption>
                    <TranslatedOption value="personal">Personal Loan</TranslatedOption>
                    <TranslatedOption value="business">Business Loan</TranslatedOption>
                    <TranslatedOption value="education">Education Loan</TranslatedOption>
                    <TranslatedOption value="vehicle">Vehicle / Car Loan</TranslatedOption>
                    <TranslatedOption value="agriculture">Agriculture / Kisan Loan</TranslatedOption>
                    <TranslatedOption value="other">Other</TranslatedOption>
                  </select>
                  <FieldHint><T>Determines which banks are eligible and minimum CIBIL cutoffs</T></FieldHint>
                </div>
              </div>

              {/* ── Section 3: Credit Profile ── */}
              <SectionLabel>{t("sectionCreditProfile")}</SectionLabel>
              <div className="space-y-5">
                <div className="space-y-2">
                  <Label htmlFor="cibilScore" className="text-base font-medium">
                    {t("cibilScore")} <span className="text-sm font-normal text-muted-foreground">(300–900)</span>
                  </Label>
                  <Input id="cibilScore" type="number" placeholder="750"
                    value={formData.cibilScore} onChange={e => handleInputChange("cibilScore", e.target.value)}
                    className="h-12 text-lg" min={300} max={900} />
                  {formData.cibilScore && (
                    <p className={cn("text-xs font-semibold", parseInt(formData.cibilScore) >= 750 ? "text-success" : parseInt(formData.cibilScore) >= 700 ? "text-blue-500" : parseInt(formData.cibilScore) >= 650 ? "text-warning" : "text-destructive")}>
                      {parseInt(formData.cibilScore) >= 750 ? <T>Excellent — lowest rates available</T>
                        : parseInt(formData.cibilScore) >= 700 ? <T>Good — most banks will approve</T>
                        : parseInt(formData.cibilScore) >= 650 ? <T>Fair — higher rate likely</T>
                        : <T>Poor — approval very difficult</T>}
                    </p>
                  )}
                  <FieldHint><T>Check free at CIBIL.com, Paytm Money, or Google Pay. Single bureau check = 1 hard enquiry</T></FieldHint>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="age" className="text-base font-medium">
                    {t("age")} <span className="text-sm font-normal text-muted-foreground">(years)</span>
                  </Label>
                  <Input id="age" type="number" placeholder="30"
                    value={formData.age} onChange={e => handleInputChange("age", e.target.value)}
                    className="h-12 text-lg" min={18} max={75} />
                  <FieldHint><T>Most banks require the loan to close before age 60 (salaried) or 65 (self-employed)</T></FieldHint>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="employmentType" className="text-base font-medium">{t("employmentType")}</Label>
                  <select id="employmentType" value={formData.employmentType}
                    onChange={e => handleInputChange("employmentType", e.target.value)}
                    className="h-12 w-full rounded-md border border-input bg-background px-3 text-base focus:outline-none focus:ring-2 focus:ring-ring">
                    <TranslatedOption value="salaried">Salaried (private/govt)</TranslatedOption>
                    <TranslatedOption value="self_employed">Self-Employed / Business Owner</TranslatedOption>
                    <TranslatedOption value="farmer">Farmer / Agriculture</TranslatedOption>
                    <TranslatedOption value="student">Student</TranslatedOption>
                    <TranslatedOption value="other">Other</TranslatedOption>
                  </select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="stabilityYears" className="text-base font-medium">
                    {t("stabilityYears")}
                  </Label>
                  <Input id="stabilityYears" type="number" placeholder="3"
                    value={formData.stabilityYears} onChange={e => handleInputChange("stabilityYears", e.target.value)}
                    className="h-12 text-lg" min={0} step={0.5} />
                  <FieldHint><T>Banks prefer 2+ years — shorter tenures may require a guarantor or collateral</T></FieldHint>
                </div>
              </div>

              {/* ── Section 4: Co-Applicant (optional) ── */}
              <SectionLabel><T>Co-Applicant (Optional)</T></SectionLabel>
              <div className="rounded-xl border border-dashed border-primary/40 bg-primary/5 p-4 space-y-4">
                <T as="p" className="text-xs text-muted-foreground">Adding a co-applicant (spouse, parent, or sibling) combines incomes and CIBILs — reducing your risk score and improving eligibility.</T>
                <div className="space-y-2">
                  <Label htmlFor="coApplicantIncome" className="text-base font-medium"><T>Co-Applicant Monthly Income</T></Label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">₹</span>
                    <Input id="coApplicantIncome" type="number" placeholder="30000"
                      value={formData.coApplicantIncome} onChange={e => handleInputChange("coApplicantIncome", e.target.value)}
                      className="h-12 pl-8 text-lg" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="coApplicantEmploymentType" className="text-base font-medium"><T>Co-Applicant Employment</T></Label>
                  <select id="coApplicantEmploymentType" value={formData.coApplicantEmploymentType}
                    onChange={e => handleInputChange("coApplicantEmploymentType", e.target.value)}
                    className="h-12 w-full rounded-md border border-input bg-background px-3 text-base focus:outline-none focus:ring-2 focus:ring-ring">
                    <TranslatedOption value="">Select…</TranslatedOption>
                    <TranslatedOption value="salaried">Salaried</TranslatedOption>
                    <TranslatedOption value="self_employed">Self-Employed</TranslatedOption>
                    <TranslatedOption value="farmer">Farmer</TranslatedOption>
                    <TranslatedOption value="other">Other</TranslatedOption>
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="coApplicantCibilScore" className="text-base font-medium">
                    <T>Co-Applicant CIBIL Score</T> <span className="text-sm font-normal text-muted-foreground">(300–900)</span>
                  </Label>
                  <Input id="coApplicantCibilScore" type="number" placeholder="720"
                    value={formData.coApplicantCibilScore} onChange={e => handleInputChange("coApplicantCibilScore", e.target.value)}
                    className="h-12 text-lg" min={300} max={900} />
                </div>
              </div>

              {error && (
                <div className="rounded-lg bg-destructive/10 px-4 py-3 text-sm text-destructive">{error}</div>
              )}

              <Button ref={submitRef} type="submit" className="h-14 w-full text-lg font-semibold" disabled={isCalculating}>
                {isCalculating ? <><Spinner className="mr-2 h-5 w-5" />{t("analyzing")}</> : t("calculateRisk")}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* ── Results panel ── */}
        <div className="space-y-6">
          <Card className="border-none shadow-lg">
            <CardHeader>
              <CardTitle>{t("riskScore")}</CardTitle>
              <CardDescription><T>Your loan risk assessment — as seen by lenders</T></CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col items-center">
              {result ? (
                <div className="w-full space-y-6">
                  <RiskGauge score={result.risk_score} />

                  {/* Risk explanation */}
                  <div className="flex items-start gap-3 rounded-xl bg-muted p-4">
                    {getRiskIcon()}
                    <p className="text-sm text-foreground"><DynamicText text={result.explanation} /></p>
                  </div>

                  {/* Key ratios */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl bg-muted p-3 text-center">
                      <T as="p" className="text-xs text-muted-foreground">FOIR (EMI to Income)</T>
                      <p className={cn("text-xl font-bold",
                        result.emi_to_income_ratio <= 0.4 ? "text-success" :
                        result.emi_to_income_ratio <= 0.5 ? "text-warning" : "text-destructive"
                      )}>
                        {(result.emi_to_income_ratio * 100).toFixed(1)}%
                      </p>
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {result.emi_to_income_ratio <= 0.4 ? <T>Banks prefer under 40%</T> :
                         result.emi_to_income_ratio <= 0.5 ? <T>Borderline — banks cap at 50%</T> : <T>Exceeds bank FOIR limit</T>}
                      </p>
                    </div>
                    <div className="rounded-xl bg-muted p-3 text-center">
                      <T as="p" className="text-xs text-muted-foreground">Debt-to-Income</T>
                      <p className="text-xl font-bold text-foreground">
                        {(result.debt_to_income_ratio * 100).toFixed(1)}%
                      </p>
                      <T as="p" className="mt-0.5 text-xs text-muted-foreground">Outstanding vs annual income</T>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex h-64 flex-col items-center justify-center text-center">
                  <div className="mb-4 rounded-full bg-muted p-6">
                    <Calculator className="h-12 w-12 text-muted-foreground" />
                  </div>
                  <T as="p" className="text-lg font-medium text-muted-foreground">Fill in your details to see your risk assessment</T>
                  <T as="p" className="mt-1 text-sm text-muted-foreground">Takes about 30 seconds</T>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Factor breakdown */}
          {result && result.factor_breakdown.length > 0 && (
            <Card className="border-none shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-primary" />
                  <T>Score Breakdown</T>
                </CardTitle>
                <CardDescription><T>CIBIL-style factor analysis — what is driving your risk</T></CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                {visibleFactors.map(f => <FactorBar key={f.name} factor={f} />)}
                {result.factor_breakdown.length > 3 && (
                  <button type="button" onClick={() => setShowAllFactors(v => !v)}
                    className="flex w-full items-center justify-center gap-1 text-sm text-muted-foreground hover:text-foreground">
                    {showAllFactors
                      ? <><ChevronUp className="h-4 w-4" /><T>Show less</T></>
                      : <><ChevronDown className="h-4 w-4" /><T>Show all factors</T></>}
                  </button>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* ── Key factors & recommendation ── */}
      {result && (
        <div className="grid gap-6 md:grid-cols-2">
          <Card className="border-none shadow-lg">
            <CardHeader>
              <CardTitle className="text-base"><T>Key Factors Affecting Approval</T></CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {result.key_factors.map((factor, i) => (
                <div key={i} className="flex items-start gap-3 rounded-lg bg-muted p-3">
                  {getRiskIcon("h-4 w-4 mt-0.5")}
                  <p className="text-sm text-foreground"><DynamicText text={factor} /></p>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="border-none shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Lightbulb className="h-4 w-4 text-primary" />
                <T>Recommendation</T>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-lg bg-primary/10 p-4">
                <p className="text-sm text-foreground"><DynamicText text={result.recommendation} /></p>
              </div>
              {result.ai_advice && (
                <div className="rounded-xl border border-primary/20 bg-primary/5 p-4">
                  <T as="p" className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-primary">AI-Powered Advice</T>
                  <p className="text-sm leading-relaxed text-foreground"><DynamicText text={result.ai_advice} /></p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* ── Next Steps ── */}
      {result && (
        <Card className="border-none shadow-lg bg-gradient-to-r from-primary/5 to-transparent">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <ArrowRight className="h-4 w-4 text-primary" />
              <T>Your Next Steps</T>
            </CardTitle>
            <CardDescription><T>You have completed Step 1. Here is what to do next.</T></CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-3">
            <Link href={bankRatesUrl}>
              <div className="group cursor-pointer rounded-xl border border-primary/30 bg-primary/5 p-4 transition-all hover:bg-primary/10 hover:shadow-sm">
                <Building2 className="mb-2 h-5 w-5 text-primary" />
                <T as="p" className="font-semibold text-sm text-foreground">Step 2 — Compare Banks</T>
                <T as="p" className="mt-1 text-xs text-muted-foreground leading-relaxed">See which banks will approve you and what rate they offer based on your CIBIL score</T>
                <p className="mt-2 flex items-center gap-1 text-xs font-semibold text-primary">
                  <T>Go</T> <ArrowRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5" />
                </p>
              </div>
            </Link>
            <Link href="/documents">
              <div className="group cursor-pointer rounded-xl border border-border bg-card p-4 transition-all hover:border-primary/30 hover:bg-primary/5 hover:shadow-sm">
                <FileText className="mb-2 h-5 w-5 text-primary" />
                <T as="p" className="font-semibold text-sm text-foreground">Step 3 — Prepare Documents</T>
                <T as="p" className="mt-1 text-xs text-muted-foreground leading-relaxed">Upload Aadhaar, PAN, CIBIL report and salary slip — AI extracts data automatically</T>
                <p className="mt-2 flex items-center gap-1 text-xs font-semibold text-primary">
                  <T>Go</T> <ArrowRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5" />
                </p>
              </div>
            </Link>
            <Link href="/roadmap">
              <div className="group cursor-pointer rounded-xl border border-border bg-card p-4 transition-all hover:border-primary/30 hover:bg-primary/5 hover:shadow-sm">
                <Map className="mb-2 h-5 w-5 text-primary" />
                <T as="p" className="font-semibold text-sm text-foreground">Step 4 — Plan Repayment</T>
                <T as="p" className="mt-1 text-xs text-muted-foreground leading-relaxed">Generate a month-by-month EMI schedule and set WhatsApp payment reminders</T>
                <p className="mt-2 flex items-center gap-1 text-xs font-semibold text-primary">
                  <T>Go</T> <ArrowRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5" />
                </p>
              </div>
            </Link>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
