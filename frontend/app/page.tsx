"use client"

import Link from "next/link"
import { useLanguage, languageNames, type Language } from "@/lib/language-context"
import { AppWrapper } from "@/components/app-wrapper"
import { usePageStrings } from "@/lib/use-page-strings"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useAppStore } from "@/lib/app-store"
import { DynamicText } from "@/lib/auto-translate"
import {
  Calculator, MessageCircle, FileText, TrendingUp, ArrowRight,
  CheckCircle2, Building2, Coins, Map, Bell, ShieldCheck,
  ChevronRight, Star, IndianRupee,
} from "lucide-react"

export default function HomePage() {
  return <AppWrapper><Dashboard /></AppWrapper>
}

function loanPurposeEnglishLabel(purpose: string): string {
  const m: Record<string, string> = {
    home: "Home loan",
    personal: "Personal loan",
    business: "Business loan",
    education: "Education loan",
    vehicle: "Vehicle / car loan",
    agriculture: "Agriculture loan",
    other: "Other loan",
  }
  return m[purpose] ?? purpose
}

const LOAN_TYPE_TABLE_ROWS = [
  { type: "Home Loan", rate: "7.15 – 9.00%", cibil: "650", tenure: "30 yrs", time: "2–4 weeks" },
  { type: "Personal Loan", rate: "9.99 – 24.00%", cibil: "700", tenure: "7 yrs", time: "1–7 days" },
  { type: "Vehicle Loan", rate: "7.60 – 12.00%", cibil: "650", tenure: "7 yrs", time: "3–5 days" },
  { type: "Education Loan", rate: "4.00 – 15.00%", cibil: "600", tenure: "15 yrs", time: "1–2 weeks" },
  { type: "Business Loan", rate: "11.00 – 20.00%", cibil: "650", tenure: "5 yrs", time: "3–7 days" },
  { type: "Gold Loan", rate: "7.50 – 14.10%", cibil: "None", tenure: "3 yrs", time: "30 mins" },
] as const

function Dashboard() {
  const { t, language, setLanguage } = useLanguage()
  const { loanRisk, roadmap } = useAppStore()
  const ps = usePageStrings({
    heroTag: "FinAI · Smart Loan Planning",
    heroTitle: "Take a Loan the Right Way. Every Step, Guided.",
    heroSub: "80% of rejections happen because applicants skip eligibility checks. FinAI walks you through CIBIL check → bank comparison → documents → repayment — so you walk in prepared.",
    askAI: "Ask AI Assistant",
    lastAssessment: "Last Assessment",
    riskCategory: "Risk Category",
    cibilScore: "CIBIL Score",
    requestedLoan: "Requested Loan",
    loanType: "Loan Type",
    viewBankRates: "View Bank Rates",
    reassess: "Re-assess",
    stepsLabel: "4 steps to approval",
    startNow: "Start now",
    completed: "Completed",
    completeFirst: "Complete previous step first",
    cibilGuideTitle: "CIBIL Score — What Banks See First",
    checkFree: "Check your score free at CIBIL.com, Paytm, or Google Pay",
    bankChecksTitle: "What Banks Check Before Approving",
    appSwitchInstantly: "App will switch language instantly",
    loanTypesTitle: "Loan Types at a Glance (April 2026 Rates)",
    loanTypeCol: "Loan Type", rateCol: "Rate Range", cibilCol: "Min CIBIL", maxCol: "Max Amount",
    tenureCol: "Max Tenure", processingCol: "Processing Time",
    loanTableNote: "Rates vary by bank, loan amount, and your credit profile. Use the Bank Rate Comparison tool for personalised quotes.",
    step1title: "Check Eligibility", step1desc: "CIBIL score, income & maximum loan amount",
    step2title: "Compare Bank Rates", step2desc: "Personalised rates from 7+ banks for your profile",
    step3title: "Prepare Documents", step3desc: "KYC & income proof — scan with AI in seconds",
    step4title: "Plan Repayment", step4desc: "Month-by-month EMI schedule & payment reminders",
    loanRiskCheck: "Loan Risk Check", loanRiskDesc: "Eligibility & risk score",
    bankRateCompare: "Bank Rate Compare", bankRateDesc: "7+ banks, personalised to CIBIL",
    docScanner: "Document Scanner", docScanDesc: "AI OCR — Aadhaar, PAN, CIBIL",
    repayRoadmap: "Repayment Roadmap", repayRoadmapDesc: "Month-by-month EMI schedule",
    goldLoanTitle: "Gold Loan", goldLoanDesc: "No CIBIL — pledge jewellery",
    creditSim: "Credit Simulator", creditSimDesc: "What-if scenario testing",
    aiAssistant: "AI Assistant", aiAssistantDesc: "Ask anything in your language",
    emiReminders: "EMI Reminders", emiRemindersDesc: "WhatsApp alerts — never miss EMI",
    cibilExcellent: "Excellent", cibilExcellentNote: "Lowest rates, fastest approval",
    cibilGood: "Good", cibilGoodNote: "Most loans approved",
    cibilFair: "Fair", cibilFairNote: "Higher rate, stricter scrutiny",
    cibilPoor: "Poor", cibilPoorNote: "Likely rejection — improve first",
    rule1: "FOIR under 50% of gross income", rule1note: "Fixed Obligation to Income Ratio — all EMIs including the new one",
    rule2: "CIBIL 650 min · 750+ for best rate", rule2note: "Checked instantly; no negotiation",
    rule3: "Age 21–58 at loan close", rule3note: "Home loans often allow up to age 70",
    rule4: "2+ years current employment", rule4note: "Or 2 years ITR for self-employed",
    rule5: "No defaults in last 12 months", rule5note: "Even one late payment shows in CIBIL",
  }, "home")

  // ── Derived state ────────────────────────────────────────────────────────
  const hasRiskResult = !!(loanRisk.result)
  const cibil = loanRisk.formData.cibilScore ? parseInt(loanRisk.formData.cibilScore) : null

  const getCibilColor = (score: number) => {
    if (score >= 750) return "text-success"
    if (score >= 700) return "text-blue-500"
    if (score >= 650) return "text-warning"
    return "text-destructive"
  }

  const getCibilLabel = (score: number) => {
    if (score >= 750) return ps.cibilExcellent
    if (score >= 700) return ps.cibilGood
    if (score >= 650) return ps.cibilFair
    return ps.cibilPoor
  }

  const hasRoadmap = !!(roadmap.result?.repayment_plan && roadmap.result.repayment_plan.length > 0)

  const journeySteps = [
    {
      step: 1,
      href: "/loan-risk",
      title: ps.step1title,
      desc: ps.step1desc,
      done: hasRiskResult,
      active: !hasRiskResult,
    },
    {
      step: 2,
      href: "/bank-rates",
      title: ps.step2title,
      desc: ps.step2desc,
      done: false,
      active: hasRiskResult,
    },
    {
      step: 3,
      href: "/documents",
      title: ps.step3title,
      desc: ps.step3desc,
      done: false,
      active: false,
    },
    {
      step: 4,
      href: "/roadmap",
      title: ps.step4title,
      desc: ps.step4desc,
      done: hasRoadmap,
      active: false,
    },
  ]

  return (
    <div className="space-y-8">

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-primary via-primary/85 to-primary/60 p-6 text-primary-foreground md:p-8">
        <div className="relative z-10 max-w-xl">
          <p className="mb-1 text-xs font-semibold uppercase tracking-widest opacity-70">{ps.heroTag}</p>
          <h1 className="mb-3 text-2xl font-bold leading-tight md:text-3xl">{ps.heroTitle}</h1>
          <p className="mb-5 text-sm leading-relaxed opacity-85">{ps.heroSub}</p>
          <div className="flex flex-wrap gap-3">
            <Link href={hasRiskResult ? "/bank-rates" : "/loan-risk"}>
              <Button variant="secondary" className="h-10 gap-2 font-semibold">
                <Calculator className="h-4 w-4" />
                {hasRiskResult ? t("compareBanks") : t("checkMyEligibility")}
              </Button>
            </Link>
            <Link href="/chat">
              <Button variant="outline" className="h-10 gap-2 border-white/30 bg-white/10 text-white hover:bg-white/20 font-medium">
                <MessageCircle className="h-4 w-4" />
                {ps.askAI}
              </Button>
            </Link>
          </div>
        </div>
        <div className="pointer-events-none absolute -right-10 -top-10 h-48 w-48 rounded-full bg-white/5" />
        <div className="pointer-events-none absolute -bottom-14 right-20 h-36 w-36 rounded-full bg-white/5" />
      </div>

      {/* ── Last assessment snapshot ─────────────────────────────────────── */}
      {hasRiskResult && loanRisk.result && (
        <Card className="border-none shadow-lg">
          <CardContent className="p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{ps.lastAssessment}</p>
                <div className="flex flex-wrap items-center gap-5">
                  <div>
                    <p className="text-xs text-muted-foreground">{ps.riskCategory}</p>
                    <p className={cn("text-2xl font-bold",
                      loanRisk.result.risk_category === "Low" ? "text-success" :
                      loanRisk.result.risk_category === "Medium" ? "text-warning" : "text-destructive"
                    )}>
                      {loanRisk.result.risk_category === "Low"
                        ? t("lowRisk")
                        : loanRisk.result.risk_category === "Medium"
                          ? t("mediumRisk")
                          : t("highRisk")}
                    </p>
                  </div>
                  {cibil && (
                    <div className="border-l border-border pl-5">
                      <p className="text-xs text-muted-foreground">{ps.cibilScore}</p>
                      <p className={cn("text-2xl font-bold", getCibilColor(cibil))}>
                        {cibil}
                        <span className={cn("ml-1.5 text-sm font-medium", getCibilColor(cibil))}>{getCibilLabel(cibil)}</span>
                      </p>
                    </div>
                  )}
                  {loanRisk.formData.loanAmount && (
                    <div className="border-l border-border pl-5">
                      <p className="text-xs text-muted-foreground">{ps.requestedLoan}</p>
                      <p className="text-2xl font-bold text-foreground">
                        ₹{parseInt(loanRisk.formData.loanAmount).toLocaleString("en-IN")}
                      </p>
                    </div>
                  )}
                  {loanRisk.formData.loanPurpose && (
                    <div className="border-l border-border pl-5">
                      <p className="text-xs text-muted-foreground">{ps.loanType}</p>
                      <p className="text-lg font-semibold text-foreground capitalize">
                        <DynamicText text={loanPurposeEnglishLabel(loanRisk.formData.loanPurpose)} />
                      </p>
                    </div>
                  )}
                </div>
              </div>
              <div className="flex gap-2 flex-wrap">
                <Link href={`/bank-rates?cibilScore=${loanRisk.formData.cibilScore}&loanAmount=${loanRisk.formData.loanAmount}&loanPurpose=${loanRisk.formData.loanPurpose}&tenure=${loanRisk.formData.tenure}`}>
                  <Button size="sm" className="gap-1.5">
                    {ps.viewBankRates} <ArrowRight className="h-3 w-3" />
                  </Button>
                </Link>
                <Link href="/loan-risk">
                  <Button size="sm" variant="outline">{ps.reassess}</Button>
                </Link>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Loan Journey ─────────────────────────────────────────────────── */}
      <div>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">{t("yourLoanJourney")}</h2>
          <span className="text-xs text-muted-foreground">{ps.stepsLabel}</span>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {journeySteps.map((step) => (
            <Link key={step.step} href={step.href}>
              <div className={cn(
                "group relative flex h-full flex-col rounded-xl border p-4 transition-all hover:shadow-md",
                step.done   ? "border-success/30 bg-success/5" :
                step.active ? "border-primary/40 bg-primary/5 shadow-sm" :
                              "border-border bg-card",
              )}>
                <div className="flex items-start gap-3">
                  <div className={cn(
                    "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold",
                    step.done   ? "bg-success text-success-foreground" :
                    step.active ? "bg-primary text-primary-foreground" :
                                  "bg-muted text-muted-foreground",
                  )}>
                    {step.done ? <CheckCircle2 className="h-4 w-4" /> : step.step}
                  </div>
                  <div>
                    <p className={cn("text-sm font-semibold",
                      step.active || step.done ? "text-foreground" : "text-muted-foreground"
                    )}>
                      {step.title}
                    </p>
                    <p className="mt-0.5 text-xs text-muted-foreground">{step.desc}</p>
                  </div>
                </div>
                <div className="mt-auto pt-3">
                  {step.active && (
                    <span className="flex items-center gap-1 text-xs font-semibold text-primary">
                      {ps.startNow} <ArrowRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5" />
                    </span>
                  )}
                  {step.done && <span className="text-xs font-medium text-success">{ps.completed} ✓</span>}
                  {!step.active && !step.done && <span className="text-xs text-muted-foreground">{ps.completeFirst}</span>}
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* ── Education row ────────────────────────────────────────────────── */}
      <div className="grid gap-6 md:grid-cols-2">

        {/* CIBIL guide */}
        <Card className="border-none shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Star className="h-4 w-4 text-primary" />
              {ps.cibilGuideTitle}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {([
              { range: "750 – 900", label: ps.cibilExcellent, note: ps.cibilExcellentNote, color: "text-success",     bg: "bg-success/10" },
              { range: "700 – 749", label: ps.cibilGood,      note: ps.cibilGoodNote,      color: "text-blue-500",    bg: "bg-blue-500/10" },
              { range: "650 – 699", label: ps.cibilFair,      note: ps.cibilFairNote,      color: "text-warning",     bg: "bg-warning/10" },
              { range: "300 – 649", label: ps.cibilPoor,      note: ps.cibilPoorNote,      color: "text-destructive", bg: "bg-destructive/10" },
            ] as {range:string,label:string,note:string,color:string,bg:string}[]).map(({ range, label, note, color, bg }) => (
              <div key={range} className={cn("flex items-center justify-between rounded-lg px-3 py-2 text-sm", bg)}>
                <div className="flex items-center gap-2 min-w-0">
                  <span className={cn("shrink-0 font-bold tabular-nums", color)}>{range}</span>
                  <span className="truncate text-xs text-muted-foreground">— {note}</span>
                </div>
                <span className={cn("ml-3 shrink-0 text-xs font-semibold", color)}>{label}</span>
              </div>
            ))}
            <p className="pt-1 text-xs text-muted-foreground">{ps.checkFree}</p>
          </CardContent>
        </Card>

        {/* Bank approval rules */}
        <Card className="border-none shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <ShieldCheck className="h-4 w-4 text-primary" />
              {ps.bankChecksTitle}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {([
              { rule: ps.rule1, note: ps.rule1note },
              { rule: ps.rule2, note: ps.rule2note },
              { rule: ps.rule3, note: ps.rule3note },
              { rule: ps.rule4, note: ps.rule4note },
              { rule: ps.rule5, note: ps.rule5note },
            ] as {rule:string,note:string}[]).map(({ rule, note }) => (
              <div key={rule} className="flex items-start gap-2.5">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />
                <div>
                  <p className="text-sm font-medium leading-tight text-foreground">{rule}</p>
                  <p className="text-xs text-muted-foreground">{note}</p>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* ── All features grid ────────────────────────────────────────────── */}
      <div>
        <h2 className="mb-4 text-lg font-semibold text-foreground">{t("allFeatures")}</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { href: "/loan-risk",  icon: Calculator,    title: ps.loanRiskCheck,   desc: ps.loanRiskDesc,       color: "bg-primary/10 text-primary" },
            { href: "/bank-rates", icon: Building2,     title: ps.bankRateCompare, desc: ps.bankRateDesc,       color: "bg-blue-500/10 text-blue-500" },
            { href: "/documents",  icon: FileText,      title: ps.docScanner,      desc: ps.docScanDesc,        color: "bg-violet-500/10 text-violet-500" },
            { href: "/roadmap",    icon: Map,           title: ps.repayRoadmap,    desc: ps.repayRoadmapDesc,   color: "bg-orange-500/10 text-orange-500" },
            { href: "/gold-loan",  icon: Coins,         title: ps.goldLoanTitle,   desc: ps.goldLoanDesc,       color: "bg-yellow-500/10 text-yellow-600" },
            { href: "/simulator",  icon: TrendingUp,    title: ps.creditSim,       desc: ps.creditSimDesc,      color: "bg-emerald-500/10 text-emerald-600" },
            { href: "/chat",       icon: MessageCircle, title: ps.aiAssistant,     desc: ps.aiAssistantDesc,    color: "bg-pink-500/10 text-pink-500" },
            { href: "/reminders",  icon: Bell,          title: ps.emiReminders,    desc: ps.emiRemindersDesc,   color: "bg-red-500/10 text-red-500" },
          ].map(({ href, icon: Icon, title, desc, color }) => (
            <Link key={href} href={href}>
              <div className="group flex cursor-pointer items-center gap-3 rounded-xl border border-border bg-card p-3.5 transition-all hover:border-primary/30 hover:shadow-md">
                <div className={cn("flex h-9 w-9 shrink-0 items-center justify-center rounded-lg", color)}>
                  <Icon className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold text-foreground">{title}</p>
                  <p className="truncate text-xs text-muted-foreground">{desc}</p>
                </div>
                <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* ── Language selector ────────────────────────────────────────────── */}
      <Card className="border-none shadow-sm">
        <CardContent className="p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-foreground">{t("selectLanguage")}</p>
              <p className="text-xs text-muted-foreground">{ps.appSwitchInstantly}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {(Object.keys(languageNames) as Language[]).map((lang) => (
                <button
                  key={lang}
                  onClick={() => setLanguage(lang)}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                    language === lang
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border bg-card text-muted-foreground hover:border-primary/40 hover:text-foreground"
                  )}
                >
                  {languageNames[lang]}
                </button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── Loan types quick reference ───────────────────────────────────── */}
      <Card className="border-none shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <IndianRupee className="h-4 w-4 text-primary" />
            {ps.loanTypesTitle}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-xs text-muted-foreground">
                  <th className="pb-2 text-left font-medium">{ps.loanTypeCol}</th>
                  <th className="pb-2 text-left font-medium">{ps.rateCol}</th>
                  <th className="pb-2 text-left font-medium">{ps.cibilCol}</th>
                  <th className="pb-2 text-left font-medium">{ps.tenureCol}</th>
                  <th className="pb-2 text-left font-medium">{ps.processingCol}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {LOAN_TYPE_TABLE_ROWS.map(({ type, rate, cibil, tenure, time }) => (
                  <tr key={type} className="text-sm">
                    <td className="py-2 font-medium text-foreground"><DynamicText text={type} /></td>
                    <td className="py-2 font-semibold text-primary"><DynamicText text={rate} /></td>
                    <td className="py-2 text-muted-foreground"><DynamicText text={cibil} /></td>
                    <td className="py-2 text-muted-foreground"><DynamicText text={tenure} /></td>
                    <td className="py-2 text-muted-foreground"><DynamicText text={time} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            {ps.loanTableNote}
          </p>
        </CardContent>
      </Card>

    </div>
  )
}
