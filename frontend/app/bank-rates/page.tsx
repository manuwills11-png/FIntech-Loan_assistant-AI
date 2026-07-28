"use client"

import { useState, useEffect } from "react"
import { useSearchParams } from "next/navigation"
import { AppWrapper } from "@/components/app-wrapper"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Spinner } from "@/components/ui/spinner"
import { cn } from "@/lib/utils"
import {
  Star, CheckCircle2, XCircle, ChevronDown, ChevronUp,
  Send, FileText, Building2, BadgeIndianRupee, X, Info,
} from "lucide-react"
import {
  getBankRates, contactBank,
  type BankRate, type BankRatesResponse, type ContactBankResponse,
} from "@/lib/api"
import { useAppStore } from "@/lib/app-store"
import { useLanguage } from "@/lib/language-context"
import { T } from "@/lib/auto-translate"

export default function BankRatesPage() {
  return <AppWrapper><BankRatesView /></AppWrapper>
}

// ── Loan cost summary ─────────────────────────────────────────────────────────

function LoanCostSummary({ loanAmount, emi, tenure, purpose }: {
  loanAmount: number; emi: number; tenure: number; purpose: string
}) {
  const totalPayable  = Math.round(emi * tenure)
  const totalInterest = Math.round(totalPayable - loanAmount)

  const timelines: Record<string, string> = {
    personal: "1–7 business days",
    home:     "2–4 weeks (includes legal/technical verification)",
    vehicle:  "3–5 business days",
    education:"1–2 weeks",
    business: "3–7 business days",
    gold:     "Same day (30–60 minutes)",
    agriculture: "1–2 weeks",
  }
  const timeline = timelines[purpose] || "Varies by bank"

  return (
    <Card className="border-none shadow-lg">
      <CardContent className="p-5">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm font-semibold text-foreground">Total Loan Cost Breakdown</p>
          <span className="rounded-full bg-muted px-3 py-0.5 text-xs text-muted-foreground capitalize">
            {purpose || "personal"} loan
          </span>
        </div>
        <div className="flex flex-wrap items-start gap-6">
          <div>
            <T as="p" className="text-xs text-muted-foreground">Loan Amount (Principal)</T>
            <p className="text-2xl font-bold text-foreground">{fmtINR(loanAmount)}</p>
          </div>
          <div className="border-l border-border pl-6">
            <T as="p" className="text-xs text-muted-foreground">Best Monthly EMI</T>
            <p className="text-2xl font-bold text-primary">{fmtINR(emi)}</p>
            <p className="text-xs text-muted-foreground">for {tenure} months</p>
          </div>
          <div className="border-l border-border pl-6">
            <T as="p" className="text-xs text-muted-foreground">Total Amount Payable</T>
            <p className="text-2xl font-bold text-foreground">{fmtINR(totalPayable)}</p>
          </div>
          <div className="border-l border-border pl-6">
            <T as="p" className="text-xs text-muted-foreground">Total Interest Cost</T>
            <p className="text-2xl font-bold text-warning">{fmtINR(totalInterest)}</p>
            <p className="text-xs text-muted-foreground">
              {totalInterest > 0 ? `${((totalInterest / loanAmount) * 100).toFixed(1)}% of principal` : ""}
            </p>
          </div>
          <div className="border-l border-border pl-6">
            <T as="p" className="text-xs text-muted-foreground">Typical Approval Time</T>
            <p className="text-sm font-semibold text-foreground">{timeline}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ── Application process steps ─────────────────────────────────────────────────

function ApprovalProcess({ purpose }: { purpose: string }) {
  const steps: Record<string, { step: string; desc: string }[]> = {
    home: [
      { step: "Apply Online / Branch",   desc: "Fill application, submit documents" },
      { step: "Document Verification",   desc: "Bank verifies income, CIBIL (1–3 days)" },
      { step: "Technical Verification",  desc: "Bank-appointed lawyer checks property title (3–7 days)" },
      { step: "Legal / Valuation",       desc: "Property valuation by approved valuer (2–5 days)" },
      { step: "Sanction Letter",         desc: "Bank issues conditional approval with rate & terms" },
      { step: "Disbursement",            desc: "Amount credited to builder / seller directly" },
    ],
    personal: [
      { step: "Apply Online / Branch",   desc: "Fill application, upload Aadhaar, PAN, salary slips" },
      { step: "CIBIL Check",             desc: "Instant credit score pull — takes seconds" },
      { step: "Document Verification",   desc: "Income proof & employment check (1–2 days)" },
      { step: "Sanction Letter",         desc: "Approval with final interest rate & terms" },
      { step: "Disbursement",            desc: "Amount credited to your bank account within 24 hrs" },
    ],
    vehicle: [
      { step: "Apply Online / Dealer",   desc: "Fill application at showroom or bank branch" },
      { step: "CIBIL & Income Check",    desc: "Bank verifies score and repayment capacity" },
      { step: "Document Verification",   desc: "Aadhaar, PAN, salary slips (2–3 days)" },
      { step: "Sanction Letter",         desc: "Approval with rate; bank pays dealer directly" },
      { step: "Vehicle Registration",    desc: "Hypothecation to bank noted on RC book" },
    ],
    education: [
      { step: "Apply at Bank Branch",    desc: "Admission letter + fee structure required" },
      { step: "Admission Verification",  desc: "Bank confirms institute is approved" },
      { step: "Co-applicant Check",      desc: "Parent/guardian income & CIBIL verified" },
      { step: "Sanction",                desc: "Loan sanctioned; disbursed semester-by-semester" },
      { step: "Moratorium Period",       desc: "EMI starts 6–12 months after course completion" },
    ],
  }

  const processSteps = steps[purpose] || steps.personal

  return (
    <Card className="border-none shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Building2 className="h-4 w-4 text-primary" />
          <T>How the Loan Approval Process Works</T>
          <span className="ml-1 rounded-full bg-muted px-2 py-0.5 text-xs font-normal capitalize text-muted-foreground">
            {purpose || "personal"} loan
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {processSteps.map((s, i) => (
            <div key={i} className="flex items-start gap-3">
              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/15 text-xs font-bold text-primary">
                {i + 1}
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">{s.step}</p>
                <p className="text-xs text-muted-foreground">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs text-muted-foreground">
          <T>Tip: Having all documents ready (Aadhaar, PAN, salary slips, bank statements) reduces processing time significantly.</T>
          {" "}<T>Use the</T> <a href="/documents" className="text-primary underline"><T>Document Scanner</T></a> <T>to extract data from your docs instantly.</T>
        </p>
      </CardContent>
    </Card>
  )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtINR(v: number) {
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(v)
}

function ScoreBadge({ score }: { score: number }) {
  const cls = score >= 750 ? "bg-success/20 text-success" :
              score >= 700 ? "bg-blue-500/20 text-blue-600" :
              score >= 650 ? "bg-warning/20 text-warning" : "bg-destructive/20 text-destructive"
  const label = score >= 750 ? "Excellent" : score >= 700 ? "Good" : score >= 650 ? "Fair" : "Poor"
  return <span className={cn("rounded-full px-2 py-0.5 text-xs font-semibold", cls)}>{label}</span>
}

function RatePill({ rate, recommended }: { rate: number; recommended: boolean }) {
  return (
    <div className={cn("rounded-xl px-3 py-2 text-center", recommended ? "bg-primary text-primary-foreground" : "bg-muted")}>
      <p className="text-xs opacity-70">Rate p.a.</p>
      <p className="text-xl font-bold">{rate.toFixed(2)}%</p>
    </div>
  )
}

// ── Contact Bank Modal ────────────────────────────────────────────────────────

function ContactModal({
  bank, params, onClose,
}: {
  bank: BankRate
  params: { loan_amount: number; loan_purpose: string; tenure_months: number; cibil_score: number }
  onClose: () => void
}) {
  const { loanRisk } = useAppStore()
  const { t } = useLanguage()
  const f = loanRisk.formData
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ContactBankResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showInquiry, setShowInquiry] = useState(false)

  const send = async () => {
    setLoading(true); setError(null)
    try {
      const res = await contactBank({
        bank: bank.bank_short,
        loan_purpose: params.loan_purpose,
        loan_amount: params.loan_amount,
        tenure_months: params.tenure_months,
        income: parseFloat(f.income) || 0,
        expenses: parseFloat(f.expenses) || 0,
        existing_loans: parseFloat(f.existingLoans) || 0,
        employment_type: f.employmentType || "salaried",
        cibil_score: params.cibil_score,
        age: f.age ? parseInt(f.age) : undefined,
      })
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { send() }, [])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-lg rounded-2xl bg-card shadow-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between border-b border-border p-4">
          <div className="flex items-center gap-3">
            <div className={cn("flex h-10 w-10 items-center justify-center rounded-full text-white text-sm font-bold", bank.logo_color)}>
              {bank.bank_short.slice(0, 2)}
            </div>
            <div>
              <p className="font-semibold text-foreground">{bank.bank}</p>
              <T as="p" className="text-xs text-muted-foreground">Loan Inquiry</T>
            </div>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="p-4 space-y-4">
          {loading && (
            <div className="flex flex-col items-center gap-3 py-8">
              <Spinner className="h-8 w-8 text-primary" />
              <p className="text-sm text-muted-foreground">Sending inquiry to {bank.bank}…</p>
            </div>
          )}

          {error && <p className="text-sm text-destructive">{error}</p>}

          {result && (
            <>
              {/* Rate + EMI */}
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-xl bg-primary/10 p-3 text-center">
                  <T as="p" className="text-xs text-muted-foreground">Your Interest Rate</T>
                  <p className="text-2xl font-bold text-primary">{result.estimated_rate.toFixed(2)}%</p>
                  <p className="text-xs text-muted-foreground">p.a.</p>
                </div>
                <div className="rounded-xl bg-muted p-3 text-center">
                  <p className="text-xs text-muted-foreground">{t("monthlyEmi")}</p>
                  <p className="text-2xl font-bold text-foreground">{fmtINR(result.estimated_emi)}</p>
                </div>
              </div>

              {/* Bank response */}
              <div className="rounded-xl bg-muted p-4">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {bank.bank} <T>Response</T>
                </p>
                <p className="text-sm text-foreground whitespace-pre-wrap leading-relaxed">{result.bank_response}</p>
              </div>

              {/* Next steps */}
              <div className="space-y-2">
                <T as="p" className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Next Steps</T>
                {result.next_steps.map((step, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/20 text-xs font-bold text-primary">{i + 1}</span>
                    <p className="text-sm text-foreground">{step}</p>
                  </div>
                ))}
              </div>

              {/* Inquiry sent (collapsible) */}
              <button
                onClick={() => setShowInquiry(v => !v)}
                className="flex w-full items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              >
                {showInquiry ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                {showInquiry ? <T>Hide</T> : <T>View</T>} <T>inquiry sent</T>
              </button>
              {showInquiry && (
                <div className="rounded-xl bg-muted p-3">
                  <p className="text-xs font-mono whitespace-pre-wrap text-muted-foreground">{result.inquiry_text}</p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Bank Card ─────────────────────────────────────────────────────────────────

function BankCard({
  bank, emi, loanAmount, onContact,
}: {
  bank: BankRate; emi: number; loanAmount: number; onContact: () => void
}) {
  const [expanded, setExpanded] = useState(false)
  const { t } = useLanguage()

  return (
    <div className={cn(
      "rounded-2xl border bg-card shadow-sm transition-all",
      bank.recommended ? "border-primary shadow-primary/20 shadow-md" : "border-border",
      !bank.eligible && "opacity-60"
    )}>
      {bank.recommended && (
        <div className="rounded-t-2xl bg-primary px-4 py-1.5 text-center text-xs font-semibold text-primary-foreground">
          ⭐ <T>Best Rate for Your CIBIL Score</T>
        </div>
      )}

      <div className="p-4">
        <div className="flex items-start gap-3">
          {/* Logo */}
          <div className={cn("flex h-12 w-12 shrink-0 items-center justify-center rounded-xl text-white text-sm font-bold", bank.logo_color)}>
            {bank.bank_short.slice(0, 3)}
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <p className="font-semibold text-foreground">{bank.bank}</p>
              <span className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">{bank.type}</span>
            </div>
            <p className="text-xs text-muted-foreground mt-0.5">
              Range: {bank.min_rate}%–{bank.max_rate}% p.a.
            </p>
          </div>

          {/* Rate pill */}
          <RatePill rate={bank.rate_for_score} recommended={bank.recommended} />
        </div>

        {/* Key stats */}
        <div className="mt-4 grid grid-cols-3 gap-2 text-center">
          <div className="rounded-lg bg-muted p-2">
            <p className="text-xs text-muted-foreground">{t("monthlyEmi")}</p>
            <p className="text-sm font-bold text-foreground">{bank.eligible ? fmtINR(emi) : "—"}</p>
          </div>
          <div className="rounded-lg bg-muted p-2">
            <T as="p" className="text-xs text-muted-foreground">Max Loan</T>
            <p className="text-sm font-bold text-foreground">{bank.max_loan}</p>
          </div>
          <div className="rounded-lg bg-muted p-2">
            <T as="p" className="text-xs text-muted-foreground">Max Tenure</T>
            <p className="text-sm font-bold text-foreground">{bank.max_tenure_years} yrs</p>
          </div>
        </div>

        {/* Eligibility */}
        <div className={cn("mt-3 flex items-start gap-2 rounded-lg p-2 text-xs",
          bank.eligible ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive"
        )}>
          {bank.eligible
            ? <CheckCircle2 className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            : <XCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />}
          <span>{bank.eligibility_note}</span>
        </div>

        {/* Expand/collapse */}
        <button
          onClick={() => setExpanded(v => !v)}
          className="mt-3 flex w-full items-center justify-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          {expanded ? <T>Less</T> : <T>Details & Documents</T>}
        </button>

        {expanded && (
          <div className="mt-3 space-y-3 border-t border-border pt-3">
            <div>
              <T as="p" className="mb-1.5 text-xs font-semibold text-muted-foreground uppercase tracking-wide">Features</T>
              <div className="flex flex-wrap gap-1.5">
                {bank.features.map(f => (
                  <span key={f} className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">{f}</span>
                ))}
              </div>
            </div>
            <div>
              <T as="p" className="mb-1.5 text-xs font-semibold text-muted-foreground uppercase tracking-wide">Documents Needed</T>
              <div className="space-y-1">
                {bank.documents.map(d => (
                  <div key={d} className="flex items-center gap-1.5 text-xs text-foreground">
                    <FileText className="h-3 w-3 text-muted-foreground" /> {d}
                  </div>
                ))}
              </div>
            </div>
            <p className="text-xs text-muted-foreground">{t("processingFee")}: {bank.processing_fee}</p>
          </div>
        )}

        {/* Contact button */}
        {bank.eligible && (
          <Button
            className="mt-3 h-10 w-full gap-2 text-sm"
            variant={bank.recommended ? "default" : "outline"}
            onClick={onContact}
          >
            <Send className="h-4 w-4" />
            {t("contactBankBtn")} – {bank.bank_short}
          </Button>
        )}
      </div>
    </div>
  )
}

// ── Main view ─────────────────────────────────────────────────────────────────

function BankRatesView() {
  const searchParams = useSearchParams()
  const { loanRisk } = useAppStore()
  const { t } = useLanguage()
  const f = loanRisk.formData

  const [cibil,      setCibil]      = useState(searchParams.get("cibilScore") || f.cibilScore || "700")
  const [loanAmount, setLoanAmount] = useState(searchParams.get("loanAmount") || f.loanAmount || "500000")
  const [purpose,    setPurpose]    = useState(searchParams.get("loanPurpose") || f.loanPurpose || "personal")
  const [tenure,     setTenure]     = useState(searchParams.get("tenure") || f.tenure || "60")
  const [data,       setData]       = useState<BankRatesResponse | null>(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState<string | null>(null)
  const [contactBank_, setContactBank] = useState<BankRate | null>(null)
  const [showIneligible, setShowIneligible] = useState(false)

  const fetch_ = async () => {
    if (!cibil || !loanAmount) return
    setLoading(true); setError(null)
    try {
      setData(await getBankRates(parseInt(cibil), parseFloat(loanAmount), purpose, parseInt(tenure)))
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetch_() }, [])

  const eligible   = data?.rates.filter(r => r.eligible)  ?? []
  const ineligible = data?.rates.filter(r => !r.eligible) ?? []

  return (
    <div className="space-y-8">
      <div>
        <p className="mb-1 text-xs font-semibold uppercase tracking-widest text-muted-foreground">Step 2 of 4</p>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">{t("bankRateComparison")}</h1>
        <T as="p" className="mt-2 text-muted-foreground">See personalised rates from 7+ banks based on your CIBIL score — find the best deal before you apply</T>
      </div>

      {/* Input form */}
      <Card className="border-none shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Star className="h-5 w-5 text-primary" />
            {t("yourLoanProfile")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="space-y-1.5">
              <Label>{t("cibilScore")}</Label>
              <Input type="number" value={cibil} onChange={e => setCibil(e.target.value)}
                placeholder="700" min={300} max={900} className="h-11" />
              {cibil && <ScoreBadge score={parseInt(cibil)} />}
            </div>
            <div className="space-y-1.5">
              <Label>{t("loanAmount")} (₹)</Label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">₹</span>
                <Input type="number" value={loanAmount} onChange={e => setLoanAmount(e.target.value)}
                  placeholder="500000" className="h-11 pl-7" />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label>{t("loanPurpose")}</Label>
              <select value={purpose} onChange={e => setPurpose(e.target.value)}
                className="h-11 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring">
                <option value="personal">Personal Loan</option>
                <option value="home">Home Loan</option>
                <option value="business">Business Loan</option>
                <option value="education">Education Loan</option>
                <option value="vehicle">Vehicle Loan</option>
                <option value="gold">Gold Loan</option>
                <option value="agriculture">Agriculture Loan</option>
              </select>
            </div>
            <div className="space-y-1.5">
              <Label>{t("tenure")}</Label>
              <Input type="number" value={tenure} onChange={e => setTenure(e.target.value)}
                placeholder="60" min={6} className="h-11" />
            </div>
          </div>
          <Button className="mt-4 h-11 w-full sm:w-auto px-8" onClick={fetch_} disabled={loading}>
            {loading ? <><Spinner className="mr-2 h-4 w-4" /> Comparing…</> : t("compareRates")}
          </Button>
          {error && <p className="mt-2 text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {data && (
        <>
          {/* Summary */}
          <Card className="border-none shadow-lg">
            <CardContent className="pt-6">
              <div className="flex flex-wrap items-start gap-4">
                <div className="flex-1">
                  <p className="text-sm font-medium text-foreground">{data.summary}</p>
                </div>
                <div className="flex gap-3 shrink-0">
                  <div className="rounded-xl bg-success/10 px-4 py-2 text-center">
                    <T as="p" className="text-xs text-muted-foreground">Eligible Banks</T>
                    <p className="text-2xl font-bold text-success">{eligible.length}</p>
                  </div>
                  <div className="rounded-xl bg-muted px-4 py-2 text-center">
                    <T as="p" className="text-xs text-muted-foreground">Best EMI</T>
                    <p className="text-2xl font-bold text-foreground">{fmtINR(data.monthly_emi)}</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Total loan cost */}
          {data.monthly_emi > 0 && (
            <LoanCostSummary
              loanAmount={parseFloat(loanAmount)}
              emi={data.monthly_emi}
              tenure={parseInt(tenure)}
              purpose={purpose}
            />
          )}

          {/* Eligible banks grid */}
          {eligible.length > 0 && (
            <div>
              <h2 className="mb-4 text-lg font-semibold text-foreground">
                {eligible.length} <T>Banks Available for You</T>
              </h2>
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {eligible.map(bank => {
                  const r = bank.rate_for_score / 100 / 12
                  const months = parseInt(tenure)
                  const emi = r > 0
                    ? Math.round(parseFloat(loanAmount) * r * (1 + r) ** months / ((1 + r) ** months - 1))
                    : Math.round(parseFloat(loanAmount) / months)
                  return (
                    <BankCard
                      key={bank.bank}
                      bank={bank}
                      emi={emi}
                      loanAmount={parseFloat(loanAmount)}
                      onContact={() => setContactBank(bank)}
                    />
                  )
                })}
              </div>
            </div>
          )}

          {/* Ineligible banks */}
          {ineligible.length > 0 && (
            <div>
              <button
                onClick={() => setShowIneligible(v => !v)}
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
              >
                {showIneligible ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                {showIneligible ? <T>Hide</T> : <T>Show</T>} {ineligible.length} <T>banks not available for your current CIBIL score</T>
              </button>
              {showIneligible && (
                <div className="mt-3 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {ineligible.map(bank => (
                    <BankCard key={bank.bank} bank={bank} emi={0} loanAmount={parseFloat(loanAmount)} onContact={() => {}} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Loan approval process */}
          <ApprovalProcess purpose={purpose} />

          {/* CIBIL score tip */}
          {parseInt(cibil) < 700 && (
            <div className="flex items-start gap-3 rounded-xl border border-warning/30 bg-warning/5 p-4">
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
              <div>
                <T as="p" className="text-sm font-semibold text-foreground">Improve your CIBIL score to unlock better rates</T>
                <T as="p" className="mt-0.5 text-xs text-muted-foreground">Pay all current EMIs on time · Reduce credit card utilisation below 30% · Avoid multiple loan applications within 6 months · A score increase from 680 → 750 can reduce your interest rate by 1–2%, saving thousands over the loan tenure.</T>
              </div>
            </div>
          )}
        </>
      )}

      {/* Contact modal */}
      {contactBank_ && data && (
        <ContactModal
          bank={contactBank_}
          params={{
            loan_amount: parseFloat(loanAmount),
            loan_purpose: purpose,
            tenure_months: parseInt(tenure),
            cibil_score: parseInt(cibil),
          }}
          onClose={() => setContactBank(null)}
        />
      )}
    </div>
  )
}
