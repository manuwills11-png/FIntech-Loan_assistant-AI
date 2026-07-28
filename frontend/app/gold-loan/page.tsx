"use client"

import { useState, useEffect } from "react"
import { AppWrapper } from "@/components/app-wrapper"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Spinner } from "@/components/ui/spinner"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { ChevronDown, ChevronUp, Shield, Zap, Scale } from "lucide-react"
import { getBankRates, type BankRate } from "@/lib/api"
import { useAppStore } from "@/lib/app-store"
import { useLanguage } from "@/lib/language-context"
import { T } from "@/lib/auto-translate"

// Approximate gold rates per gram (Apr 2026)
const GOLD_RATE: Record<string, number> = { "24": 7500, "22": 6875, "18": 5625 }

function fmtINR(v: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency", currency: "INR", maximumFractionDigits: 0,
  }).format(v)
}

export default function GoldLoanPage() {
  return <AppWrapper><GoldLoanView /></AppWrapper>
}

function GoldLoanView() {
  const { goldLoan, setGoldLoan } = useAppStore()
  const { t } = useLanguage()

  const [weightGrams,  setWeightGrams]  = useState(goldLoan.goldWeightGrams  || "")
  const [purity,       setPurity]       = useState(goldLoan.goldPurityKarats || "22")
  const [ratePerGram,  setRatePerGram]  = useState(GOLD_RATE[goldLoan.goldPurityKarats || "22"].toString())
  const [loanAmount,   setLoanAmount]   = useState(goldLoan.loanAmount || "")
  const [tenure,       setTenure]       = useState("12")
  const [banks,        setBanks]        = useState<BankRate[]>([])
  const [loading,      setLoading]      = useState(false)
  const [goldValue,    setGoldValue]    = useState(0)
  const [maxLoan,      setMaxLoan]      = useState(0)
  const [showInelig,   setShowInelig]   = useState(false)

  // Recalculate gold value whenever inputs change
  useEffect(() => {
    const w    = parseFloat(weightGrams) || 0
    const rate = parseFloat(ratePerGram) || GOLD_RATE[purity] || 6875
    const val  = w * rate
    setGoldValue(val)
    setMaxLoan(Math.floor(val * 0.75))
  }, [weightGrams, purity, ratePerGram])

  // Sync purity change → update rate
  const handlePurityChange = (p: string) => {
    setPurity(p)
    setRatePerGram(GOLD_RATE[p].toString())
  }

  const handleCompare = async () => {
    if (!loanAmount || !tenure) return
    setLoading(true)
    setBanks([])
    try {
      // Gold loans don't use CIBIL — pass 750 to get base rates, eligible cutoff is 0
      const res = await getBankRates(750, parseFloat(loanAmount), "gold", parseInt(tenure))
      setBanks(res.rates)
      // Persist the current gold details
      setGoldLoan({ goldWeightGrams: weightGrams, goldPurityKarats: purity, loanAmount })
    } finally {
      setLoading(false)
    }
  }

  const eligible   = banks.filter(b => b.eligible)
  const ineligible = banks.filter(b => !b.eligible)

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">{t("goldLoan")}</h1>
        <T as="p" className="mt-2 text-muted-foreground">Instant loan against gold jewellery — no income proof, no CIBIL score required</T>
      </div>

      {/* Key benefits */}
      <div className="grid gap-4 sm:grid-cols-3">
        {[
          { icon: Zap,    id: "instant",  title: <T>Instant Disbursal</T>,  desc: <T>Most lenders disburse within 30 minutes of gold valuation.</T> },
          { icon: Shield, id: "noincome", title: <T>No Income Proof</T>,     desc: <T>Your gold is the collateral. No salary slips or ITR needed.</T> },
          { icon: Scale,  id: "rbi",      title: <T>RBI Regulated</T>,       desc: <T>Max 75% LTV is mandated by RBI. Your gold is fully insured.</T> },
        ].map(({ icon: Icon, id, title, desc }) => (
          <Card key={id} className="border-none shadow-sm">
            <CardContent className="flex items-start gap-3 pt-5">
              <Icon className="h-5 w-5 shrink-0 text-primary mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">{title}</p>
                <p className="mt-0.5 text-sm text-muted-foreground">{desc}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* How Gold Loan Works */}
      <Card className="border-none shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base"><T>How a Gold Loan Works — Step by Step</T></CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {[
              { step: 1, title: <T>Visit Branch</T>,       desc: <T>Go to bank/NBFC with your gold ornaments, Aadhaar & PAN</T> },
              { step: 2, title: <T>Gold Valuation</T>,     desc: <T>Lender's assayer tests purity & weighs gold on-site (15 min)</T> },
              { step: 3, title: <T>Loan Offer</T>,         desc: <T>Bank offers up to 75% of gold value (RBI LTV cap)</T> },
              { step: 4, title: <T>Sign & Pledge</T>,      desc: <T>Sign loan agreement; gold is sealed, stored in bank vault</T> },
              { step: 5, title: <T>Disbursement</T>,       desc: <T>Cash/NEFT within 30–60 minutes of pledge</T> },
            ].map(({ step, title, desc }) => (
              <div key={step} className="relative flex flex-col items-center text-center">
                <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                  {step}
                </div>
                <p className="mt-2 text-sm font-semibold text-foreground">{title}</p>
                <p className="mt-1 text-xs text-muted-foreground">{desc}</p>
              </div>
            ))}
          </div>
          <div className="mt-4 rounded-xl bg-muted px-4 py-3">
            <p className="text-xs text-muted-foreground">
                  <strong className="text-foreground"><T>Repayment options:</T></strong> <T>Monthly EMI · Bullet repayment (principal + interest at end) · Interest-only monthly payments. Gold is returned in the same sealed condition once the loan is fully repaid.</T>
              <strong className="text-foreground"> <T>If you default</T></strong>, <T>the lender auctions the gold after a notice period — so never pledge more than you can repay.</T>
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Gold details form */}
      <Card className="border-none shadow-lg">
        <CardHeader>
          <CardTitle><T>Your Gold Details</T></CardTitle>
          <CardDescription><T>Enter your gold details to calculate the maximum loan you can get</T></CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="weight">{t("goldWeight")}</Label>
              <Input
                id="weight"
                type="number"
                placeholder="e.g. 50"
                value={weightGrams}
                onChange={e => setWeightGrams(e.target.value)}
                className="h-11"
                min={1}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="purity">{t("goldPurity")}</Label>
              <select
                id="purity"
                value={purity}
                onChange={e => handlePurityChange(e.target.value)}
                className="h-11 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              >
                <option value="24">24 Karat — 99.9% pure</option>
                <option value="22">22 Karat — 91.6% hallmark</option>
                <option value="18">18 Karat — 75% pure</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="rate"><T>Gold Rate (₹ / gram)</T></Label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">₹</span>
                <Input
                  id="rate"
                  type="number"
                  value={ratePerGram}
                  onChange={e => setRatePerGram(e.target.value)}
                  className="h-11 pl-7"
                  placeholder="6875"
                />
              </div>
              <T as="p" className="text-xs text-muted-foreground">Approx. Apr 2026 market rate</T>
            </div>
            <div className="space-y-2">
              <Label htmlFor="tenure">{t("tenure")}</Label>
              <Input
                id="tenure"
                type="number"
                placeholder="12"
                value={tenure}
                onChange={e => setTenure(e.target.value)}
                className="h-11"
                min={1}
                max={36}
              />
              <T as="p" className="text-xs text-muted-foreground">Gold loans: typically 3–36 months</T>
            </div>
          </div>

          {/* LTV summary */}
          {goldValue > 0 && (
            <div className="grid grid-cols-3 gap-4">
              <div className="rounded-xl bg-yellow-500/10 p-4 text-center">
                <T as="p" className="text-xs text-muted-foreground">Gold Market Value</T>
                <p className="text-xl font-bold text-yellow-600">{fmtINR(goldValue)}</p>
                <p className="text-xs text-muted-foreground mt-1">{weightGrams}g × ₹{ratePerGram}/g</p>
              </div>
              <div className="rounded-xl bg-primary/10 p-4 text-center">
                <p className="text-xs text-muted-foreground">{t("eligibleLoanAmt")}</p>
                <p className="text-xl font-bold text-primary">{fmtINR(maxLoan)}</p>
                <p className="text-xs text-muted-foreground mt-1">75% LTV (RBI max)</p>
              </div>
              <div className="rounded-xl bg-muted p-4 text-center">
                <T as="p" className="text-xs text-muted-foreground">LTV Ratio</T>
                <p className="text-xl font-bold text-foreground">75%</p>
                <T as="p" className="text-xs text-muted-foreground mt-1">Regulated max</T>
              </div>
            </div>
          )}

          {/* Loan amount input + compare button */}
          {goldValue > 0 && (
            <div className="flex items-end gap-4">
              <div className="flex-1 space-y-2">
                <Label htmlFor="loanAmt">{t("loanAmount")} (₹)</Label>
                <div className="relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">₹</span>
                  <Input
                    id="loanAmt"
                    type="number"
                    value={loanAmount}
                    onChange={e => setLoanAmount(e.target.value)}
                    placeholder={String(maxLoan)}
                    className="h-11 pl-7"
                    max={maxLoan}
                  />
                </div>
                {parseFloat(loanAmount) > maxLoan && maxLoan > 0 && (
                  <p className="text-xs text-destructive">
                    <T>Exceeds maximum eligible amount of</T> {fmtINR(maxLoan)} <T>(75% LTV)</T>
                  </p>
                )}
              </div>
              <Button
                className="h-11 px-8"
                onClick={handleCompare}
                disabled={loading || !loanAmount || parseFloat(loanAmount) <= 0}
              >
                {loading
                  ? <><Spinner className="mr-2 h-4 w-4" /> Comparing…</>
                  : t("compareBanks")
                }
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Bank comparison */}
      {banks.length > 0 && (
        <div className="space-y-6">
          <h2 className="text-xl font-semibold text-foreground">
            {eligible.length} <T>Banks Available for Gold Loans</T>
          </h2>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {eligible.map(bank => (
              <GoldBankCard
                key={bank.bank}
                bank={bank}
                loanAmount={parseFloat(loanAmount)}
                tenure={parseInt(tenure)}
              />
            ))}
          </div>

          {ineligible.length > 0 && (
            <div>
              <button
                onClick={() => setShowInelig(v => !v)}
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
              >
                {showInelig ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                {showInelig ? <T>Hide</T> : <T>Show</T>} {ineligible.length} <T>other lenders</T>
              </button>
              {showInelig && (
                <div className="mt-3 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {ineligible.map(bank => (
                    <GoldBankCard key={bank.bank} bank={bank} loanAmount={parseFloat(loanAmount)} tenure={parseInt(tenure)} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Required documents */}
          <Card className="border-none shadow-sm">
            <CardHeader>
              <CardTitle className="text-base"><T>Documents Required for All Gold Loans</T></CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-2 sm:grid-cols-2">
                {[
                  "Aadhaar Card (identity + address proof)",
                  "PAN Card",
                  "Gold ornaments / coins / bars to be pledged",
                  "Passport-size photographs (2)",
                  "Filled loan application form",
                  "Gold purity certificate (for coins/bars)",
                ].map(doc => (
                  <div key={doc} className="flex items-center gap-2 text-sm text-foreground">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
                    {doc}
                  </div>
                ))}
              </div>
              <T as="p" className="mt-4 text-xs text-muted-foreground">Note: No income proof, salary slips, ITR, or CIBIL score is required for gold loans. The gold is pledged with the lender and returned after full repayment.</T>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

function GoldBankCard({
  bank, loanAmount, tenure,
}: {
  bank: BankRate; loanAmount: number; tenure: number
}) {
  const [expanded, setExpanded] = useState(false)
  const r   = bank.rate_for_score / 100 / 12
  const emi = r > 0
    ? Math.round(loanAmount * r * (1 + r) ** tenure / ((1 + r) ** tenure - 1))
    : Math.round(loanAmount / tenure)

  return (
    <div className={cn(
      "rounded-2xl border bg-card shadow-sm transition-all",
      bank.recommended ? "border-primary shadow-primary/20 shadow-md" : "border-border",
      !bank.eligible && "opacity-60",
    )}>
      {bank.recommended && (
        <div className="rounded-t-2xl bg-primary px-4 py-1.5 text-center text-xs font-semibold text-primary-foreground">
          ⭐ <T>Best Rate</T>
        </div>
      )}

      <div className="p-4">
        <div className="flex items-start gap-3">
          <div className={cn(
            "flex h-12 w-12 shrink-0 items-center justify-center rounded-xl text-white text-xs font-bold",
            bank.logo_color,
          )}>
            {bank.bank_short.slice(0, 4)}
          </div>
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-foreground">{bank.bank}</p>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">{bank.type}</span>
              <Badge variant="outline" className="text-xs px-1.5">No CIBIL needed</Badge>
            </div>
            <p className="text-xs text-muted-foreground mt-0.5">{bank.min_rate}%–{bank.max_rate}% p.a.</p>
          </div>
          <div className={cn(
            "rounded-xl px-3 py-2 text-center shrink-0",
            bank.recommended ? "bg-primary text-primary-foreground" : "bg-muted",
          )}>
            <p className="text-xs opacity-70">Rate p.a.</p>
            <p className="text-xl font-bold">{bank.rate_for_score.toFixed(2)}%</p>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-2 text-center">
          <div className="rounded-lg bg-muted p-2">
            <T as="p" className="text-xs text-muted-foreground">Monthly EMI</T>
            <p className="text-sm font-bold text-foreground">{bank.eligible ? fmtINR(emi) : "—"}</p>
          </div>
          <div className="rounded-lg bg-muted p-2">
            <p className="text-xs text-muted-foreground">Max Loan</p>
            <p className="text-sm font-bold text-foreground">{bank.max_loan}</p>
          </div>
          <div className="rounded-lg bg-muted p-2">
            <p className="text-xs text-muted-foreground">Max Tenure</p>
            <p className="text-sm font-bold text-foreground">{bank.max_tenure_years} yrs</p>
          </div>
        </div>

        <button
          onClick={() => setExpanded(v => !v)}
          className="mt-3 flex w-full items-center justify-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          {expanded ? <T>Less</T> : <T>Features & Documents</T>}
        </button>

        {expanded && (
          <div className="mt-3 space-y-3 border-t border-border pt-3">
            <div>
              <T as="p" className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Features</T>
              <div className="flex flex-wrap gap-1.5">
                {bank.features.map(f => (
                  <span key={f} className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">{f}</span>
                ))}
              </div>
            </div>
            <div>
              <T as="p" className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Documents</T>
              {bank.documents.map(d => (
                <div key={d} className="flex items-center gap-1.5 text-xs text-foreground">
                  <span className="h-1 w-1 rounded-full bg-muted-foreground" />{d}
                </div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground"><T>Processing fee:</T> {bank.processing_fee}</p>
          </div>
        )}
      </div>
    </div>
  )
}
