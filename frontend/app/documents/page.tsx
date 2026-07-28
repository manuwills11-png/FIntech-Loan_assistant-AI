"use client"

import { useState, useRef } from "react"
import { useLanguage } from "@/lib/language-context"
import { AppWrapper } from "@/components/app-wrapper"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { cn } from "@/lib/utils"
import {
  Upload, FileText, CheckCircle2, X, ArrowRight, AlertCircle,
  Wallet, TrendingUp, TrendingDown, Receipt, CreditCard, User,
  Calendar, Users, Building2, BadgeIndianRupee, Star, Plus,
} from "lucide-react"
import { uploadDocument, type DocumentExtractOutput } from "@/lib/api"
import { useAppStore } from "@/lib/app-store"
import { useRouter } from "next/navigation"
import { T } from "@/lib/auto-translate"

export default function DocumentsPage() {
  return <AppWrapper><DocumentUpload /></AppWrapper>
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatINR(v: number) {
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(v)
}

function InfoRow({ icon: Icon, label, value, mono = false, color = "" }: {
  icon: React.ElementType; label: string; value?: string | number | null; mono?: boolean; color?: string
}) {
  return (
    <div className="rounded-xl bg-muted p-3">
      <div className="mb-1 flex items-center gap-2">
        <Icon className="h-4 w-4 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <p className={cn("text-sm font-semibold text-foreground", mono && "font-mono tracking-widest", color)}>
        {value ?? <span className="text-muted-foreground font-normal italic">Not detected</span>}
      </p>
    </div>
  )
}

const DOC_LABEL: Record<string, string> = {
  pan_card: "PAN Card", aadhaar: "Aadhaar Card",
  cibil_report: "CIBIL Report", salary_slip: "Salary Slip",
  bank_statement: "Bank Statement", financial: "Financial Document",
}

const DOC_ICON: Record<string, React.ElementType> = {
  pan_card: CreditCard, aadhaar: User, cibil_report: Star,
  salary_slip: BadgeIndianRupee, bank_statement: Receipt, financial: FileText,
}

// ── Per-document result card ──────────────────────────────────────────────────

function DocResultCard({ data, onRemove }: { data: DocumentExtractOutput; onRemove: () => void }) {
  const Icon = DOC_ICON[data.document_type] ?? FileText
  const [open, setOpen] = useState(true)

  return (
    <div className="rounded-2xl border border-border bg-card shadow-sm">
      <div className="flex items-center gap-3 p-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
          <Icon className="h-5 w-5 text-primary" />
        </div>
        <div className="flex-1">
          <p className="font-semibold text-foreground">{DOC_LABEL[data.document_type]}</p>
          <p className={cn("text-xs font-medium capitalize",
            data.confidence === "high" ? "text-success" :
            data.confidence === "medium" ? "text-warning" : "text-muted-foreground"
          )}>{data.confidence} confidence</p>
        </div>
        <button onClick={() => setOpen(v => !v)} className="text-xs text-muted-foreground hover:text-foreground mr-1">
          {open ? "hide" : "show"}
        </button>
        <button onClick={onRemove} className="text-muted-foreground hover:text-destructive">
          <X className="h-4 w-4" />
        </button>
      </div>

      {open && (
        <div className="border-t border-border px-4 pb-4 pt-3">
          {data.document_type === "pan_card" && (
            <div className="grid gap-2 sm:grid-cols-2">
              <InfoRow icon={CreditCard} label="PAN Number"    value={data.pan_number} mono />
              <InfoRow icon={User}       label="Name"          value={data.pan_name} />
              <InfoRow icon={Calendar}   label="Date of Birth" value={data.pan_dob} />
              <InfoRow icon={Users}      label="Father's Name" value={data.pan_father_name} />
              {data.age_from_dob && (
                <div className="sm:col-span-2 text-xs text-success font-medium">
                  Age: <strong>{data.age_from_dob} yrs</strong> — will be auto-filled
                </div>
              )}
            </div>
          )}

          {data.document_type === "aadhaar" && (
            <div className="grid gap-2 sm:grid-cols-2">
              <InfoRow icon={User}     label="Name"          value={data.aadhaar_name} />
              <InfoRow icon={Calendar} label="Date of Birth" value={data.aadhaar_dob} />
              <InfoRow icon={Users}    label="Gender"        value={data.aadhaar_gender} />
              {data.aadhaar_address && <InfoRow icon={Building2} label="Address" value={data.aadhaar_address} />}
              {data.age_from_dob && (
                <div className="sm:col-span-2 text-xs text-success font-medium">
                  Age: <strong>{data.age_from_dob} yrs</strong> — will be auto-filled
                </div>
              )}
            </div>
          )}

          {data.document_type === "cibil_report" && (
            <div className="space-y-2">
              {data.cibil_score && (
                <div className="flex items-center justify-between rounded-xl bg-muted px-4 py-3">
                  <span className="text-sm text-muted-foreground">CIBIL Score</span>
                  <span className={cn("text-2xl font-bold",
                    data.cibil_score >= 750 ? "text-success" :
                    data.cibil_score >= 650 ? "text-warning" : "text-destructive"
                  )}>{data.cibil_score}</span>
                </div>
              )}
              <div className="grid gap-2 sm:grid-cols-2">
                <InfoRow icon={BadgeIndianRupee} label="Total Outstanding"   value={data.total_outstanding ? formatINR(data.total_outstanding) : null} />
                <InfoRow icon={Receipt}          label="Total Monthly EMI"   value={data.total_monthly_emi ? formatINR(data.total_monthly_emi) : null} />
                <InfoRow icon={FileText}         label="Active Accounts"     value={data.active_loans_count ?? null} />
                <InfoRow icon={Calendar}         label="Report Date"         value={data.cibil_report_date ?? null} />
              </div>
            </div>
          )}

          {data.document_type === "salary_slip" && (
            <div className="grid gap-2 sm:grid-cols-2">
              <InfoRow icon={TrendingUp}       label="Gross Salary"     value={data.gross_salary ? formatINR(data.gross_salary) : null} />
              <InfoRow icon={Wallet}           label="Net Salary"       value={data.net_salary ? formatINR(data.net_salary) : null} />
              <InfoRow icon={BadgeIndianRupee} label="Basic Salary"     value={data.basic_salary ? formatINR(data.basic_salary) : null} />
              <InfoRow icon={TrendingDown}     label="Total Deductions" value={data.total_deductions ? formatINR(data.total_deductions) : null} />
              <InfoRow icon={Building2}        label="Employer"         value={data.employer_name} />
              <InfoRow icon={Calendar}         label="Salary Month"     value={data.salary_month} />
            </div>
          )}

          {(data.document_type === "bank_statement" || data.document_type === "financial") && (
            <div className="grid gap-2 sm:grid-cols-2">
              <InfoRow icon={TrendingUp}   label="Estimated Income"   value={data.estimated_income ? formatINR(data.estimated_income) : null} />
              <InfoRow icon={TrendingDown} label="Monthly Expenses"   value={data.monthly_expenses ? formatINR(data.monthly_expenses) : null} />
              <InfoRow icon={Receipt}      label="Transactions Found" value={data.transactions.length} />
              <InfoRow icon={Wallet}       label="Net Savings"
                value={data.estimated_income && data.monthly_expenses
                  ? formatINR(data.estimated_income - data.monthly_expenses) : null} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Merge extracted data from all docs into URL params ────────────────────────

function buildParams(docs: DocumentExtractOutput[]): URLSearchParams {
  const p = new URLSearchParams()
  const get = (type: string) => docs.find(d => d.document_type === type)

  const pan    = get("pan_card")
  const aadh   = get("aadhaar")
  const cibil  = get("cibil_report")
  const salary = get("salary_slip")
  const bank   = get("bank_statement") ?? get("financial")

  // Age — PAN > Aadhaar
  const age = pan?.age_from_dob ?? aadh?.age_from_dob
  if (age) p.set("age", String(age))

  // CIBIL
  if (cibil?.cibil_score)       p.set("cibilScore",    String(cibil.cibil_score))
  if (cibil?.total_outstanding) p.set("existingLoans", String(cibil.total_outstanding))
  if (cibil?.total_monthly_emi) p.set("emi",           String(cibil.total_monthly_emi))

  // Income — salary > bank
  const income = salary?.estimated_income ?? bank?.estimated_income
  if (income) p.set("income", String(income))

  // Expenses — bank statement
  if (bank?.monthly_expenses) p.set("expenses", String(bank.monthly_expenses))

  return p
}

// ── Main component ────────────────────────────────────────────────────────────

function DocumentUpload() {
  const { t } = useLanguage()
  const router = useRouter()
  const { loanRisk } = useAppStore()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [uploads, setUploads] = useState<{ file: File; data: DocumentExtractOutput | null; error: string | null; loading: boolean }[]>([])
  const [dragActive, setDragActive] = useState(false)

  const processFile = async (file: File, idx: number) => {
    setUploads(prev => prev.map((u, i) => i === idx ? { ...u, loading: true, error: null } : u))
    try {
      const data = await uploadDocument(file)
      setUploads(prev => prev.map((u, i) => i === idx ? { ...u, data, loading: false } : u))
    } catch (err: unknown) {
      setUploads(prev => prev.map((u, i) => i === idx ? {
        ...u, loading: false, error: err instanceof Error ? err.message : "Failed to extract"
      } : u))
    }
  }

  const addFiles = (files: FileList | File[]) => {
    const valid = Array.from(files).filter(f => {
      const ok = ["application/pdf","image/png","image/jpeg","image/jpg"].includes(f.type)
        && f.size <= 10 * 1024 * 1024
      return ok
    })
    const startIdx = uploads.length
    const newEntries = valid.map(f => ({ file: f, data: null, error: null, loading: true }))
    setUploads(prev => [...prev, ...newEntries])
    valid.forEach((f, i) => processFile(f, startIdx + i))
  }

  const removeUpload = (idx: number) => {
    setUploads(prev => prev.filter((_, i) => i !== idx))
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation()
    setDragActive(e.type === "dragenter" || e.type === "dragover")
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setDragActive(false)
    if (e.dataTransfer.files) addFiles(e.dataTransfer.files)
  }

  const handleInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(e.target.files)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const completed = uploads.filter(u => u.data)
  const loading   = uploads.some(u => u.loading)
  const params    = buildParams(completed.map(u => u.data!))

  const handleFillForm = () => {
    router.push(`/loan-risk?${params.toString()}`)
  }

  // What will be filled
  const preview: { label: string; value: string }[] = []
  if (params.get("age"))          preview.push({ label: "Age",             value: `${params.get("age")} yrs` })
  if (params.get("cibilScore"))   preview.push({ label: "CIBIL Score",     value: params.get("cibilScore")! })
  if (params.get("existingLoans"))preview.push({ label: "Existing Loans",  value: formatINR(+params.get("existingLoans")!) })
  if (params.get("emi"))          preview.push({ label: "Monthly EMI",     value: formatINR(+params.get("emi")!) })
  if (params.get("income"))       preview.push({ label: "Monthly Income",  value: formatINR(+params.get("income")!) })
  if (params.get("expenses"))     preview.push({ label: "Monthly Expenses",value: formatINR(+params.get("expenses")!) })

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-balance text-3xl font-bold tracking-tight text-foreground">
          {t("documentUpload")}
        </h1>
        <T as="p" className="mt-2 text-muted-foreground">Upload all your documents at once — we&apos;ll extract and combine everything</T>
      </div>

      {/* Document type guide — what FinAI extracts */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {[
          { Icon: CreditCard,       label: "PAN Card",       fills: "Name · DOB · PAN No." },
          { Icon: User,             label: "Aadhaar",         fills: "Name · DOB · Address" },
          { Icon: Star,             label: "CIBIL Report",    fills: "Score · EMI · Balance" },
          { Icon: BadgeIndianRupee, label: "Salary Slip",     fills: "Gross · Net · Employer" },
          { Icon: FileText,         label: "Bank Statement",  fills: "Income · Expenses" },
        ].map(({ Icon, label, fills }) => (
          <div key={label} className="flex flex-col items-center rounded-xl bg-muted p-3 text-center gap-1">
            <Icon className="h-6 w-6 text-primary" />
            <p className="text-xs font-semibold text-foreground">{label}</p>
            <p className="text-xs text-muted-foreground">{fills}</p>
          </div>
        ))}
      </div>

      {/* Required documents by loan type */}
      <Card className="border-none shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base"><T>Documents Required by Loan Type</T></CardTitle>
          <CardDescription><T>Have these ready before visiting the bank — reduces processing time significantly</T></CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {[
              {
                type: "Personal Loan",
                color: "bg-blue-500/10 text-blue-600",
                docs: ["Aadhaar Card", "PAN Card", "3 months salary slips", "6 months bank statements", "Employment letter / Offer letter", "Form 16 (last year)"],
              },
              {
                type: "Home Loan",
                color: "bg-green-500/10 text-green-600",
                docs: ["Aadhaar + PAN Card", "3 months salary slips", "6 months bank statements", "ITR last 2 years", "Property documents (sale deed, NOC)", "Approved building plan"],
              },
              {
                type: "Vehicle Loan",
                color: "bg-orange-500/10 text-orange-600",
                docs: ["Aadhaar + PAN Card", "Driving licence", "3 months salary slips or ITR", "Vehicle proforma invoice", "Bank statements (3 months)"],
              },
              {
                type: "Education Loan",
                color: "bg-purple-500/10 text-purple-600",
                docs: ["Aadhaar + PAN (student + co-applicant)", "Admission letter from institute", "Course fee schedule", "Academic records (10th, 12th, graduation)", "Co-applicant income proof (ITR/salary slips)"],
              },
              {
                type: "Business Loan",
                color: "bg-red-500/10 text-red-600",
                docs: ["Aadhaar + PAN (personal + business)", "GST registration", "ITR last 2–3 years", "CA-certified balance sheet", "6–12 months current account statements", "Business ownership proof"],
              },
              {
                type: "Gold Loan",
                color: "bg-yellow-500/10 text-yellow-600",
                docs: ["Aadhaar Card", "PAN Card", "Gold ornaments / coins to pledge", "2 passport photographs", "No income proof required", "No CIBIL score required"],
              },
            ].map(({ type, color, docs }) => (
              <div key={type} className="space-y-2">
                <div className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${color}`}>
                  {type}
                </div>
                <ul className="space-y-1.5">
                  {docs.map(doc => (
                    <li key={doc} className="flex items-center gap-2 text-sm text-foreground">
                      <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                      {doc}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <T as="p" className="mt-4 text-xs text-muted-foreground">Upload the documents above and FinAI will automatically extract your name, CIBIL score, income, and more — filling forms instantly.</T>
        </CardContent>
      </Card>

      {/* Drop zone */}
      <div
        className={cn(
          "relative rounded-2xl border-2 border-dashed p-8 transition-colors text-center",
          dragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
        )}
        onDragEnter={handleDrag} onDragLeave={handleDrag}
        onDragOver={handleDrag} onDrop={handleDrop}
      >
        <input
          ref={fileInputRef} type="file" multiple accept=".pdf,.png,.jpg,.jpeg"
          onChange={handleInput}
          className="absolute inset-0 cursor-pointer opacity-0"
        />
        <div className="flex flex-col items-center gap-3">
          <div className="rounded-full bg-primary/10 p-4">
            <Upload className="h-8 w-8 text-primary" />
          </div>
          <p className="text-lg font-medium text-foreground">
            {uploads.length === 0
              ? <T>Drop all your documents here</T>
              : <T>Drop more documents to add</T>}
          </p>
          <T as="p" className="text-sm text-muted-foreground">PDF, PNG, JPG — multiple files at once — up to 10 MB each</T>
          <Button variant="outline" className="pointer-events-none gap-2">
            <Plus className="h-4 w-4" /> <T>Choose Files</T>
          </Button>
        </div>
      </div>

      {/* Upload results */}
      {uploads.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-foreground">
            <T>Uploaded Documents</T> ({completed.length}/{uploads.length} <T>processed</T>)
          </h2>

          {uploads.map((u, idx) => (
            <div key={idx}>
              {u.loading && (
                <div className="flex items-center gap-3 rounded-2xl border border-border bg-card p-4">
                  <Spinner className="h-5 w-5 text-primary" />
                  <div>
                    <p className="text-sm font-medium text-foreground">{u.file.name}</p>
                    <T as="p" className="text-xs text-muted-foreground">Analysing…</T>
                  </div>
                  <button onClick={() => removeUpload(idx)} className="ml-auto text-muted-foreground hover:text-destructive">
                    <X className="h-4 w-4" />
                  </button>
                </div>
              )}
              {u.error && (
                <div className="flex items-center gap-3 rounded-2xl border border-destructive/30 bg-destructive/5 p-4">
                  <AlertCircle className="h-5 w-5 text-destructive shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-foreground">{u.file.name}</p>
                    <p className="text-xs text-destructive">{u.error}</p>
                  </div>
                  <button onClick={() => removeUpload(idx)} className="ml-auto text-muted-foreground hover:text-destructive">
                    <X className="h-4 w-4" />
                  </button>
                </div>
              )}
              {u.data && (
                <DocResultCard data={u.data} onRemove={() => removeUpload(idx)} />
              )}
            </div>
          ))}
        </div>
      )}

      {/* Combined fill preview + CTA */}
      {preview.length > 0 && (
        <Card className="border-none shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-success" />
              <T>Ready to Auto-fill Loan Form</T>
            </CardTitle>
            <CardDescription><T>These fields will be filled from your documents</T></CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-3">
              {preview.map(({ label, value }) => (
                <div key={label} className="rounded-xl bg-muted p-3 text-center">
                  <p className="text-xs text-muted-foreground">{label}</p>
                  <p className="text-base font-bold text-foreground">{value}</p>
                </div>
              ))}
            </div>
            <Button
              className="h-14 w-full text-lg font-semibold"
              onClick={handleFillForm}
              disabled={loading}
            >
              <T>Auto-fill Loan Risk Form</T>
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
