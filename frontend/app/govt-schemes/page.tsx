"use client"

import { useState } from "react"
import { AppWrapper } from "@/components/app-wrapper"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import {
  Sprout, GraduationCap, Home, Briefcase, Car,
  Users, ArrowUpRight, ChevronDown, ChevronUp, Info,
  BadgeCheck, Landmark, Globe,
} from "lucide-react"

// ── Scheme data ───────────────────────────────────────────────────────────────

interface Scheme {
  id: string
  name: string
  fullName: string
  category: string
  bank: string
  rate: string
  maxAmount: string
  maxTenure: string
  eligibility: string[]
  benefits: string[]
  documents: string[]
  applyUrl: string
  tag?: string
}

const schemes: Scheme[] = [
  // Agriculture
  {
    id: "kcc",
    name: "Kisan Credit Card",
    fullName: "Kisan Credit Card (KCC) Scheme",
    category: "agriculture",
    bank: "All Banks / NABARD",
    rate: "4–7% p.a. (after Govt subsidy)",
    maxAmount: "₹3 lakh (no collateral) / higher with land",
    maxTenure: "12 months (revolving)",
    eligibility: ["Farmers, sharecroppers, tenant farmers", "Allied activities (fisheries, animal husbandry)"],
    benefits: ["Govt pays 2% interest subvention", "ATM card for withdrawals", "Crop insurance included", "No processing fee below ₹3 lakh"],
    documents: ["Aadhaar", "Land ownership proof or lease agreement", "Khasra/Khatauni", "Passport photo"],
    applyUrl: "https://www.nabard.org",
    tag: "Subsidised",
  },
  {
    id: "pmfby",
    name: "PM Fasal Bima Yojana",
    fullName: "Pradhan Mantri Fasal Bima Yojana",
    category: "agriculture",
    bank: "Empanelled Insurance Companies",
    rate: "Premium: 1.5–5% of sum insured",
    maxAmount: "Based on crop value / state norms",
    maxTenure: "Per season",
    eligibility: ["All farmers growing notified crops", "Compulsory for KCC borrowers"],
    benefits: ["Govt pays 90%+ of premium", "Covers drought, flood, pest, cyclone", "Settlement within 2 months of harvest"],
    documents: ["Aadhaar", "Land record / KCC details", "Bank account"],
    applyUrl: "https://pmfby.gov.in",
    tag: "Insurance",
  },
  // Home
  {
    id: "pmay",
    name: "PMAY – CLSS",
    fullName: "PM Awas Yojana – Credit Linked Subsidy",
    category: "home",
    bank: "All scheduled banks / HFCs",
    rate: "6.5% effective (after upfront subsidy)",
    maxAmount: "₹6–12 lakh subsidy (NPV)",
    maxTenure: "20 years",
    eligibility: ["First-time home buyers only", "EWS: income < ₹3 lakh/yr", "LIG: ₹3–6 lakh", "MIG-I: ₹6–12 lakh", "MIG-II: ₹12–18 lakh"],
    benefits: ["Upfront interest subsidy credited to loan account", "Reduces EMI significantly", "No upper limit on loan amount"],
    documents: ["Aadhaar", "Income certificate", "Property documents", "No previous pucca house affidavit"],
    applyUrl: "https://pmaymis.gov.in",
    tag: "Popular",
  },
  // Business
  {
    id: "mudra-shishu",
    name: "MUDRA – Shishu",
    fullName: "Pradhan Mantri MUDRA Yojana – Shishu",
    category: "business",
    bank: "All Banks / MFIs / NBFCs",
    rate: "8–12% p.a.",
    maxAmount: "₹50,000",
    maxTenure: "5 years",
    eligibility: ["Micro-enterprises, startups", "Non-farm income generating activities", "No collateral required"],
    benefits: ["No collateral", "No processing fee (most banks)", "Govt guarantee via MUDRA Ltd"],
    documents: ["Aadhaar", "PAN", "Business proof / plan", "Bank statements (6 months)"],
    applyUrl: "https://www.mudra.org.in",
    tag: "No Collateral",
  },
  {
    id: "mudra-kishore",
    name: "MUDRA – Kishore",
    fullName: "Pradhan Mantri MUDRA Yojana – Kishore",
    category: "business",
    bank: "All Banks / MFIs",
    rate: "9–14% p.a.",
    maxAmount: "₹5 lakh",
    maxTenure: "5 years",
    eligibility: ["Existing micro/small businesses", "Expansion or upgrade of existing business"],
    benefits: ["No collateral", "CIBIL score not mandatory for small amounts", "Quick 7-day approval at many PSU banks"],
    documents: ["Aadhaar", "PAN", "2 years business proof", "IT returns / bank statements"],
    applyUrl: "https://www.mudra.org.in",
  },
  {
    id: "mudra-tarun",
    name: "MUDRA – Tarun",
    fullName: "Pradhan Mantri MUDRA Yojana – Tarun",
    category: "business",
    bank: "All Scheduled Banks",
    rate: "10–14% p.a.",
    maxAmount: "₹10 lakh",
    maxTenure: "5 years",
    eligibility: ["Established micro/small businesses", "Manufacturing, trading, services"],
    benefits: ["CGTMSE government guarantee", "No personal security required", "Interest subvention for SC/ST/OBC"],
    documents: ["Aadhaar", "PAN", "3 years IT returns", "Audited balance sheet", "Business registration"],
    applyUrl: "https://www.mudra.org.in",
  },
  {
    id: "cgtmse",
    name: "CGTMSE",
    fullName: "Credit Guarantee Fund Trust for Micro & Small Enterprises",
    category: "business",
    bank: "All Scheduled Commercial Banks",
    rate: "10–14% p.a.",
    maxAmount: "₹2 crore",
    maxTenure: "7 years",
    eligibility: ["MSMEs (new or existing)", "Loans for business purposes only", "No agricultural/residential loans"],
    benefits: ["85% govt guarantee (no personal security)", "Covers collateral-free term loans and working capital", "Both new and expansion projects eligible"],
    documents: ["Aadhaar", "PAN", "MSME registration (Udyam)", "Project report / CMA data", "IT returns (3 years)"],
    applyUrl: "https://www.cgtmse.in",
    tag: "Guarantee",
  },
  // Education
  {
    id: "vidya-lakshmi",
    name: "Vidya Lakshmi",
    fullName: "Vidya Lakshmi Education Loan Portal",
    category: "education",
    bank: "38+ banks via single portal",
    rate: "8.5–12% p.a. (0.5% concession for girls)",
    maxAmount: "₹7.5 lakh (no collateral) / ₹40 lakh (with collateral)",
    maxTenure: "15 years",
    eligibility: ["Indian nationals with admission letter", "Courses: UG, PG, professional, vocational", "Abroad studies also eligible"],
    benefits: ["Apply to 3 banks with one form", "Moratorium: course duration + 1 year", "Govt interest subvention for EWS (income < ₹4.5 lakh/yr)"],
    documents: ["Aadhaar", "PAN", "Admission letter", "Fee structure", "Income proof", "Mark sheets"],
    applyUrl: "https://www.vidyalakshmi.co.in",
    tag: "Portal",
  },
  {
    id: "pmsss",
    name: "PM Scholarship",
    fullName: "PM Scholarship Scheme (PMSS) – for wards of ex-defence personnel",
    category: "education",
    bank: "KSB / Central Govt",
    rate: "Scholarship (not loan)",
    maxAmount: "₹2,500–3,000/month",
    maxTenure: "Course duration",
    eligibility: ["Wards and widows of ex-servicemen", "First degree professional courses only", "Min 60% in 10+2"],
    benefits: ["Non-repayable scholarship", "Paid directly to student's bank account", "Covers engineering, MBA, medical, etc."],
    documents: ["Aadhaar", "ESM certificate", "12th marksheet", "Admission letter", "Bank account"],
    applyUrl: "https://ksb.gov.in",
  },
  // Personal / Other
  {
    id: "jan-samarth",
    name: "Jan Samarth",
    fullName: "Jan Samarth – Govt Loan Portal",
    category: "personal",
    bank: "All participating banks",
    rate: "As per individual scheme",
    maxAmount: "Varies by scheme",
    maxTenure: "Varies",
    eligibility: ["Any Indian citizen — eligibility checked per scheme"],
    benefits: ["Single portal for 13 Govt credit-linked schemes", "Instant eligibility check", "Digital end-to-end application"],
    documents: ["Aadhaar", "PAN (scheme-specific)"],
    applyUrl: "https://www.jansamarth.in",
    tag: "New",
  },
  {
    id: "svnidhi",
    name: "PM SVANidhi",
    fullName: "PM Street Vendor's AtmaNirbhar Nidhi",
    category: "personal",
    bank: "All scheduled banks / MFIs",
    rate: "7% effective (after 7% interest subsidy)",
    maxAmount: "₹50,000 (in 3 tranches: ₹10k → ₹20k → ₹50k)",
    maxTenure: "12 months per tranche",
    eligibility: ["Street vendors with valid vending certificate", "Urban street vendors only"],
    benefits: ["Interest subsidy of 7% credited quarterly", "Cash-back reward for digital payments", "Credit history building"],
    documents: ["Aadhaar", "Vending certificate / Letter of Recommendation from ULB"],
    applyUrl: "https://pmsvanidhi.mohua.gov.in",
    tag: "Street Vendors",
  },
]

const categories = [
  { id: "all", label: "All Schemes", icon: Globe },
  { id: "agriculture", label: "Agriculture", icon: Sprout },
  { id: "home", label: "Home Loan", icon: Home },
  { id: "business", label: "Business / MSME", icon: Briefcase },
  { id: "education", label: "Education", icon: GraduationCap },
  { id: "personal", label: "Personal / Other", icon: Users },
]

// ── Components ────────────────────────────────────────────────────────────────

function SchemeCard({ scheme }: { scheme: Scheme }) {
  const [expanded, setExpanded] = useState(false)

  const catColor: Record<string, string> = {
    agriculture: "bg-green-500/10 text-green-600 dark:text-green-400",
    home: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
    business: "bg-orange-500/10 text-orange-600 dark:text-orange-400",
    education: "bg-purple-500/10 text-purple-600 dark:text-purple-400",
    personal: "bg-pink-500/10 text-pink-600 dark:text-pink-400",
  }

  return (
    <Card className="border-none shadow-md hover:shadow-lg transition-shadow">
      <CardContent className="pt-5 pb-4 space-y-3">

        {/* Header row */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className={cn("rounded-full px-2 py-0.5 text-xs font-semibold capitalize", catColor[scheme.category] ?? "bg-muted")}>
                {scheme.category}
              </span>
              {scheme.tag && (
                <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-semibold text-primary">
                  {scheme.tag}
                </span>
              )}
            </div>
            <h3 className="font-bold text-foreground text-base leading-tight">{scheme.name}</h3>
            <p className="text-xs text-muted-foreground mt-0.5">{scheme.fullName}</p>
          </div>
          <Landmark className="h-8 w-8 text-muted-foreground/40 shrink-0 mt-1" />
        </div>

        {/* Key stats */}
        <div className="grid grid-cols-3 gap-2">
          <div className="rounded-lg bg-muted px-3 py-2">
            <p className="text-xs text-muted-foreground">Interest Rate</p>
            <p className="text-xs font-semibold text-foreground mt-0.5">{scheme.rate}</p>
          </div>
          <div className="rounded-lg bg-muted px-3 py-2">
            <p className="text-xs text-muted-foreground">Max Amount</p>
            <p className="text-xs font-semibold text-foreground mt-0.5">{scheme.maxAmount}</p>
          </div>
          <div className="rounded-lg bg-muted px-3 py-2">
            <p className="text-xs text-muted-foreground">Bank / Source</p>
            <p className="text-xs font-semibold text-foreground mt-0.5">{scheme.bank}</p>
          </div>
        </div>

        {/* Expand / Collapse */}
        <button
          onClick={() => setExpanded(v => !v)}
          className="flex items-center gap-1 text-xs text-primary font-medium hover:underline"
        >
          {expanded ? <><ChevronUp className="h-3.5 w-3.5" /> Hide details</> : <><ChevronDown className="h-3.5 w-3.5" /> View eligibility &amp; benefits</>}
        </button>

        {expanded && (
          <div className="space-y-3 pt-1 border-t border-border">
            <div>
              <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-1.5">Eligibility</p>
              <ul className="space-y-1">
                {scheme.eligibility.map((e, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-xs text-foreground">
                    <BadgeCheck className="h-3.5 w-3.5 text-success shrink-0 mt-0.5" />
                    {e}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-1.5">Key Benefits</p>
              <ul className="space-y-1">
                {scheme.benefits.map((b, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-xs text-foreground">
                    <span className="text-primary mt-0.5">•</span>
                    {b}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-1.5">Documents Needed</p>
              <div className="flex flex-wrap gap-1.5">
                {scheme.documents.map((d, i) => (
                  <span key={i} className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">{d}</span>
                ))}
              </div>
            </div>
            <a href={scheme.applyUrl} target="_blank" rel="noopener noreferrer">
              <Button size="sm" className="w-full gap-1.5 mt-1">
                Apply / Learn More <ArrowUpRight className="h-3.5 w-3.5" />
              </Button>
            </a>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

function GovtSchemesContent() {
  const [activeCategory, setActiveCategory] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")

  const filtered = schemes.filter(s => {
    const matchCat = activeCategory === "all" || s.category === activeCategory
    const q = searchQuery.toLowerCase()
    const matchSearch = !q
      || s.name.toLowerCase().includes(q)
      || s.fullName.toLowerCase().includes(q)
      || s.eligibility.some(e => e.toLowerCase().includes(q))
      || s.benefits.some(b => b.toLowerCase().includes(q))
    return matchCat && matchSearch
  })

  return (
    <div className="space-y-8">

      {/* Header */}
      <div>
        <p className="mb-1 text-xs font-semibold uppercase tracking-widest text-muted-foreground">Government Schemes</p>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Subsidised Loan Schemes</h1>
        <p className="mt-2 text-muted-foreground">
          Central government schemes with interest subsidies, guarantees, or special eligibility — all in one place.
        </p>
      </div>

      {/* Info banner */}
      <div className="flex items-start gap-3 rounded-xl border border-primary/30 bg-primary/5 px-4 py-3">
        <Info className="h-4 w-4 text-primary shrink-0 mt-0.5" />
        <p className="text-sm text-foreground">
          These are genuine Govt of India schemes. Rates shown are as per official portals (April 2026). Always verify current rates at the respective bank or portal before applying.
        </p>
      </div>

      {/* Search */}
      <input
        type="text"
        placeholder="Search schemes, eligibility, benefits…"
        value={searchQuery}
        onChange={e => setSearchQuery(e.target.value)}
        className="w-full rounded-xl border border-input bg-background px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
      />

      {/* Category filter */}
      <div className="flex flex-wrap gap-2">
        {categories.map(cat => {
          const Icon = cat.icon
          const count = cat.id === "all" ? schemes.length : schemes.filter(s => s.category === cat.id).length
          return (
            <button
              key={cat.id}
              onClick={() => setActiveCategory(cat.id)}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-medium transition-colors",
                activeCategory === cat.id
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:bg-muted/80"
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {cat.label}
              <span className={cn("rounded-full px-1.5 py-0.5 text-xs",
                activeCategory === cat.id ? "bg-primary-foreground/20 text-primary-foreground" : "bg-background text-muted-foreground"
              )}>
                {count}
              </span>
            </button>
          )
        })}
      </div>

      {/* Results count */}
      <p className="text-sm text-muted-foreground">
        Showing <span className="font-semibold text-foreground">{filtered.length}</span> scheme{filtered.length !== 1 ? "s" : ""}
      </p>

      {/* Scheme grid */}
      {filtered.length > 0 ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {filtered.map(s => <SchemeCard key={s.id} scheme={s} />)}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Globe className="h-12 w-12 text-muted-foreground mb-3" />
          <p className="text-muted-foreground">No schemes match your search. Try different keywords.</p>
        </div>
      )}

      {/* Footer link */}
      <div className="rounded-xl border border-border bg-muted/50 p-4 text-center">
        <p className="text-sm text-muted-foreground mb-2">Looking for more? The Govt&apos;s official portal has 13+ credit-linked schemes.</p>
        <a href="https://www.jansamarth.in" target="_blank" rel="noopener noreferrer">
          <Button variant="outline" className="gap-2">
            <Globe className="h-4 w-4" /> Visit JanSamarth.in <ArrowUpRight className="h-4 w-4" />
          </Button>
        </a>
      </div>
    </div>
  )
}

export default function GovtSchemesPage() {
  return <AppWrapper><GovtSchemesContent /></AppWrapper>
}
