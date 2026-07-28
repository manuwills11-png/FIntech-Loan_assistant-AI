"use client"

import { useState, useRef, useEffect } from "react"
import { AppWrapper } from "@/components/app-wrapper"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Spinner } from "@/components/ui/spinner"
import { cn } from "@/lib/utils"
import { useAppStore } from "@/lib/app-store"
import { useLanguage } from "@/lib/language-context"
import { toBackendLang } from "@/lib/api"
import {
  Handshake, Send, ChevronDown, ChevronUp, Lightbulb,
  TrendingDown, Copy, CheckCheck, AlertCircle, Scale,
} from "lucide-react"

const API_URL = "/api"

interface NegotiationProfile {
  income: string
  existingEmi: string
  cibilScore: string
  loanAmount: string
  tenure: string
  loanPurpose: string
  employmentType: string
  offeredRate: string
  targetBank: string
  competitorRate: string
}

interface Message {
  role: "user" | "assistant"
  content: string
}

const TACTICS = [
  { label: "Counter-offer script", prompt: "Give me a word-for-word counter-offer script I can say to the banker right now." },
  { label: "Leverage my CIBIL", prompt: "What exact leverage does my CIBIL score give me? What rate can I realistically demand?" },
  { label: "Competitor comparison", prompt: "Draft a message I can show the banker about my competitor bank offer to negotiate a better rate." },
  { label: "Processing fee waiver", prompt: "How do I negotiate the processing fee waiver? What arguments work best?" },
  { label: "Prepayment terms", prompt: "What prepayment and foreclosure clause should I ask for? What is standard vs what I should push for?" },
  { label: "Risk-based pricing ask", prompt: "My risk profile is strong. Draft a formal request for risk-based preferential pricing." },
]

// ── Negotiation engine ────────────────────────────────────────────────────────

async function getNegotiationAdvice(
  messages: Message[],
  profile: NegotiationProfile,
  language: string,
): Promise<string> {
  const systemPrompt = `You are an expert loan negotiation coach helping an Indian borrower negotiate better terms with a bank.
You have deep knowledge of Indian banking regulations, RBI guidelines, and how banks price loans.

Borrower profile:
- Monthly Income: ₹${profile.income || "not provided"}
- Existing EMI: ₹${profile.existingEmi || "0"}
- CIBIL Score: ${profile.cibilScore || "not provided"}
- Loan Amount: ₹${profile.loanAmount || "not provided"}
- Tenure: ${profile.tenure || "not provided"} months
- Purpose: ${profile.loanPurpose || "personal"}
- Employment: ${profile.employmentType || "salaried"}
- Bank's Offered Rate: ${profile.offeredRate ? profile.offeredRate + "%" : "not provided"}
- Target Bank: ${profile.targetBank || "not specified"}
- Competitor's Rate: ${profile.competitorRate ? profile.competitorRate + "%" : "not provided"}

Your job:
1. Assess the borrower's negotiation leverage (CIBIL, income stability, competitor offers, loan size)
2. Give specific, actionable scripts and tactics — not generic advice
3. Quote real RBI/bank policies when relevant
4. Be direct — tell them exactly what to say, what to ask for, and what they can realistically get
5. If they have weak leverage, tell them honestly what can still be improved

Always respond in ${language === "en" ? "English" : language}. Be concise and practical.`

  const body = {
    model: "llama-3.3-70b-versatile",
    messages: [
      { role: "system", content: systemPrompt },
      ...messages.map(m => ({ role: m.role, content: m.content })),
    ],
    temperature: 0.5,
    max_tokens: 800,
  }

  const groqKey = ""  // routed via backend
  const res = await fetch(`${API_URL}/chat/negotiate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, profile, language }),
  })
  if (!res.ok) throw new Error("Failed to get negotiation advice")
  const data = await res.json()
  return data.reply
}

// ── Message bubble ────────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: Message }) {
  const [copied, setCopied] = useState(false)
  const isUser = msg.role === "user"

  const copy = () => {
    navigator.clipboard.writeText(msg.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={cn("flex gap-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary">
          <Scale className="h-4 w-4 text-primary-foreground" />
        </div>
      )}
      <div className={cn(
        "max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
        isUser
          ? "bg-primary text-primary-foreground rounded-tr-sm"
          : "bg-card border border-border rounded-tl-sm shadow-sm"
      )}>
        <p className="whitespace-pre-wrap">{msg.content}</p>
        {!isUser && (
          <button onClick={copy} className="mt-2 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
            {copied ? <><CheckCheck className="h-3 w-3" /> Copied</> : <><Copy className="h-3 w-3" /> Copy</>}
          </button>
        )}
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

function NegotiateContent() {
  const { loanRisk } = useAppStore()
  const { language } = useLanguage()
  const lang = toBackendLang(language)

  const [profile, setProfile] = useState<NegotiationProfile>({
    income:           loanRisk.formData.income || "",
    existingEmi:      loanRisk.formData.emi || "",
    cibilScore:       loanRisk.formData.cibilScore || "",
    loanAmount:       loanRisk.formData.loanAmount || "",
    tenure:           loanRisk.formData.tenure || "",
    loanPurpose:      loanRisk.formData.loanPurpose || "",
    employmentType:   loanRisk.formData.employmentType || "salaried",
    offeredRate:      "",
    targetBank:       "",
    competitorRate:   "",
  })
  const [showProfile, setShowProfile] = useState(true)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const profileFilled = !!(profile.income && profile.loanAmount && profile.cibilScore)

  const handleSend = async (text?: string) => {
    const msg = (text ?? input).trim()
    if (!msg || isLoading) return
    setInput("")
    setError(null)

    const updated: Message[] = [...messages, { role: "user", content: msg }]
    setMessages(updated)
    setIsLoading(true)

    try {
      const res = await fetch(`${API_URL}/chat/negotiate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: updated, profile, language: lang }),
      })
      if (!res.ok) throw new Error("Failed to get negotiation advice")
      const data = await res.json()
      setMessages(prev => [...prev, { role: "assistant", content: data.reply }])
    } catch {
      setError("Could not reach the negotiation assistant. Please try again.")
      setMessages(updated)
    } finally {
      setIsLoading(false)
    }
  }

  const startSession = () => {
    const greeting = `I want to negotiate a better interest rate on my ${profile.loanPurpose || "personal"} loan of ₹${profile.loanAmount} from ${profile.targetBank || "my bank"}. Their offered rate is ${profile.offeredRate ? profile.offeredRate + "%" : "not yet known"}. My CIBIL is ${profile.cibilScore}. What is my negotiation leverage and where do I start?`
    setShowProfile(false)
    handleSend(greeting)
  }

  return (
    <div className="space-y-6">

      {/* Header */}
      <div>
        <p className="mb-1 text-xs font-semibold uppercase tracking-widest text-muted-foreground">AI Negotiation</p>
        <h1 className="text-3xl font-bold tracking-tight text-foreground flex items-center gap-3">
          <Handshake className="h-8 w-8 text-primary" />
          Loan Negotiation Assistant
        </h1>
        <p className="mt-2 text-muted-foreground">
          Get AI-powered scripts, counter-offers, and tactics to negotiate a lower rate, processing fee waiver, or better terms from your bank.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[380px_1fr]">

        {/* Profile panel */}
        <div className="space-y-4">
          <Card className="border-none shadow-lg">
            <CardHeader className="pb-3">
              <button
                onClick={() => setShowProfile(v => !v)}
                className="flex items-center justify-between w-full"
              >
                <CardTitle className="text-base flex items-center gap-2">
                  <Scale className="h-4 w-4 text-primary" />
                  Your Negotiation Profile
                </CardTitle>
                {showProfile ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
              </button>
              <CardDescription className="text-xs">Fill in your details for personalised tactics</CardDescription>
            </CardHeader>
            {showProfile && (
              <CardContent className="space-y-4 pt-0">
                <div className="grid grid-cols-2 gap-3">
                  {([
                    ["income", "Monthly Income (₹)", "50000"],
                    ["existingEmi", "Existing EMI (₹)", "10000"],
                    ["cibilScore", "CIBIL Score", "750"],
                    ["loanAmount", "Loan Amount (₹)", "500000"],
                    ["tenure", "Tenure (months)", "60"],
                    ["offeredRate", "Bank's Offered Rate (%)", "12.5"],
                  ] as [keyof NegotiationProfile, string, string][]).map(([key, label, ph]) => (
                    <div key={key} className="space-y-1">
                      <Label className="text-xs font-medium">{label}</Label>
                      <Input
                        type="number"
                        placeholder={ph}
                        value={profile[key]}
                        onChange={e => setProfile(p => ({ ...p, [key]: e.target.value }))}
                        className="h-9 text-sm"
                      />
                    </div>
                  ))}
                </div>

                <div className="space-y-1">
                  <Label className="text-xs font-medium">Target Bank</Label>
                  <Input placeholder="e.g. SBI, HDFC, Axis" value={profile.targetBank}
                    onChange={e => setProfile(p => ({ ...p, targetBank: e.target.value }))}
                    className="h-9 text-sm" />
                </div>

                <div className="space-y-1">
                  <Label className="text-xs font-medium">Competitor Rate (%) — if you have one</Label>
                  <Input type="number" placeholder="11.5" value={profile.competitorRate}
                    onChange={e => setProfile(p => ({ ...p, competitorRate: e.target.value }))}
                    className="h-9 text-sm" />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <Label className="text-xs font-medium">Loan Purpose</Label>
                    <select value={profile.loanPurpose}
                      onChange={e => setProfile(p => ({ ...p, loanPurpose: e.target.value }))}
                      className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring">
                      <option value="personal">Personal</option>
                      <option value="home">Home</option>
                      <option value="business">Business</option>
                      <option value="education">Education</option>
                      <option value="vehicle">Vehicle</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs font-medium">Employment</Label>
                    <select value={profile.employmentType}
                      onChange={e => setProfile(p => ({ ...p, employmentType: e.target.value }))}
                      className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring">
                      <option value="salaried">Salaried</option>
                      <option value="self_employed">Self-Employed</option>
                      <option value="farmer">Farmer</option>
                      <option value="student">Student</option>
                    </select>
                  </div>
                </div>

                <Button
                  onClick={startSession}
                  disabled={!profileFilled || isLoading}
                  className="w-full gap-2"
                >
                  <TrendingDown className="h-4 w-4" />
                  Analyse My Leverage &amp; Start
                </Button>
                {!profileFilled && (
                  <p className="text-xs text-muted-foreground text-center">
                    Fill income, loan amount, and CIBIL score to start
                  </p>
                )}
              </CardContent>
            )}
          </Card>

          {/* Quick tactics */}
          {messages.length > 0 && (
            <Card className="border-none shadow-lg">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Lightbulb className="h-4 w-4 text-warning" />
                  Quick Tactics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 pt-0">
                {TACTICS.map(t => (
                  <button
                    key={t.label}
                    onClick={() => handleSend(t.prompt)}
                    disabled={isLoading}
                    className="w-full rounded-lg border border-border bg-muted/50 px-3 py-2 text-left text-xs font-medium text-foreground hover:bg-muted transition-colors disabled:opacity-50"
                  >
                    {t.label}
                  </button>
                ))}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Chat panel */}
        <Card className="border-none shadow-lg flex flex-col" style={{ minHeight: "600px" }}>
          <CardHeader className="pb-3 shrink-0">
            <CardTitle className="text-base">Negotiation Coach</CardTitle>
            <CardDescription>Ask anything — get word-for-word scripts and tactics</CardDescription>
          </CardHeader>

          <CardContent className="flex flex-col flex-1 pt-0 gap-4">

            {messages.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center text-center gap-4 py-8">
                <div className="rounded-full bg-primary/10 p-5">
                  <Handshake className="h-10 w-10 text-primary" />
                </div>
                <div>
                  <p className="font-semibold text-foreground mb-1">Your AI Negotiation Coach</p>
                  <p className="text-sm text-muted-foreground max-w-xs">
                    Fill your profile on the left and click &quot;Analyse My Leverage&quot; to get personalised negotiation scripts — or ask anything directly.
                  </p>
                </div>
                <div className="grid grid-cols-1 gap-2 w-full max-w-sm">
                  {[
                    "What rate can I realistically negotiate to?",
                    "Can I get a processing fee waiver?",
                    "How do I use a competitor offer as leverage?",
                  ].map(q => (
                    <button
                      key={q}
                      onClick={() => handleSend(q)}
                      disabled={isLoading}
                      className="rounded-xl border border-border bg-muted/50 px-4 py-2.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground transition-colors text-left disabled:opacity-50"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto space-y-4 pr-1" style={{ maxHeight: "440px" }}>
                {messages.map((m, i) => <MessageBubble key={i} msg={m} />)}
                {isLoading && (
                  <div className="flex gap-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary">
                      <Scale className="h-4 w-4 text-primary-foreground" />
                    </div>
                    <div className="rounded-2xl rounded-tl-sm border border-border bg-card px-4 py-3 shadow-sm">
                      <div className="flex gap-1">
                        <span className="animate-bounce h-1.5 w-1.5 rounded-full bg-muted-foreground" style={{ animationDelay: "0ms" }} />
                        <span className="animate-bounce h-1.5 w-1.5 rounded-full bg-muted-foreground" style={{ animationDelay: "150ms" }} />
                        <span className="animate-bounce h-1.5 w-1.5 rounded-full bg-muted-foreground" style={{ animationDelay: "300ms" }} />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={bottomRef} />
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2 rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4 shrink-0" />
                {error}
              </div>
            )}

            {/* Input */}
            <div className="flex gap-2 shrink-0">
              <Input
                placeholder="Ask anything — rate tactics, fee waivers, scripts…"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend() } }}
                className="h-11 flex-1"
                disabled={isLoading}
              />
              <Button onClick={() => handleSend()} disabled={!input.trim() || isLoading} size="icon" className="h-11 w-11 shrink-0">
                {isLoading ? <Spinner className="h-4 w-4" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
          </CardContent>
        </Card>

      </div>
    </div>
  )
}

export default function NegotiatePage() {
  return <AppWrapper><NegotiateContent /></AppWrapper>
}
