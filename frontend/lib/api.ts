const API_URL = "/api"

// Map frontend language codes to backend-supported ones
// ks (Kashmiri), mni (Manipuri), kok (Konkani) fall back to Hindi
const LANG_MAP: Record<string, string> = {
  ks: "hi", mni: "hi", kok: "hi",
}
export function toBackendLang(lang: string): string {
  return LANG_MAP[lang] ?? lang
}

// ─── Types ────────────────────────────────────────────────────────────────────

export interface LoanRiskInput {
  monthly_income: number
  monthly_expenses: number
  existing_loans: number
  emi_amount: number
  repayment_history_score?: number
  loan_amount_requested: number
  loan_tenure_months: number
  employment_type?: string
  language?: string
  cibil_score?: number
  age?: number
  loan_purpose?: string
  employment_stability_years?: number
}

export interface FactorScore {
  name: string
  label: string
  score: number       // 0–100, higher = worse
  weight: number      // percentage weight (sums to 100)
  status: "good" | "fair" | "poor"
}

export interface LoanRiskOutput {
  risk_score: number
  risk_category: "Low" | "Medium" | "High"
  explanation: string
  key_factors: string[]
  recommendation: string
  debt_to_income_ratio: number
  emi_to_income_ratio: number
  factor_breakdown: FactorScore[]
  ai_advice?: string
}

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export interface UserFinancialContext {
  monthly_income?: number
  monthly_expenses?: number
  existing_loans?: number
  emi_amount?: number
  loan_amount_requested?: number
  loan_tenure_months?: number
  risk_score?: number
  risk_category?: string
  cibil_score?: number
  age?: number
  loan_purpose?: string
  employment_type?: string
  employment_stability_years?: number
  gold_weight_grams?: number
  gold_purity_karats?: number
}

export interface ChatInput {
  message: string
  language?: string
  conversation_history?: ChatMessage[]
  return_audio?: boolean
  user_context?: UserFinancialContext
}

export interface ChatAction {
  navigate?: "loan-risk" | "roadmap" | "simulate" | "reminders" | "chat" | "bank-rates" | "gold-loan"
  prefill_loan_risk?: {
    income?: string
    expenses?: string
    existingLoans?: string
    emi?: string
    loanAmount?: string
    tenure?: string
  }
  prefill_gold_loan?: {
    goldWeightGrams?: string
    goldPurityKarats?: string
    loanAmount?: string
  }
  save_user_data?: {
    income?: string | null
    expenses?: string | null
    existingLoans?: string | null
    emi?: string | null
    loanAmount?: string | null
    tenure?: string | null
    employmentType?: string | null
    cibilScore?: string | null
    age?: string | null
    loanPurpose?: string | null
    stabilityYears?: string | null
    goldWeightGrams?: string | null
    goldPurityKarats?: string | null
  }
  submit_loan_risk?: boolean
  set_phone?: string
}

export interface ChatOutput {
  reply: string
  detected_language: string
  audio_base64?: string
  conversation_history: ChatMessage[]
  actions: ChatAction[]
}

export interface RoadmapInput {
  monthly_income: number
  monthly_expenses: number
  existing_loans: number
  emi_amount: number
  loan_amount_requested: number
  loan_tenure_months: number
  risk_score?: number
  language?: string
}

export interface MonthlyPlanItem {
  month: number
  opening_balance: number
  emi_payment: number
  closing_balance: number
}

export interface RoadmapOutput {
  repayment_plan: MonthlyPlanItem[]
  expense_reduction_tips: string[]
  income_improvement_tips: string[]
  summary: string
  total_interest_payable: number
  suggested_emi: number
}

export interface ReminderInput {
  phone_number: string
  message: string
  remind_at: string
  repeat?: string
}

export interface ReminderOutput {
  job_id: string
  status: string
  remind_at: string
  message: string
}

export interface Transaction {
  date?: string
  description: string
  amount: number
  type: string
}

export interface DocumentExtractOutput {
  document_type: "pan_card" | "aadhaar" | "cibil_report" | "salary_slip" | "bank_statement" | "financial"
  confidence: string
  raw_text_preview: string
  // financial / bank statement
  estimated_income?: number
  monthly_expenses?: number
  transactions: Transaction[]
  // PAN card
  pan_number?: string
  pan_name?: string
  pan_dob?: string
  pan_father_name?: string
  // Aadhaar
  aadhaar_name?: string
  aadhaar_dob?: string
  aadhaar_gender?: string
  aadhaar_address?: string
  // shared identity
  age_from_dob?: number
  // CIBIL report
  cibil_score?: number
  cibil_report_date?: string
  active_loans_count?: number
  total_outstanding?: number
  total_monthly_emi?: number
  credit_accounts: { type: string; outstanding?: number }[]
  // salary slip
  gross_salary?: number
  net_salary?: number
  basic_salary?: number
  total_deductions?: number
  employer_name?: string
  salary_month?: string
}

// ─── API Functions ────────────────────────────────────────────────────────────

export async function predictRisk(data: LoanRiskInput): Promise<LoanRiskOutput> {
  const res = await fetch(`${API_URL}/predict-risk`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Failed to predict risk")
  }
  return res.json()
}

export async function sendChat(data: ChatInput): Promise<ChatOutput> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Failed to get chat response")
  }
  return res.json()
}

export async function generateRoadmap(data: RoadmapInput): Promise<RoadmapOutput> {
  const res = await fetch(`${API_URL}/roadmap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Failed to generate roadmap")
  }
  return res.json()
}

export async function setReminder(data: ReminderInput): Promise<ReminderOutput> {
  const res = await fetch(`${API_URL}/set-reminder`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Failed to set reminder")
  }
  return res.json()
}

export async function getReminders(): Promise<ReminderOutput[]> {
  const res = await fetch(`${API_URL}/set-reminder`)
  if (!res.ok) throw new Error("Failed to fetch reminders")
  return res.json()
}

export async function deleteReminder(jobId: string): Promise<void> {
  const res = await fetch(`${API_URL}/set-reminder/${jobId}`, { method: "DELETE" })
  if (!res.ok) throw new Error("Failed to delete reminder")
}

export async function sendVoiceChat(
  audio: Blob,
  language: string,
  conversationHistory: ChatMessage[] = [],
  userContext: UserFinancialContext = {},
): Promise<ChatOutput & { transcribed_text: string }> {
  const formData = new FormData()
  formData.append("audio", audio, "recording.webm")
  formData.append("language", toBackendLang(language))
  formData.append("return_audio", "true")
  formData.append("conversation_history", JSON.stringify(conversationHistory))
  formData.append("user_context", JSON.stringify(userContext))
  const res = await fetch(`${API_URL}/chat/voice`, {
    method: "POST",
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Voice chat failed")
  }
  return res.json()
}

export interface WhatsAppJoinInfo {
  configured: boolean
  link: string | null
  sandbox_number: string
  join_text?: string
  message?: string
}

export async function getWhatsAppJoinLink(): Promise<WhatsAppJoinInfo> {
  const res = await fetch(`${API_URL}/send-whatsapp/join-link`)
  if (!res.ok) throw new Error("Failed to fetch join link")
  return res.json()
}

// ─── Bank Rates ───────────────────────────────────────────────────────────────

export interface BankRate {
  bank: string
  bank_short: string
  type: string
  logo_color: string
  min_rate: number
  max_rate: number
  rate_for_score: number
  max_loan: string
  max_tenure_years: number
  processing_fee: string
  eligible: boolean
  eligibility_note: string
  recommended: boolean
  rank: number
  documents: string[]
  features: string[]
}

export interface BankRatesResponse {
  loan_purpose: string
  cibil_score: number
  loan_amount: number
  monthly_emi: number
  rates: BankRate[]
  best_pick: string
  summary: string
}

export interface ContactBankRequest {
  bank: string
  loan_purpose: string
  loan_amount: number
  tenure_months: number
  income: number
  expenses: number
  existing_loans: number
  employment_type: string
  cibil_score?: number
  age?: number
}

export interface ContactBankResponse {
  bank: string
  inquiry_text: string
  bank_response: string
  estimated_rate: number
  estimated_emi: number
  next_steps: string[]
}

export async function getBankRates(
  cibil_score: number,
  loan_amount: number,
  loan_purpose: string,
  tenure_months: number,
): Promise<BankRatesResponse> {
  const params = new URLSearchParams({
    cibil_score: String(cibil_score),
    loan_amount: String(loan_amount),
    loan_purpose,
    tenure_months: String(tenure_months),
  })
  const res = await fetch(`${API_URL}/bank-rates?${params}`)
  if (!res.ok) throw new Error("Failed to fetch bank rates")
  return res.json()
}

export async function contactBank(data: ContactBankRequest): Promise<ContactBankResponse> {
  const res = await fetch(`${API_URL}/bank-rates/contact`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error("Failed to contact bank")
  return res.json()
}

export async function uploadDocument(file: File): Promise<DocumentExtractOutput> {
  const formData = new FormData()
  formData.append("file", file)
  const res = await fetch(`${API_URL}/upload-document`, {
    method: "POST",
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Failed to extract document")
  }
  return res.json()
}
