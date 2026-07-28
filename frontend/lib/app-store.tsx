"use client"

import { createContext, useContext, useState, useEffect, type ReactNode } from "react"
import type { LoanRiskOutput, RoadmapOutput, ChatMessage } from "@/lib/api"

interface LoanRiskState {
  formData: {
    income: string
    expenses: string
    existingLoans: string
    emi: string
    loanAmount: string
    tenure: string
    cibilScore: string
    age: string
    loanPurpose: string
    employmentType: string
    stabilityYears: string
  }
  result: LoanRiskOutput | null
}

interface RoadmapState {
  formData: {
    income: string
    expenses: string
    existingLoans: string
    emi: string
    loanAmount: string
    tenure: string
  }
  result: RoadmapOutput | null
}

interface GoldLoanState {
  goldWeightGrams: string
  goldPurityKarats: string
  loanAmount: string
}

interface AppStore {
  loanRisk: LoanRiskState
  setLoanRisk: (s: LoanRiskState) => void
  roadmap: RoadmapState
  setRoadmap: (s: RoadmapState) => void
  goldLoan: GoldLoanState
  setGoldLoan: (s: GoldLoanState) => void
  chatHistory: ChatMessage[]
  setChatHistory: (h: ChatMessage[]) => void
  simulatorValues: { income: number; emi: number; expenses: number; existingLoans: number }
  setSimulatorValues: (v: AppStore["simulatorValues"]) => void
  userPhone: string
  setUserPhone: (p: string) => void
  pendingLoanRiskSubmit: boolean
  setPendingLoanRiskSubmit: (v: boolean) => void
}

const defaultLoanRisk: LoanRiskState = {
  formData: { income: "", expenses: "", existingLoans: "", emi: "", loanAmount: "", tenure: "", cibilScore: "", age: "", loanPurpose: "", employmentType: "salaried", stabilityYears: "" },
  result: null,
}

const defaultRoadmap: RoadmapState = {
  formData: { income: "", expenses: "", existingLoans: "", emi: "", loanAmount: "", tenure: "" },
  result: null,
}

const defaultGoldLoan: GoldLoanState = {
  goldWeightGrams: "",
  goldPurityKarats: "22",
  loanAmount: "",
}

const AppContext = createContext<AppStore | null>(null)

function loadFromStorage<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

export function AppStoreProvider({ children }: { children: ReactNode }) {
  const [loanRisk, setLoanRiskState] = useState<LoanRiskState>(defaultLoanRisk)
  const [roadmap, setRoadmapState] = useState<RoadmapState>(defaultRoadmap)
  const [goldLoan, setGoldLoanState] = useState<GoldLoanState>(defaultGoldLoan)
  const [chatHistory, setChatHistoryState] = useState<ChatMessage[]>([])
  const [simulatorValues, setSimulatorState] = useState(
    { income: 50000, emi: 15000, expenses: 20000, existingLoans: 100000 }
  )
  const [userPhone, setUserPhoneState] = useState("")
  const [pendingLoanRiskSubmit, setPendingLoanRiskSubmitState] = useState(false)
  const [hydrated, setHydrated] = useState(false)

  // Load from localStorage on mount
  useEffect(() => {
    setLoanRiskState(loadFromStorage("finedge_loanrisk", defaultLoanRisk))
    setRoadmapState(loadFromStorage("finedge_roadmap", defaultRoadmap))
    setGoldLoanState(loadFromStorage("finedge_goldloan", defaultGoldLoan))
    setChatHistoryState(loadFromStorage("finedge_chat", []))
    setSimulatorState(loadFromStorage("finedge_simulator", { income: 50000, emi: 15000, expenses: 20000, existingLoans: 100000 }))
    setUserPhoneState(loadFromStorage("finedge_phone", ""))
    setHydrated(true)
  }, [])

  const setLoanRisk = (s: LoanRiskState) => {
    setLoanRiskState(s)
    localStorage.setItem("finedge_loanrisk", JSON.stringify(s))
  }
  const setRoadmap = (s: RoadmapState) => {
    setRoadmapState(s)
    localStorage.setItem("finedge_roadmap", JSON.stringify(s))
  }
  const setGoldLoan = (s: GoldLoanState) => {
    setGoldLoanState(s)
    localStorage.setItem("finedge_goldloan", JSON.stringify(s))
  }
  const setChatHistory = (h: ChatMessage[]) => {
    setChatHistoryState(h)
    localStorage.setItem("finedge_chat", JSON.stringify(h))
  }
  const setSimulatorValues = (v: AppStore["simulatorValues"]) => {
    setSimulatorState(v)
    localStorage.setItem("finedge_simulator", JSON.stringify(v))
  }

  const setUserPhone = (p: string) => {
    setUserPhoneState(p)
    localStorage.setItem("finedge_phone", JSON.stringify(p))
  }

  const setPendingLoanRiskSubmit = (v: boolean) => {
    setPendingLoanRiskSubmitState(v)
  }

  if (!hydrated) return null

  return (
    <AppContext.Provider value={{
      loanRisk, setLoanRisk,
      roadmap, setRoadmap,
      goldLoan, setGoldLoan,
      chatHistory, setChatHistory,
      simulatorValues, setSimulatorValues,
      userPhone, setUserPhone,
      pendingLoanRiskSubmit, setPendingLoanRiskSubmit,
    }}>
      {children}
    </AppContext.Provider>
  )
}

export function useAppStore() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error("useAppStore must be used within AppStoreProvider")
  return ctx
}
