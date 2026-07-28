"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { useRouter } from "next/navigation"
import { useLanguage } from "@/lib/language-context"
import { AppWrapper } from "@/components/app-wrapper"
import { T, useTrans, DynamicText } from "@/lib/auto-translate"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { Send, Mic, Volume2, VolumeX, Bot, User, Zap, Trash2 } from "lucide-react"
import {
  sendChat, sendVoiceChat, toBackendLang,
  type ChatMessage, type UserFinancialContext, type ChatAction,
} from "@/lib/api"
import { useAppStore } from "@/lib/app-store"

const WELCOME_TEXT =
  "Hello! I'm your FinEdge loan assistant. " +
  "To assess your loan eligibility, please share: " +
  "(1) Monthly income, (2) Monthly expenses, (3) Total existing loans, " +
  "(4) Current EMI, (5) Loan amount needed, (6) Repayment period in months."

const VOICE_PLACEHOLDER = "\u{1F3A4} Processing voice..."

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  audio_base64?: string
  isVoice?: boolean
}

function historyToMessages(history: ChatMessage[]): Message[] {
  const welcome: Message = {
    id: "welcome",
    role: "assistant",
    content: WELCOME_TEXT,
    timestamp: new Date(0), // stable timestamp so it doesn't jump on remount
  }
  if (history.length === 0) return [welcome]
  return [
    welcome,
    ...history.map((m, i) => ({
      id: `h${i}`,
      role: m.role as "user" | "assistant",
      content: m.content,
      timestamp: new Date(0),
    })),
  ]
}

export default function ChatPage() {
  return (
    <AppWrapper>
      <ChatInterface />
    </AppWrapper>
  )
}

function ChatInterface() {
  const { t, language } = useLanguage()
  const router = useRouter()
  const {
    chatHistory, setChatHistory,
    loanRisk, setLoanRisk,
    roadmap: storedRoadmap, setRoadmap,
    goldLoan, setGoldLoan,
    setUserPhone,
  } = useAppStore()

  // Keep refs to latest state to avoid stale closures in executeActions
  const loanRiskRef = useRef(loanRisk)
  useEffect(() => { loanRiskRef.current = loanRisk }, [loanRisk])
  const roadmapRef = useRef(storedRoadmap)
  useEffect(() => { roadmapRef.current = storedRoadmap }, [storedRoadmap])
  const goldLoanRef = useRef(goldLoan)
  useEffect(() => { goldLoanRef.current = goldLoan }, [goldLoan])

  // ── Message display state ───────────────────────────────────────────────────
  // Initialised from persisted chatHistory so remounting restores the conversation
  const [messages, setMessages] = useState<Message[]>(() => historyToMessages(chatHistory))
  const [conversationHistory, setConversationHistory] = useState<ChatMessage[]>(chatHistory)

  // Sync display if the store was hydrated after the component mounted
  // (edge case: context loads from localStorage asynchronously)
  const syncedRef = useRef(false)
  useEffect(() => {
    if (!syncedRef.current && chatHistory.length > 0) {
      syncedRef.current = true
      setMessages(historyToMessages(chatHistory))
      setConversationHistory(chatHistory)
    }
  }, [chatHistory])

  const [input, setInput] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessingVoice, setIsProcessingVoice] = useState(false)
  const [autoSpeak, setAutoSpeak] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [playingId, setPlayingId] = useState<string | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const speechRecognitionRef = useRef<any>(null)
  const currentAudioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // ── Build user context from stored financial data ───────────────────────────
  const getUserContext = useCallback((): UserFinancialContext => {
    const ctx: UserFinancialContext = {}
    const f = loanRiskRef.current.formData
    if (f.income)          ctx.monthly_income              = parseFloat(f.income)
    if (f.expenses)        ctx.monthly_expenses            = parseFloat(f.expenses)
    if (f.existingLoans)   ctx.existing_loans              = parseFloat(f.existingLoans)
    if (f.emi)             ctx.emi_amount                  = parseFloat(f.emi)
    if (f.loanAmount)      ctx.loan_amount_requested       = parseFloat(f.loanAmount)
    if (f.tenure)          ctx.loan_tenure_months          = parseInt(f.tenure)
    if (f.cibilScore)      ctx.cibil_score                 = parseInt(f.cibilScore)
    if (f.age)             ctx.age                         = parseInt(f.age)
    if (f.loanPurpose)     ctx.loan_purpose                = f.loanPurpose
    if (f.employmentType)  ctx.employment_type             = f.employmentType
    if (f.stabilityYears)  ctx.employment_stability_years  = parseFloat(f.stabilityYears)
    const gl = goldLoanRef.current
    if (gl.goldWeightGrams)  ctx.gold_weight_grams   = parseFloat(gl.goldWeightGrams)
    if (gl.goldPurityKarats) ctx.gold_purity_karats  = parseInt(gl.goldPurityKarats)
    const r = loanRiskRef.current.result
    if (r) { ctx.risk_score = r.risk_score; ctx.risk_category = r.risk_category }
    return ctx
  }, [])

  // ── TTS ─────────────────────────────────────────────────────────────────────
  const BROWSER_LANG_MAP: Record<string, string> = {
    en: "en-IN", hi: "hi-IN", ta: "ta-IN", te: "te-IN",
    kn: "kn-IN", mr: "mr-IN", bn: "bn-IN", ml: "ml-IN",
    gu: "gu-IN", ur: "ur-PK", pa: "pa-IN", as: "as-IN",
    or: "or-IN", ks: "hi-IN", mni: "hi-IN", kok: "hi-IN",
  }
  const speakWithBrowser = useCallback((text: string) => {
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = BROWSER_LANG_MAP[language] ?? "en-IN"
    utterance.rate = 0.9
    window.speechSynthesis.speak(utterance)
  }, [language])

  useEffect(() => {
    const last = messages[messages.length - 1]
    if (!last || last.role !== "assistant" || !autoSpeak) return
    if (last.audio_base64) playAudio(last.id, last.audio_base64)
    else if (last.content && !last.content.startsWith("🎤")) speakWithBrowser(last.content)
  }, [messages, autoSpeak])

  const playAudio = (messageId: string, audio_base64: string) => {
    if (currentAudioRef.current) { currentAudioRef.current.pause(); currentAudioRef.current = null }
    if (playingId === messageId) { setPlayingId(null); return }
    const audio = new Audio(`data:audio/mp3;base64,${audio_base64}`)
    currentAudioRef.current = audio
    setPlayingId(messageId)
    audio.onended = () => { setPlayingId(null); currentAudioRef.current = null }
    audio.play().catch(() => setPlayingId(null))
  }

  // ── Execute app-control actions from the AI ─────────────────────────────────
  const executeActions = useCallback((actions: ChatAction[]) => {
    if (!actions?.length) return

    let navigatePage: string | null = null
    let prefill: ChatAction["prefill_loan_risk"] | null = null
    let autosubmit = false

    for (const action of actions) {
      // Persist financial values into loanRisk + roadmap + goldLoan stores immediately
      if (action.save_user_data) {
        const sd = action.save_user_data
        const cur = loanRiskRef.current
        setLoanRisk({
          formData: {
            income:         sd.income         ?? cur.formData.income,
            expenses:       sd.expenses       ?? cur.formData.expenses,
            existingLoans:  sd.existingLoans  ?? cur.formData.existingLoans,
            emi:            sd.emi            ?? cur.formData.emi,
            loanAmount:     sd.loanAmount     ?? cur.formData.loanAmount,
            tenure:         sd.tenure         ?? cur.formData.tenure,
            cibilScore:     sd.cibilScore     ?? cur.formData.cibilScore,
            age:            sd.age            ?? cur.formData.age,
            loanPurpose:    sd.loanPurpose    ?? cur.formData.loanPurpose,
            employmentType: sd.employmentType ?? cur.formData.employmentType,
            stabilityYears: sd.stabilityYears ?? cur.formData.stabilityYears,
          },
          result: cur.result,
        })
        // Sync shared fields into roadmap store
        const rm = roadmapRef.current
        setRoadmap({
          formData: {
            income:        sd.income        ?? rm.formData.income,
            expenses:      sd.expenses      ?? rm.formData.expenses,
            existingLoans: sd.existingLoans ?? rm.formData.existingLoans,
            emi:           sd.emi           ?? rm.formData.emi,
            loanAmount:    sd.loanAmount    ?? rm.formData.loanAmount,
            tenure:        sd.tenure        ?? rm.formData.tenure,
          },
          result: rm.result,
        })
        // Sync gold fields into goldLoan store
        if (sd.goldWeightGrams || sd.goldPurityKarats) {
          const gl = goldLoanRef.current
          setGoldLoan({
            goldWeightGrams:  sd.goldWeightGrams  ?? gl.goldWeightGrams,
            goldPurityKarats: sd.goldPurityKarats ?? gl.goldPurityKarats,
            loanAmount:       sd.loanAmount       ?? gl.loanAmount,
          })
        }
      }
      if (action.prefill_loan_risk) prefill = action.prefill_loan_risk
      if (action.submit_loan_risk)  autosubmit = true
      if (action.navigate)          navigatePage = action.navigate
      if (action.set_phone)         setUserPhone(action.set_phone)
    }

    if (navigatePage) {
      if (navigatePage === "loan-risk" && prefill) {
        const params = new URLSearchParams()
        if (prefill.income)        params.set("income",        prefill.income)
        if (prefill.expenses)      params.set("expenses",      prefill.expenses)
        if (prefill.existingLoans) params.set("existingLoans", prefill.existingLoans)
        if (prefill.emi)           params.set("emi",           prefill.emi)
        if (prefill.loanAmount)    params.set("loanAmount",    prefill.loanAmount)
        if (prefill.tenure)        params.set("tenure",        prefill.tenure)
        if (autosubmit)            params.set("autosubmit",    "1")
        setTimeout(() => router.push(`/loan-risk?${params.toString()}`), 150)
      } else if (navigatePage === "bank-rates") {
        // Build URL params from the now-updated loanRisk store
        const f = loanRiskRef.current.formData
        const params = new URLSearchParams()
        if (f.cibilScore)   params.set("cibilScore",  f.cibilScore)
        if (f.loanAmount)   params.set("loanAmount",  f.loanAmount)
        if (f.loanPurpose)  params.set("loanPurpose", f.loanPurpose)
        if (f.tenure)       params.set("tenure",      f.tenure)
        setTimeout(() => router.push(`/bank-rates?${params.toString()}`), 150)
      } else if (navigatePage === "gold-loan") {
        setTimeout(() => router.push("/gold-loan"), 150)
      } else {
        setTimeout(() => router.push(`/${navigatePage}`), 150)
      }
    }
  }, [setLoanRisk, setRoadmap, setGoldLoan, setUserPhone, router])

  // ── Shared helper to persist a completed exchange ───────────────────────────
  const persistExchange = useCallback((newHistory: ChatMessage[]) => {
    setConversationHistory(newHistory)
    setChatHistory(newHistory) // writes to localStorage
  }, [setChatHistory])

  const addMessage = (msg: Message) => setMessages((prev) => [...prev, msg])

  // ── Send text message ────────────────────────────────────────────────────────
  const handleSend = async () => {
    if (!input.trim() || isTyping) return
    const userText = input.trim()
    const userMsg: Message = { id: Date.now().toString(), role: "user", content: userText, timestamp: new Date() }
    addMessage(userMsg)
    setInput("")
    setIsTyping(true)
    setError(null)

    try {
      const response = await sendChat({
        message: userText,
        language: toBackendLang(language),
        conversation_history: conversationHistory,
        return_audio: false,
        user_context: getUserContext(),
      })
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.reply,
        timestamp: new Date(),
        audio_base64: response.audio_base64 ?? undefined,
      }
      persistExchange(response.conversation_history)
      addMessage(aiMsg)
      if (response.actions?.length) executeActions(response.actions)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to get response")
      setMessages((prev) => prev.filter((m) => m.id !== userMsg.id))
    } finally {
      setIsTyping(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  // ── Voice recording ──────────────────────────────────────────────────────────

  const BROWSER_STT_LANG: Record<string, string> = {
    en: "en-IN", hi: "hi-IN", ta: "ta-IN", te: "te-IN",
    kn: "kn-IN", mr: "mr-IN", bn: "bn-IN", ml: "ml-IN",
    gu: "gu-IN", ur: "ur-PK", pa: "pa-IN", as: "as-IN",
    or: "or-IN", ks: "hi-IN", mni: "hi-IN", kok: "hi-IN",
  }

  /** Send a transcribed text string through the chat pipeline */
  const handleTranscribedText = async (transcript: string) => {
    setIsProcessingVoice(true)
    const placeholderId = Date.now().toString()
    addMessage({ id: placeholderId, role: "user", content: `🎤 ${transcript}`, timestamp: new Date(), isVoice: true })
    try {
      const { sendChat } = await import("@/lib/api")
      const response = await sendChat({
        message: transcript,
        language: toBackendLang(language) as any,
        conversation_history: conversationHistory,
        return_audio: autoSpeak,
        user_context: getUserContext(),
      })
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.reply,
        timestamp: new Date(),
        audio_base64: response.audio_base64 ?? undefined,
      }
      persistExchange(response.conversation_history)
      addMessage(aiMsg)
      if (response.actions?.length) executeActions(response.actions)
    } catch (err: unknown) {
      setMessages((prev) => prev.filter((m) => m.id !== placeholderId))
      setError(err instanceof Error ? err.message : "Voice processing failed")
    } finally {
      setIsProcessingVoice(false)
    }
  }

  /** Start MediaRecorder (hold-to-record mode) */
  const startMediaRecorder = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true }
      })
      audioChunksRef.current = []
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm"
      const recorder = new MediaRecorder(stream, { mimeType })
      recorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data) }
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        await handleVoiceBlob(new Blob(audioChunksRef.current, { type: "audio/webm" }))
      }
      mediaRecorderRef.current = recorder
      recorder.start()
      setIsRecording(true)
      setError(null)
    } catch {
      setError("Microphone access denied.")
    }
  }

  /** Primary voice path: browser Web Speech API → falls back to MediaRecorder on error */
  const startBrowserSpeechRecognition = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    const recognition: any = new SpeechRecognition()
    recognition.lang = BROWSER_STT_LANG[language] ?? "en-IN"
    recognition.interimResults = false
    recognition.maxAlternatives = 1
    recognition.continuous = false

    let resultReceived = false
    let errorHandled = false

    recognition.onstart = () => { setIsRecording(true); setError(null) }

    recognition.onresult = async (e: any) => {
      resultReceived = true
      const transcript = e.results[0][0].transcript.trim()
      if (!transcript) return
      setIsRecording(false)
      await handleTranscribedText(transcript)
    }

    recognition.onerror = async (e: any) => {
      errorHandled = true
      setIsRecording(false)
      speechRecognitionRef.current = null
      if (e.error === "not-allowed" || e.error === "audio-capture") {
        setError("Microphone access denied. Please allow mic access and try again.")
      } else if (e.error === "no-speech") {
        setError("No speech detected. Please speak closer to your microphone.")
      } else if (e.error === "aborted") {
        // user cancelled — do nothing
      } else {
        // network / service-not-allowed / unknown → fall back to MediaRecorder silently
        await startMediaRecorder()
      }
    }

    recognition.onend = () => {
      setIsRecording(false)
      speechRecognitionRef.current = null
      // Only show hint if nothing else already handled this session
      if (!resultReceived && !errorHandled) {
        setError("No speech detected. Tap the mic and speak clearly.")
      }
    }

    speechRecognitionRef.current = recognition
    recognition.start()
  }

  const startRecording = async () => {
    setError(null)
    const hasSpeechAPI =
      typeof window !== "undefined" &&
      ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition)

    if (hasSpeechAPI) {
      startBrowserSpeechRecognition()
    } else {
      await startMediaRecorder()
    }
  }

  const stopRecording = () => {
    if (speechRecognitionRef.current && isRecording) {
      speechRecognitionRef.current.stop()
      speechRecognitionRef.current = null
      return
    }
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const cancelRecording = () => {
    if (speechRecognitionRef.current) {
      speechRecognitionRef.current.abort()
      speechRecognitionRef.current = null
      setIsRecording(false)
      return
    }
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.onstop = () => mediaRecorderRef.current?.stream?.getTracks().forEach((t) => t.stop())
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      audioChunksRef.current = []
    }
  }

  /** Fallback: send recorded blob to backend for Groq/Gemini transcription */
  const handleVoiceBlob = async (audioBlob: Blob) => {
    setIsProcessingVoice(true)
    setError(null)
    const placeholderId = Date.now().toString()
    addMessage({ id: placeholderId, role: "user", content: VOICE_PLACEHOLDER, timestamp: new Date(), isVoice: true })

    try {
      const response = await sendVoiceChat(audioBlob, language, conversationHistory, getUserContext())
      setMessages((prev) =>
        prev.map((m) => m.id === placeholderId ? { ...m, content: `🎤 ${response.transcribed_text}` } : m)
      )
      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.reply,
        timestamp: new Date(),
        audio_base64: response.audio_base64 ?? undefined,
      }
      persistExchange(response.conversation_history)
      addMessage(aiMsg)
      if (response.actions?.length) executeActions(response.actions)
    } catch (err: unknown) {
      setMessages((prev) => prev.filter((m) => m.id !== placeholderId))
      setError(err instanceof Error ? err.message : "Voice processing failed")
    } finally {
      setIsProcessingVoice(false)
    }
  }

  // ── Quick reply suggestions ──────────────────────────────────────────────────
  const quickReplies = loanRisk.result
    ? ["Show me the repayment plan", "Contact a bank for me", "How can I improve my score?"]
    : loanRisk.formData.income
    ? ["Calculate my risk now", "What loan schemes am I eligible for?"]
    : ["My income is ₹40,000/month", "My income is ₹75,000/month", "My income is ₹1,50,000/month"]

  return (
    <div className="flex h-[calc(100vh-7rem)] flex-col">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary">
            <Bot className="h-6 w-6 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-foreground">{t("aiChat")}</h1>
            <p className="text-sm text-muted-foreground">
              {isRecording ? <T>🔴 Recording...</T> : <T>Your loan application guide</T>}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => setAutoSpeak((v) => !v)} className="gap-2">
            {autoSpeak ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
            <span className="hidden sm:inline">{autoSpeak ? <T>Voice ON</T> : <T>Voice OFF</T>}</span>
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="gap-2 text-destructive hover:bg-destructive/10 hover:text-destructive"
            onClick={() => {
              setChatHistory([])
              setConversationHistory([])
              setMessages(historyToMessages([]))
              syncedRef.current = true
            }}
          >
            <Trash2 className="h-4 w-4" />
            <span className="hidden sm:inline"><T>Clear</T></span>
          </Button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto rounded-2xl bg-card p-4 shadow-lg">
        <div className="space-y-4">
          {messages.map((message) => (
            <ChatMessageRow
              key={message.id}
              message={message}
              playAudio={playAudio}
              speakWithBrowser={speakWithBrowser}
            />
          ))}

          {(isTyping || isProcessingVoice) && (
            <div className="flex gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-secondary">
                <Bot className="h-5 w-5 text-secondary-foreground" />
              </div>
              <div className="flex items-center gap-1 rounded-2xl bg-muted px-4 py-3">
                <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]" />
                <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]" />
                <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground" />
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-xl bg-destructive/10 px-4 py-3 text-sm text-destructive">
              <DynamicText text={error} />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Quick replies */}
      <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
        {quickReplies.map((qr) => (
          <QuickReplyChip
            key={qr}
            text={qr}
            onPick={(text) => { setInput(text); setTimeout(() => inputRef.current?.focus(), 50) }}
          />
        ))}
      </div>

      {/* Input */}
      <div className="mt-2 flex items-center gap-2">
        {isRecording ? (
          <>
            <Button variant="outline" className="h-12 flex-1 rounded-full" onClick={cancelRecording}><T>Cancel</T></Button>
            <div className="flex h-12 flex-1 items-center justify-center rounded-full bg-destructive/10 text-destructive text-sm font-medium animate-pulse">
              <T>🔴 Recording...</T>
            </div>
            <Button className="h-12 flex-1 rounded-full" onClick={stopRecording}><T>Send</T></Button>
          </>
        ) : (
          <>
            <Button
              variant="outline" size="icon"
              className="h-12 w-12 shrink-0 rounded-full"
              onClick={startRecording}
              disabled={isTyping || isProcessingVoice}
            >
              <Mic className="h-5 w-5" />
            </Button>
            <div className="relative flex-1">
              <Input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={t("typeMessage")}
                className="h-12 pr-12 text-base"
                disabled={isTyping || isProcessingVoice}
              />
              <Button
                size="icon"
                className="absolute right-1 top-1/2 h-10 w-10 -translate-y-1/2 rounded-full"
                onClick={handleSend}
                disabled={!input.trim() || isTyping || isProcessingVoice}
              >
                <Send className="h-5 w-5" />
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function UserMessageBody({ content }: { content: string }) {
  const processing = useTrans(VOICE_PLACEHOLDER)
  const shown = content === VOICE_PLACEHOLDER ? processing : content
  return <span className="whitespace-pre-wrap">{shown}</span>
}

function AssistantMessageRow({
  message,
  playAudio,
  speakWithBrowser,
}: {
  message: Message
  playAudio: (messageId: string, audio_base64: string) => void
  speakWithBrowser: (text: string) => void
}) {
  const display = useTrans(message.content)
  const readAloudLabel = useTrans("Read aloud")
  return (
    <div className="flex gap-3 flex-row">
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-secondary">
        <Bot className="h-5 w-5 text-secondary-foreground" />
      </div>
      <div className="max-w-[80%] rounded-2xl bg-muted px-4 py-3 text-foreground">
        <p className="text-sm leading-relaxed whitespace-pre-wrap">{display}</p>
        <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
          <button
            type="button"
            onClick={() =>
              message.audio_base64                ? playAudio(message.id, message.audio_base64)
                : speakWithBrowser(display)
            }
            className="rounded-full p-1 transition-colors hover:bg-muted-foreground/20"
            aria-label={readAloudLabel}
          >
            <Volume2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

function ChatMessageRow({
  message,
  playAudio,
  speakWithBrowser,
}: {
  message: Message
  playAudio: (messageId: string, audio_base64: string) => void
  speakWithBrowser: (text: string) => void
}) {
  if (message.role === "assistant") {
    return (
      <AssistantMessageRow
        message={message}
        playAudio={playAudio}
        speakWithBrowser={speakWithBrowser}
      />
    )
  }
  return (
    <div className="flex gap-3 flex-row-reverse">
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary">
        <User className="h-5 w-5 text-primary-foreground" />
      </div>
      <div className="max-w-[80%] rounded-2xl bg-primary px-4 py-3 text-primary-foreground">
        <p className="text-sm leading-relaxed">
          <UserMessageBody content={message.content} />
        </p>
      </div>
    </div>
  )
}

function QuickReplyChip({ text, onPick }: { text: string; onPick: (t: string) => void }) {
  const label = useTrans(text)
  return (
    <button
      type="button"
      onClick={() => onPick(label)}
      className="flex shrink-0 items-center gap-1 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary hover:text-primary"
    >
      <Zap className="h-3 w-3" />
      {label}
    </button>
  )
}
