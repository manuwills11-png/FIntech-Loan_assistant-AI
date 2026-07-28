"use client"

import { useState, useEffect } from "react"
import { useLanguage } from "@/lib/language-context"
import { AppWrapper } from "@/components/app-wrapper"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Spinner } from "@/components/ui/spinner"
import { cn } from "@/lib/utils"
import {
  Bell,
  Calendar,
  Clock,
  MessageCircle,
  CheckCircle2,
  Plus,
  Trash2,
  Smartphone,
  ExternalLink,
  ShieldCheck,
} from "lucide-react"
import { setReminder, getReminders, deleteReminder, getWhatsAppJoinLink, type ReminderOutput, type WhatsAppJoinInfo } from "@/lib/api"
import { useAppStore } from "@/lib/app-store"
import { T } from "@/lib/auto-translate"

export default function RemindersPage() {
  return (
    <AppWrapper>
      <ReminderSetup />
    </AppWrapper>
  )
}

function ReminderSetup() {
  const { t } = useLanguage()
  const { userPhone, setUserPhone } = useAppStore()
  const STORAGE_KEY = "finedge_reminders"

  const loadStoredReminders = (): ReminderOutput[] => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]")
    } catch { return [] }
  }

  const saveStoredReminders = (list: ReminderOutput[]) => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(list)) } catch {}
  }

  const [reminders, setReminders] = useState<ReminderOutput[]>(() => loadStoredReminders())
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [showSuccess, setShowSuccess] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [joinInfo, setJoinInfo] = useState<WhatsAppJoinInfo | null>(null)
  const [newReminder, setNewReminder] = useState({
    title: "",
    phone: userPhone || "",
    date: "",
    time: "",
    whatsappEnabled: !!userPhone,
  })

  useEffect(() => {
    loadReminders()
    getWhatsAppJoinLink().then(setJoinInfo).catch(() => {})
  }, [])

  const loadReminders = async () => {
    setIsLoading(true)
    try {
      const backendJobs = await getReminders()
      // Merge: keep locally-stored reminders whose job_id still exists on the backend.
      // If the backend was restarted (jobs lost), keep local entries so the user can
      // still see and manually dismiss them.
      const backendIds = new Set(backendJobs.map((j: any) => j.job_id))
      const local = loadStoredReminders()
      // Update status from backend where we have a match; keep local-only entries as-is
      const merged = local.map((r) => {
        const live = backendJobs.find((j: any) => j.job_id === r.job_id)
        return live ? { ...r, status: live.status } : r
      })
      // Also add any backend jobs not already in local (shouldn't normally happen)
      backendJobs.forEach((j: any) => {
        if (!merged.find((r) => r.job_id === j.job_id)) merged.push(j)
      })
      setReminders(merged)
      saveStoredReminders(merged)
    } catch {
      // Backend unreachable — show locally-stored reminders
    } finally {
      setIsLoading(false)
    }
  }

  const handleAddReminder = async () => {
    if (!newReminder.title || !newReminder.date || !newReminder.time) return
    if (newReminder.whatsappEnabled && !newReminder.phone) {
      setError("Phone number is required for WhatsApp reminders")
      return
    }

    setIsSaving(true)
    setError(null)

    try {
      const remindAt = `${newReminder.date}T${newReminder.time}:00`
      const phone = newReminder.whatsappEnabled
        ? newReminder.phone.startsWith("+") ? newReminder.phone : `+91${newReminder.phone}`
        : "+910000000000"

      // Save phone globally so other pages can use it
      if (newReminder.whatsappEnabled && newReminder.phone) {
        setUserPhone(newReminder.phone.startsWith("+") ? newReminder.phone : `+91${newReminder.phone}`)
      }

      const result = await setReminder({
        phone_number: phone,
        message: newReminder.title,
        remind_at: remindAt,
      })

      setReminders((prev) => {
        const updated = [...prev, result]
        saveStoredReminders(updated)
        return updated
      })
      setNewReminder({ title: "", phone: "", date: "", time: "", whatsappEnabled: false })
      setShowSuccess(true)
      setTimeout(() => setShowSuccess(false), 3000)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to set reminder")
    } finally {
      setIsSaving(false)
    }
  }

  const handleDeleteReminder = async (jobId: string) => {
    // Optimistically remove from UI immediately
    setReminders((prev) => {
      const updated = prev.filter((r) => r.job_id !== jobId)
      saveStoredReminders(updated)
      return updated
    })
    try {
      await deleteReminder(jobId)
    } catch {
      // 404 = job already gone from backend (e.g. after restart) — UI already cleaned up, that's fine
      // Any other error is also fine to ignore since the UI is already updated
    }
  }

  const formatDateTime = (isoStr: string) => {
    try {
      return new Date(isoStr).toLocaleString("en-IN", {
        weekday: "short",
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      })
    } catch {
      return isoStr
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-balance text-3xl font-bold tracking-tight text-foreground">
          {t("reminders")}
        </h1>
        <T as="p" className="mt-2 text-muted-foreground">Never miss an EMI — a single late payment can drop your CIBIL score by 50–100 points</T>
      </div>

      {/* Why reminders matter */}
      <div className="rounded-xl border border-warning/30 bg-warning/5 p-4">
        <div className="flex items-start gap-3">
          <Bell className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
          <div>
            <T as="p" className="text-sm font-semibold text-foreground">Impact of missed EMIs on your CIBIL score</T>
            <div className="mt-2 grid gap-2 sm:grid-cols-3">
              {[
                { delay: "30 days late",   impact: "−50 to −100 points",  color: "text-warning" },
                { delay: "60 days late",   impact: "−100 to −200 points", color: "text-orange-500" },
                { delay: "90+ days late",  impact: "Marked NPA — severe", color: "text-destructive" },
              ].map(({ delay, impact, color }) => (
                <div key={delay} className="rounded-lg bg-background px-3 py-2">
                  <p className="text-xs text-muted-foreground">{delay}</p>
                  <p className={`text-sm font-semibold ${color}`}>{impact}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Quick preset templates */}
      <div>
        <T as="p" className="mb-3 text-sm font-semibold text-foreground">Quick Add — Common Reminders</T>
        <div className="flex flex-wrap gap-2">
          {[
            { title: "Home Loan EMI",       icon: "🏠" },
            { title: "Personal Loan EMI",   icon: "💳" },
            { title: "Car Loan EMI",        icon: "🚗" },
            { title: "Credit Card Bill",    icon: "💰" },
            { title: "Education Loan EMI",  icon: "🎓" },
            { title: "Gold Loan Interest",  icon: "🪙" },
          ].map(({ title, icon }) => (
            <button
              key={title}
              onClick={() => setNewReminder(prev => ({ ...prev, title: `${icon} ${title}` }))}
              className="flex items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1.5 text-sm font-medium text-foreground transition-colors hover:border-primary/40 hover:bg-primary/5"
            >
              <span>{icon}</span>
              {title}
            </button>
          ))}
        </div>
      </div>

      {/* Success Toast */}
      {showSuccess && (
        <div className="fixed right-4 top-20 z-50 flex items-center gap-2 rounded-xl bg-success px-4 py-3 text-success-foreground shadow-lg md:top-4">
          <CheckCircle2 className="h-5 w-5" />
          <span className="font-medium">{t("confirmReminder")}</span>
        </div>
      )}

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Add New Reminder */}
        <Card className="border-none shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Plus className="h-5 w-5 text-primary" />
              {t("setReminder")}
            </CardTitle>
            <CardDescription><T>Create a new payment or financial reminder</T></CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="title" className="text-base font-medium">
                <T>Reminder Message</T>
              </Label>
              <Input
                id="title"
                placeholder="e.g., EMI Payment Due, Credit Card Bill"
                value={newReminder.title}
                onChange={(e) => setNewReminder((prev) => ({ ...prev, title: e.target.value }))}
                className="h-12"
              />
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="date" className="text-base font-medium"><T>Date</T></Label>
                <div className="relative">
                  <Calendar className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    id="date"
                    type="date"
                    value={newReminder.date}
                    onChange={(e) => setNewReminder((prev) => ({ ...prev, date: e.target.value }))}
                    className="h-12 pl-10"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="time" className="text-base font-medium"><T>Time</T></Label>
                <div className="relative">
                  <Clock className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    id="time"
                    type="time"
                    value={newReminder.time}
                    onChange={(e) => setNewReminder((prev) => ({ ...prev, time: e.target.value }))}
                    className="h-12 pl-10"
                  />
                </div>
              </div>
            </div>

            {/* WhatsApp Toggle */}
            <div className="flex items-center justify-between rounded-xl bg-muted p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-success p-2">
                  <MessageCircle className="h-5 w-5 text-success-foreground" />
                </div>
                <div>
                  <p className="font-medium text-foreground">{t("whatsappReminder")}</p>
                  <T as="p" className="text-sm text-muted-foreground">Receive reminder on WhatsApp</T>
                </div>
              </div>
              <Switch
                checked={newReminder.whatsappEnabled}
                onCheckedChange={(checked) =>
                  setNewReminder((prev) => ({ ...prev, whatsappEnabled: checked }))
                }
              />
            </div>

            {newReminder.whatsappEnabled && (
              <div className="space-y-4">
                {/* Phone number input */}
                <div className="space-y-2">
                  <Label htmlFor="phone" className="text-base font-medium">
                    <T>WhatsApp Number</T>
                  </Label>
                  <Input
                    id="phone"
                    type="tel"
                    placeholder="+919876543210"
                    value={newReminder.phone}
                    onChange={(e) => setNewReminder((prev) => ({ ...prev, phone: e.target.value }))}
                    className="h-12"
                  />
                  <T as="p" className="text-xs text-muted-foreground">Include country code, e.g. +91XXXXXXXXXX</T>
                </div>

                {/* One-time WhatsApp activation */}
                {joinInfo?.configured && joinInfo.link ? (
                  <div className="rounded-xl border border-success/30 bg-success/5 p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="h-5 w-5 text-success shrink-0" />
                      <T as="p" className="font-medium text-foreground text-sm">One-time WhatsApp Activation</T>
                    </div>
                    <T as="p" className="text-xs text-muted-foreground leading-relaxed">Tap the button below. WhatsApp will open with a message already typed — just press Send. You only need to do this once, ever.</T>
                    <a
                      href={joinInfo.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center justify-center gap-2 rounded-lg bg-success px-4 py-2.5 text-sm font-semibold text-white hover:bg-success/90 transition-colors"
                    >
                      <MessageCircle className="h-4 w-4" />
                      <T>Open WhatsApp & Tap Send</T>
                      <ExternalLink className="h-3.5 w-3.5 opacity-70" />
                    </a>
                    <T as="p" className="text-xs text-center text-muted-foreground">After sending, come back and create your reminder.</T>
                  </div>
                ) : joinInfo && !joinInfo.configured ? (
                  <div className="rounded-xl border border-warning/30 bg-warning/5 p-4">
                    <p className="text-xs text-muted-foreground">
                      <strong>Setup needed:</strong> Add your Twilio sandbox word to <code className="bg-muted px-1 rounded">.env</code> as <code className="bg-muted px-1 rounded">TWILIO_SANDBOX_WORD</code>.
                      Find it at console.twilio.com → Messaging → Try it out → Send a WhatsApp message.
                    </p>
                  </div>
                ) : null}
              </div>
            )}

            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}

            <Button
              onClick={handleAddReminder}
              className="h-14 w-full text-lg font-semibold"
              disabled={!newReminder.title || !newReminder.date || !newReminder.time || isSaving}
            >
              {isSaving ? (
                <>
                  <Spinner className="mr-2 h-5 w-5" />
                  <T>Setting reminder...</T>
                </>
              ) : (
                <>
                  <Bell className="mr-2 h-5 w-5" />
                  <T>Create Reminder</T>
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Existing Reminders */}
        <Card className="border-none shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5 text-primary" />
              Your Reminders
            </CardTitle>
            <CardDescription><T>Manage your active payment reminders</T></CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex h-48 items-center justify-center">
                <Spinner className="h-8 w-8 text-primary" />
              </div>
            ) : reminders.length > 0 ? (
              <div className="space-y-4">
                {reminders.map((reminder) => (
                  <div
                    key={reminder.job_id}
                    className="flex items-center justify-between rounded-xl bg-muted p-4"
                  >
                    <div className="flex items-center gap-4">
                      <div className="rounded-full bg-primary/20 p-3">
                        <Smartphone className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium text-foreground">{reminder.message}</p>
                        <p className="text-sm text-muted-foreground">
                          {formatDateTime(reminder.remind_at)}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          <T>Status:</T> {reminder.status}
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDeleteReminder(reminder.job_id)}
                      className="text-muted-foreground hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex h-48 flex-col items-center justify-center text-center">
                <div className="mb-4 rounded-full bg-muted p-4">
                  <Bell className="h-8 w-8 text-muted-foreground" />
                </div>
                <T as="p" className="text-lg font-medium text-muted-foreground">No reminders yet</T>
                <T as="p" className="mt-1 text-sm text-muted-foreground">Create your first reminder to get started</T>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Info Card */}
      <Card className="border-none bg-primary/5 shadow-md">
        <CardContent className="flex items-start gap-4 p-6">
          <div className="rounded-full bg-primary/10 p-3">
            <MessageCircle className="h-6 w-6 text-primary" />
          </div>
          <div>
            <T as="h3" className="mb-1 font-semibold text-foreground">WhatsApp Integration</T>
            <T as="p" className="text-sm text-muted-foreground">Enable WhatsApp reminders to receive payment alerts directly on your phone. Never miss a due date again with timely notifications in your preferred language.</T>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
