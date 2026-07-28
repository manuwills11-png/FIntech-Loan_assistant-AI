# FinEdge Loan — Feature Audit

Exhaustive feature inventory extracted directly from the codebase (FastAPI backend + Next.js frontend). Every feature below is traced to specific file(s). "Fully implemented" means the code path works end-to-end with no missing pieces; "Partially implemented" means it runs but has a known gap, silent fallback, or degraded mode; "Planned/stubbed" means present in schema/UI but not backed by real logic.

---

## 1. API Routes / Endpoints

| Feature | What it does | File(s) | Status |
|---|---|---|---|
| `GET /` | Root health/info endpoint — service name, version, docs link | `backend/app/main.py` | Fully implemented |
| `GET /health` | Detailed health check: ML model presence, active LLM provider, WhatsApp/Google Cloud config state | `backend/app/main.py` | Partially implemented — only checks `GEMINI_API_KEY`/`OPENAI_API_KEY` for the `llm` field, never reports Groq even though Groq is tried first at runtime (misleading output) |
| `POST /predict-risk` | Runs the full loan risk assessment (CIBIL-style formula + ML blend + AI advice) for a single applicant | `backend/app/routes/predict.py` | Fully implemented |
| `POST /simulate` | "What-if" simulator — reruns risk scoring against a neutral baseline so users can see the effect of changing inputs | `backend/app/routes/simulate.py` | Fully implemented |
| `POST /roadmap` | Generates a month-by-month EMI repayment schedule, expense/income tips, and an LLM narrative summary | `backend/app/routes/roadmap.py` | Fully implemented |
| `POST /chat` | Text chat with the agentic AI assistant (tool-calling loop) | `backend/app/routes/chat.py` | Partially implemented — tool-calling and data extraction work, but the agent frequently hits its 5-iteration cap and returns a generic "Analysis complete" fallback instead of a real reply (verified reproducible, see Known Issues) |
| `POST /chat/voice` | Voice-in, voice-out chat: transcribes uploaded audio, runs the same agent loop, returns TTS audio | `backend/app/routes/chat.py` | Fully implemented (inherits the same chat-loop reliability caveat) |
| `POST /chat/negotiate` | Standalone loan-negotiation coaching endpoint — gives scripts/tactics for negotiating with a bank | `backend/app/routes/chat.py` | Fully implemented |
| `POST /translate/batch` | Batch-translates a list of English UI strings into a target language in one request | `backend/app/routes/translate.py` | Fully implemented |
| `GET /set-reminder` | Lists all currently scheduled reminders | `backend/app/routes/reminder.py` | Fully implemented, but in-memory only (see Known Issues) |
| `POST /set-reminder` | Schedules a one-time or recurring (cron) WhatsApp reminder | `backend/app/routes/reminder.py` | Fully implemented, but in-memory only |
| `DELETE /set-reminder/{job_id}` | Cancels a scheduled reminder | `backend/app/routes/reminder.py` | Fully implemented |
| `GET /send-whatsapp/join-link` | Returns a `wa.me` deep link to join the Twilio WhatsApp sandbox | `backend/app/routes/whatsapp.py` | Fully implemented |
| `POST /send-whatsapp` | Sends an immediate WhatsApp message via Twilio | `backend/app/routes/whatsapp.py` | Fully implemented (falls back to a "simulated" no-op response if Twilio isn't configured) |
| `POST /upload-document` | Uploads a PDF/image (bank statement, salary slip, PAN, Aadhaar, CIBIL report) for OCR + structured extraction | `backend/app/routes/document.py` | Partially implemented — primary path (Gemini Vision) is broken (see Known Issues), silently falls back to Tesseract + regex |
| `GET /demo-user` | Lists all pre-loaded demo personas | `backend/app/routes/demo.py` | Fully implemented |
| `GET /demo-user/{id}` | Loads a demo persona (farmer / student / salaried / high_risk) and runs a live risk assessment on it | `backend/app/routes/demo.py` | Fully implemented |
| `GET /bank-rates` | Compares loan interest rates across 6–9 hardcoded Indian banks by CIBIL score, purpose, and amount; ranks and recommends | `backend/app/routes/bank_rates.py` | Fully implemented (rate table is static/hardcoded data, not live-sourced) |
| `POST /bank-rates/contact` | Drafts a formal loan inquiry letter and generates a template-based simulated bank response | `backend/app/routes/bank_rates.py` | Fully implemented (bank reply is templated, not AI-generated, despite the docstring) |

---

## 2. ML / AI Components

| Feature | What it does | File(s) | Status |
|---|---|---|---|
| Gradient Boosting risk classifier | `sklearn` `GradientBoostingClassifier` (300 estimators) trained on 10,000 synthetic Indian loan applicant records to predict probability of "high risk"; blended 40% into the final risk score alongside the 60%-weighted rule-based formula | `backend/ml/train_model.py`, loaded in `backend/app/services/risk_service.py` | Fully implemented. Model file (`loan_risk_model.pkl`) is trained from **synthetic, not real, data** — no real-world loan outcome dataset is used |
| Synthetic dataset generator | Generates a realistic Indian income/employment/debt distribution (55% salaried-low, 25% salaried-mid, 10% self-employed/farmer, 10% student) and derives a rule-consistent risk label for supervised training | `backend/ml/train_model.py` | Fully implemented |
| CIBIL-style 7-factor rule-based scorer | Deterministic weighted formula (Credit Score 30%, EMI Burden 25%, Debt Load 15%, Savings Buffer 15%, Loan Size 10%, Age 3%, Employment Stability 2%) producing a 0–100 risk score with per-factor breakdown and traffic-light status | `backend/app/services/risk_service.py` | Fully implemented — this is the primary/dominant scoring path, ML is a minority blend |
| Co-applicant risk blending | If a co-applicant is supplied, combines incomes for EMI/savings ratios and blends CIBIL scores 70/30 (primary/co-applicant) | `backend/app/services/risk_service.py` | Fully implemented |
| Groq LLM (Llama 3) — agentic chat | Primary conversational engine with function/tool calling (assess risk, generate roadmap, fetch schemes, contact bank, save data, navigate, save phone) | `backend/app/services/ai_service.py` | Partially implemented — works but unreliable (see Known Issues) |
| Groq/Gemini/OpenAI cascading fallback | `generate_response()` tries Groq → Gemini → OpenAI → rule-based text fallback, in order, for simple (non-agentic) text generation (used for risk advice, roadmap narrative) | `backend/app/services/ai_service.py` | Fully implemented |
| Groq Whisper speech-to-text | Transcribes uploaded voice audio (`whisper-large-v3`), with a follow-up Groq LLM call to fix domain-specific mis-transcriptions (loan/EMI/bank terms) in Indian languages | `backend/app/services/speech_service.py` | Fully implemented |
| Gemini multimodal STT fallback | If Groq Whisper is unavailable, falls back to Gemini 2.0/2.5 Flash for audio transcription | `backend/app/services/speech_service.py` | Fully implemented |
| Gemini Vision document OCR | Extracts structured fields (name, DOB, PAN, salary, CIBIL score, etc.) from an uploaded image using Gemini Vision with a strict JSON-only prompt | `backend/app/services/ocr_service.py` | **Broken** — hardcodes deprecated model `gemini-1.5-flash` (line 589), returns 404 at call time, always falls through to the Tesseract fallback |
| Tesseract OCR + regex parser fallback | Local OCR (`pytesseract`, eng+hin) with hand-written regex parsers for 5 document types (PAN, Aadhaar, CIBIL report, salary slip, bank statement) | `backend/app/services/ocr_service.py` | Fully implemented — this is effectively the **primary** working OCR path today, not the fallback it's designed to be |
| gTTS text-to-speech | Converts assistant replies to speech (MP3, base64) in 8 supported languages | `backend/app/services/tts_service.py` | Fully implemented |
| AI-generated personalized loan advice | Feeds the computed risk score + factor breakdown to the LLM cascade for 2–3 sentence plain-language advice targeted at rural/low-literacy users | `backend/app/services/risk_service.py::_get_ai_advice` | Fully implemented |
| AI-generated roadmap narrative | LLM-generated plain-language summary of the repayment plan | `backend/app/services/ai_service.py::generate_roadmap_narrative`, called from `backend/app/routes/roadmap.py` | Fully implemented |

---

## 3. Multilingual / NLP Support

| Feature | What it does | File(s) | Status |
|---|---|---|---|
| 13 backend-supported languages | English, Hindi, Tamil, Telugu, Kannada, Marathi, Bengali, Malayalam, Gujarati, Urdu, Punjabi, Assamese, Odia | `backend/app/models/schemas.py::SupportedLanguage` | Fully implemented |
| 15 frontend UI languages | Same 13 plus Kashmiri, Manipuri, Konkani (aliased to Hindi on the backend since they're low-resource for Google Translate) | `frontend/lib/language-context.tsx`, `frontend/lib/api.ts::LANG_MAP` | Fully implemented |
| Static UI dictionary | ~95 hand-translated UI strings (buttons, labels, section headers) per language, hardcoded in a translation table | `frontend/lib/language-context.tsx` | Fully implemented for the ~95 keys covered; any UI text not in this dictionary is not covered by it (falls to the auto-translate layer below) |
| Runtime auto-translation layer (`<T>` / `useTrans`) | Wraps arbitrary English JSX text; batches all on-page strings into one `/api/translate/batch` call per language switch, caches per-language results in `localStorage`, retries on failure with backoff, falls back to English if translation fails | `frontend/lib/auto-translate.tsx` | Fully implemented — well-engineered (debounced batching, cache, retry) |
| Backend translation service (Google Translate via `deep-translator`) | Free, no-API-key translation for English↔target language, both single-string and batched | `backend/app/services/translate_service.py` | Fully implemented |
| Language auto-detection | Detects input text language using `langdetect` | `backend/app/services/translate_service.py::detect_language` | Implemented but **not wired into any route** — no endpoint calls it (dead code / unused capability) |
| Risk explanation/recommendation translation | Translates the AI risk explanation, recommendation, and key factors into the requested language for `/predict-risk` | `backend/app/routes/predict.py` | Fully implemented — verified producing correct Devanagari Hindi output |
| Roadmap summary translation | Translates the LLM-generated roadmap narrative | `backend/app/routes/roadmap.py` | Fully implemented |
| Multilingual chat replies | Chat agent's system prompt instructs the LLM to reply directly in the requested language (no post-hoc translation) | `backend/app/routes/chat.py::_build_system_prompt` | Partially implemented — functional but inconsistent quality: verified producing **romanized Hindi ("Hinglish")** rather than Devanagari script, unlike the translation-service-backed features above |
| Multilingual voice transcription hints | Per-language prompt hints passed to Whisper to improve recognition of financial terms | `backend/app/services/speech_service.py::_get_lang_prompt` | Fully implemented (10 languages) |
| Multilingual TTS | Speech output in 8 of the 13 backend languages (Tamil, Telugu, Kannada, Marathi, Bengali, Malayalam, Hindi, English) — Gujarati, Urdu, Punjabi, Assamese, Odia fall back to English audio | `backend/app/services/tts_service.py::_LANG_MAP` | Partially implemented — 5 of 13 supported languages have no TTS voice mapping |

---

## 4. Loan Risk Assessment Logic

| Feature | What it does | File(s) | Status |
|---|---|---|---|
| 7-factor weighted risk formula | See ML section above — this is the core scoring engine | `backend/app/services/risk_service.py::_compute_cibil_factors` | Fully implemented |
| CIBIL-score-to-risk mapping | Piecewise function mapping 300–900 CIBIL score to 0–100 risk (750+ = low risk band, 650–749 = fair, <650 = poor) | `backend/app/services/risk_service.py::_cibil_to_risk` | Fully implemented |
| Debt-to-income & EMI-to-income ratio computation | Explicit ratios returned in the API output, also used as scoring inputs | `backend/app/services/risk_service.py::predict_risk` | Fully implemented |
| Explainability / factor breakdown | Per-factor score, weight, and good/fair/poor status returned so the UI can render a "score breakdown" chart | `backend/app/models/schemas.py::FactorScore`, `risk_service.py` | Fully implemented |
| Key-factor plain-language extraction | Rule-based sentence generator that turns numeric red flags (high EMI burden, negative savings, low CIBIL, self-employment volatility, etc.) into human-readable bullet points | `backend/app/services/risk_service.py::_identify_key_factors` | Fully implemented |
| Risk category banding | 0–39 = Low, 40–69 = Medium, 70–100 = High | `backend/app/utils/helpers.py::score_to_category` | Fully implemented |
| Co-applicant eligibility boost | Combined income/CIBIL scoring for joint applications | `backend/app/services/risk_service.py` (see ML section) | Fully implemented |
| Credit simulator ("what-if") | Recomputes risk against a neutral (repayment_history_score=50) baseline and reports the delta plus contextual improvement tips | `backend/app/routes/simulate.py` | Fully implemented |
| Financial roadmap / EMI amortization | Full month-by-month amortization schedule using risk-category-derived interest rate (10%/14%/18% p.a. for Low/Medium/High) | `backend/app/routes/roadmap.py` | Fully implemented |

---

## 5. User-Facing Flows

| Feature | What it does | File(s) | Status |
|---|---|---|---|
| Loan risk assessment form (multi-step, "Step 1 of 4") | Full loan application form: income/expenses/EMI/existing loans, loan amount/tenure/purpose, CIBIL/age/employment, optional co-applicant section | `frontend/app/loan-risk/page.tsx` | Fully implemented — verified working end-to-end in browser |
| Conversational AI assistant (text) | Chat UI with quick-reply chips, tool-driven auto-navigation/pre-fill of other pages, TTS playback toggle | `frontend/app/chat/page.tsx` | Partially implemented — UI is solid; underlying agent reliability issue carries through (see Known Issues) |
| Voice input (chat) | Two-tier: browser-native `SpeechRecognition` Web API first (if available), falls back to `MediaRecorder` + server-side Whisper transcription on error/unavailability | `frontend/app/chat/page.tsx` | Fully implemented |
| Voice output (TTS playback) | Plays back the assistant's spoken reply via base64 MP3 | `frontend/app/chat/page.tsx`, `backend/app/services/tts_service.py` | Fully implemented |
| Document upload & extraction | Drag-and-drop / file-picker upload of PAN, Aadhaar, salary slip, bank statement, or CIBIL report; shows structured extracted fields; supports multiple simultaneous uploads | `frontend/app/documents/page.tsx` | Fully implemented at the UI level; backend extraction quality degraded (Gemini Vision path broken) |
| Bank rate comparison & inquiry | Compares 6–9 banks by purpose, ranks by rate/eligibility, shows a "best pick", and a "Contact Bank" flow that drafts an inquiry + shows a simulated bank reply with next steps | `frontend/app/bank-rates/page.tsx`, `backend/app/routes/bank_rates.py` | Fully implemented |
| Gold loan calculator | Computes gold value (weight × purity-adjusted rate/gram) and max eligible loan (75% LTV per RBI mandate), then compares gold-loan-specific bank offers (no CIBIL required) | `frontend/app/gold-loan/page.tsx` | Fully implemented |
| Government loan scheme directory | Static catalog of ~10+ government-backed schemes (KCC, PMFBY, PMAY-CLSS, MUDRA, Vidya Lakshmi, etc.) with eligibility, benefits, and documents required per category (agriculture/home/business/education) | `frontend/app/govt-schemes/page.tsx` | Fully implemented (static content, not fetched from a live government API) |
| Loan negotiation coach | Chat-style interface with 6 canned "tactic" prompts (counter-offer script, CIBIL leverage, competitor comparison, fee waiver, prepayment terms, risk-based pricing) that get answered by the AI coach | `frontend/app/negotiate/page.tsx`, `backend/app/routes/chat.py::negotiate` | Fully implemented |
| Financial roadmap page | Displays the generated amortization schedule, tips, and narrative summary | `frontend/app/roadmap/page.tsx` | Fully implemented |
| Credit simulator page | Interactive sliders/inputs for instant what-if risk recalculation | `frontend/app/simulator/page.tsx` | Fully implemented |
| Reminders page | Set/list/cancel WhatsApp EMI payment reminders (one-time or recurring via cron expression) | `frontend/app/reminders/page.tsx`, `backend/app/routes/reminder.py` | Fully implemented, but reminders are lost on backend restart (in-memory scheduler, no persistence) |
| Demo mode | 4 pre-configured personas (rural farmer, student, salaried professional, high-risk self-employed) for quick demonstration without manual data entry | `backend/app/routes/demo.py` | Fully implemented (backend only — no dedicated frontend page found wired to it) |
| Dashboard / loan journey tracker | Landing page showing a 4-step guided journey (Check Eligibility → Compare Banks → Prepare Documents → Plan Repayment) with progress gating | `frontend/app/page.tsx` | Fully implemented |
| Financial health score gauge | Circular SVG gauge visualizing a 0–100 score with color-coded bands | `frontend/components/health-score.tsx` | Fully implemented (presentational component) |
| Risk gauge visualization | Dedicated risk-score gauge component used on the loan-risk results panel | `frontend/components/risk-gauge.tsx` | Fully implemented |
| Client-side app state persistence | Form data, chat history, gold-loan inputs, simulator values, and phone number persisted to `localStorage` and rehydrated on load | `frontend/lib/app-store.tsx` | Fully implemented |
| Chat-driven cross-page automation | Chat tool calls can save data, pre-fill and auto-submit the loan-risk form, navigate to roadmap/bank-rates/gold-loan, and save a WhatsApp number — all without the user manually re-entering data | `backend/app/routes/chat.py` (`TOOLS`, action builders), `frontend/lib/api.ts::ChatAction` | Fully implemented mechanically, but gated by the chat agent reliability issue |

---

## 6. Authentication, Data Storage, Database Schema

| Feature | What it does | File(s) | Status |
|---|---|---|---|
| User authentication | **None found.** No login, signup, session, JWT, or password handling anywhere in the codebase. `PyJWT`/`python-jose` appear in `requirements.txt` but are not imported or used by any route/service. | — | Not implemented |
| Database / persistent storage | **None found.** No SQL/NoSQL database, ORM, or migration files exist anywhere in the repo. | — | Not implemented |
| Reminder storage | Reminders live only in the `APScheduler` `BackgroundScheduler`'s in-memory job store; nothing is written to disk. Restarting the backend process silently drops all scheduled reminders. | `backend/app/scheduler/reminder_scheduler.py` | Partially implemented (functional but non-persistent) |
| Client-side state storage | All user financial data (loan form inputs, chat history, gold loan calc, simulator values, phone number) is stored exclusively in browser `localStorage`, per-device, with no server-side record | `frontend/lib/app-store.tsx` | Fully implemented as a client-only substitute for a backend data store |
| ML training data | Model is trained from a synthetic dataset generated at training time, not from any stored/collected user data | `backend/ml/train_model.py` | N/A — no real user data is retained or used for training |

**Inference flag:** the complete absence of auth/DB is directly evidenced by an exhaustive grep across the repo for `auth|login|jwt|session|database|sqlite|postgres|mongo|prisma` — the only matches are unrelated (Twilio "session", UI component library internals, `.env.example` placeholders).

---

## 7. Admin / Dashboard Features

**None found.** There is no admin panel, no role-based access, no user-management screen, and no analytics/metrics dashboard for operators. The only "dashboard" is the end-user-facing landing page (`frontend/app/page.tsx`) showing that single user's own loan journey — not an administrative view of the system or its users.

---

## 8. Third-Party Integrations

| Integration | Role | File(s) | Status |
|---|---|---|---|
| Groq API (Llama 3 models) | Primary LLM for chat agent, negotiation coach, risk advice, roadmap narrative, Whisper transcription, transcription correction | `backend/app/services/ai_service.py`, `speech_service.py`, `chat.py` | Fully implemented, verified live (200 OK responses observed) |
| Google Gemini API | Fallback LLM (chat, advice, roadmap), fallback STT, primary (but broken) document OCR | `backend/app/services/ai_service.py`, `speech_service.py`, `ocr_service.py` | Fully implemented for text; OCR path broken (deprecated model name) |
| OpenAI API | Third-tier fallback LLM if Groq and Gemini both fail/unset | `backend/app/services/ai_service.py` | Fully implemented (untested live — depends on all other providers failing) |
| Twilio WhatsApp API | Sends WhatsApp messages/reminders; requires sandbox "join" step in development | `backend/app/services/whatsapp_service.py` | Fully implemented, verified configured with real credentials |
| Google Translate (via `deep-translator`, free/no-key) | Backend translation for UI strings, risk explanations, roadmap summaries | `backend/app/services/translate_service.py` | Fully implemented |
| gTTS (Google Translate TTS endpoint, free/no-key) | Text-to-speech audio generation | `backend/app/services/tts_service.py` | Fully implemented |
| Tesseract OCR (`pytesseract`) | Local, offline OCR fallback for document extraction | `backend/app/services/ocr_service.py` | Fully implemented, verified working (currently the de-facto primary OCR path) |
| `scikit-learn` | Trains and runs the Gradient Boosting risk classifier | `backend/ml/train_model.py`, `risk_service.py` | Fully implemented |
| `APScheduler` | Background job scheduling for reminders (one-time `DateTrigger` and recurring `CronTrigger`) | `backend/app/scheduler/reminder_scheduler.py` | Fully implemented |
| `PyPDF2` / `pdf2image` | PDF text extraction and PDF→image conversion for OCR of PDF documents | `backend/app/services/ocr_service.py` | Fully implemented |
| Render (hosting) | Backend deployment target — builds venv, trains model at build time, runs `start.py` | `render.yaml` | Configured for deployment |
| Vercel (hosting) | Frontend deployment target — Next.js build | `vercel.json` | Configured for deployment |
| Recharts | Chart rendering library (bundled, used for score visualizations) | `frontend/package.json` | Fully implemented |
| Radix UI primitives + shadcn/ui component set | Full accessible component library (dialogs, dropdowns, tabs, accordion, tooltip, etc.) underlying the entire UI | `frontend/package.json`, `frontend/components/ui/*` | Fully implemented |

---

## 9. Accessibility / Rural-User / Low-Literacy Design Features

| Feature | What it does | File(s) | Status |
|---|---|---|---|
| Voice-first interaction | Users can speak instead of type, in their own language, both in and out (STT + TTS) | `frontend/app/chat/page.tsx`, `speech_service.py`, `tts_service.py` | Fully implemented |
| 15-language coverage including regional/low-resource languages | Kashmiri, Manipuri, Konkani included alongside the 12 major Indian languages + English | `frontend/lib/language-context.tsx` | Fully implemented |
| Simplified-language system prompt | Explicit LLM instruction to "use very simple words, avoid jargon, be encouraging not scary," targeted at rural/first-time borrowers | `backend/app/utils/helpers.py::build_system_prompt`, `risk_service.py::_get_ai_advice` prompt | Fully implemented |
| Demo personas reflecting rural/informal-income users | Farmer with irregular seasonal income (Hindi-language demo), self-employed vendor persona | `backend/app/routes/demo.py` | Fully implemented |
| Employment-type-aware guidance | Distinguishes salaried / self-employed / farmer / student in scoring, income-improvement tips (SHG credit, KCC, seasonal labor, gig platforms), and government scheme matching | `risk_service.py`, `roadmap.py::_income_tips`, `govt-schemes/page.tsx` | Fully implemented |
| WhatsApp-based reminders | Meets users on a channel that doesn't require a smartphone banking app or literacy with a web dashboard | `backend/app/services/whatsapp_service.py` | Fully implemented |
| Document OCR instead of manual form-filling | Lets users photograph existing documents rather than transcribe numbers themselves | `frontend/app/documents/page.tsx`, `ocr_service.py` | Partially implemented (degraded OCR quality, see Known Issues) |
| Mobile-first responsive navigation | Bottom tab bar on mobile, full sidebar on desktop | `frontend/components/app-wrapper.tsx`, `navigation.tsx` | Fully implemented |
| No-CIBIL-required path (gold loans) | Explicitly designed for users without formal credit history — gold serves as sole collateral | `bank_rates.py` (`gold` category, cutoff 0), `gold-loan/page.tsx` | Fully implemented |

---

## 10. Security / Compliance Features

| Feature | What it does | File(s) | Status |
|---|---|---|---|
| CORS middleware | Configurable allowed-origins list (defaults to `*` if unset) | `backend/app/main.py` | Fully implemented, but defaults are permissive (`allow_origins=["*"]`) unless `ALLOWED_ORIGINS` is explicitly set |
| Global exception handler | Catches unhandled exceptions and returns a generic 500 without leaking stack traces to the client | `backend/app/main.py` | Fully implemented |
| File upload validation | Content-type allowlist and 10 MB size cap on document uploads | `backend/app/routes/document.py` | Fully implemented |
| Input validation | Pydantic field constraints (ranges, required fields) on all request schemas | `backend/app/models/schemas.py` | Fully implemented |
| Secrets via environment variables | API keys (Groq/Gemini/OpenAI/Twilio) loaded from `.env`, never hardcoded | `backend/app/main.py::load_dotenv()`, `.env.example` | Fully implemented |
| Data encryption at rest / in transit (beyond standard HTTPS at hosting layer) | **None found** — no explicit encryption of stored data (there is no persistent store to encrypt in the first place) | — | Not implemented |
| Data privacy / consent handling | **None found** — no privacy policy, consent capture, or data-deletion mechanism in code | — | Not implemented |
| Rate limiting | **None found** on any endpoint | — | Not implemented |
| Sensitive-document redaction | OCR prompt explicitly instructs the model to extract real data even from documents watermarked "SPECIMEN/MOCK/DEMO" — no redaction or masking of PAN/Aadhaar numbers in responses or logs | `backend/app/services/ocr_service.py::_GEMINI_PROMPT` | Not implemented (arguably a minor privacy gap worth flagging) |

---

## Known Issues (discovered via live testing, relevant to implementation-status calls above)

1. **Chat agent reliability**: `backend/app/routes/chat.py::_run_agent` caps at `MAX_ITER = 5` tool-calling iterations. Reproduced twice (English and Hindi, with both incomplete and complete input data) hitting this cap and returning the generic fallback "Analysis complete. Please check the results on screen." instead of a real conversational reply. Underlying actions (save data, navigate, pre-fill, generate roadmap) still fire correctly even when this happens.
2. **OCR primary path broken**: `backend/app/services/ocr_service.py:589` hardcodes `gemini-1.5-flash`, a deprecated model name that returns HTTP 404. Every document upload silently falls through to the Tesseract regex-based fallback, which works but is materially weaker (confirmed via test: `"Gross Salary"` OCR'd as `"Gress Salary. 550000"`).
3. **`/health` endpoint misreports active LLM**: only checks `GEMINI_API_KEY`/`OPENAI_API_KEY`, never `GROQ_API_KEY`, so it reports `"llm":"gemini"` even when Groq is the model actually serving every chat request.
4. **Reminder persistence**: reminders exist only in the `APScheduler` in-process job store; any backend restart or redeploy silently drops all scheduled reminders with no error surfaced to the user.

---

# College Project Presentation Brief

*Sections below are synthesized strictly from what the code evidences. Anything not directly observable in the codebase is explicitly flagged as inferred.*

## Project Idea / Problem Statement

FinEdge is an AI-powered, multilingual financial assistant that helps Indian loan applicants understand and improve their loan eligibility before they approach a bank. The system takes an applicant's financial details (income, expenses, existing debt, requested loan) and produces a transparent, explainable risk score (CIBIL-inspired 7-factor formula blended with a machine-learning classifier), plain-language advice, a personalized repayment roadmap, bank rate comparisons, and document-based data extraction — all through a conversational, voice-capable interface in the user's own language. *(Problem framing — "why this matters" — is inferred from the feature set and the explicit rural/low-literacy design choices in the code; no mission statement or README exists to state the problem directly.)*

## Target Audience

Directly evidenced in code:
- Indian retail loan applicants generally (salaried, self-employed, farmers, students) — covered explicitly by employment-type logic and the four demo personas.
- Rural and semi-urban, non-technical, low-literacy users — explicitly named in the LLM system prompt (`build_system_prompt`: "helping rural and semi-urban users in India... use very simple words").
- Non-English-first speakers across 15 Indian languages including low-resource ones (Kashmiri, Manipuri, Konkani).
- Users without a formal credit history — the gold-loan path requires no CIBIL score.

*(Inferred: "first-time borrowers" and "financially underserved populations" as a framing — reasonable given the feature set, but not a label found literally in the code.)*

## Benefits and Impact on Society

Directly evidenced by functionality:
- Reduces loan-rejection risk by surfacing eligibility issues (via risk score + explainable factor breakdown) *before* a formal bank application, where rejections can hurt credit history.
- Increases financial literacy access via voice interaction and native-language explanations for users who may not read English or navigate complex banking forms.
- Surfaces subsidized government schemes (KCC, PMAY-CLSS, PMFBY, Vidya Lakshmi, MUDRA) that users might not otherwise discover.
- Provides a no-CIBIL-required path (gold loans) for the credit-invisible population.
- Automates document data entry (OCR) to reduce the burden of manual form-filling.
- Bank rate comparison and a negotiation coach aim to reduce information asymmetry between borrowers and lenders.

*(Inferred: quantified societal impact, e.g. "financial inclusion for X million underbanked Indians" — no such claim or data exists in the code; this is a reasonable narrative extrapolation, not an evidenced fact.)*

## Threats / Challenges Related to the Project

Directly evidenced by the audit above:
- **No authentication or database** — the system cannot currently support real user accounts, persistent loan history, or multi-device continuity; all state is per-browser `localStorage` or lost on backend restart.
- **No data privacy/consent mechanism** — despite processing sensitive PII (PAN, Aadhaar, salary, CIBIL data via OCR), there is no consent capture, redaction, or data-retention policy in code.
- **ML model trained on synthetic data only** — the Gradient Boosting classifier has never seen real loan-outcome data; its 40% weight in the blended score is a plausibility model, not a validated one.
- **Third-party LLM/API dependency risk** — core features (chat, advice, OCR, translation, TTS) depend on external APIs (Groq, Gemini, OpenAI, Google Translate, Twilio); several have observed reliability gaps (chat agent iteration cap, deprecated OCR model) even in normal operation.
- **No rate limiting** — all endpoints, including paid third-party API calls (Twilio SMS, LLM calls), are exposed with no throttling, creating cost/abuse exposure.
- **Static bank-rate data** — interest rates are hardcoded (dated "April 2026" in comments) rather than sourced live, so accuracy decays over time without manual updates.
- **Hardcoded model names break silently** — as seen with the OCR Gemini model, provider-side model deprecations fail silently into a degraded fallback rather than surfacing an error.

## Requirements for the Project

**Technical stack (directly evidenced):**
- Backend: Python 3, FastAPI, Uvicorn, Pydantic v2, APScheduler
- ML: scikit-learn (GradientBoostingClassifier), joblib, NumPy, pandas
- AI/NLP: Groq SDK (Llama 3 + Whisper), `google-generativeai` (Gemini), `openai` SDK, `deep-translator`, `langdetect`, `gTTS`
- OCR: `pytesseract` (+ system Tesseract binary), `Pillow`, `PyPDF2`, `pdf2image`
- Messaging: Twilio SDK (WhatsApp)
- Frontend: Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4, Radix UI / shadcn-ui component set, Recharts

**Data requirements:**
- No real user data required to run the system as built (synthetic training data, hardcoded bank rates and scheme data).
- Requires API keys for Groq, Gemini, OpenAI (optional fallback), Twilio (for real WhatsApp sending), and optionally `GOOGLE_APPLICATION_CREDENTIALS` (referenced in `.env.example` though no Google Cloud SDK usage was found in the current OCR implementation — likely a leftover from an earlier design).

**Infrastructure:**
- Backend deploys to Render (`render.yaml`): trains the ML model at build time, runs via `python start.py`.
- Frontend deploys to Vercel (`vercel.json`), proxying `/api/*` to the backend via Next.js rewrites.
- No database or managed persistence layer required by the current architecture.

## Other Relevant Details

- **Multi-provider LLM cascade with graceful degradation**: chat/advice features try Groq → Gemini → OpenAI → rule-based text, so the app keeps functioning (in a reduced form) even if one or more AI providers are down or unconfigured — a notable resilience differentiator.
- **Dual OCR strategy** (cloud Gemini Vision + local Tesseract fallback) is architecturally sound, even though the cloud path currently has a bug.
- **Explainable AI risk scoring**: the system does not just output a black-box score — it returns a full per-factor weighted breakdown with plain-language justifications, which is a meaningful transparency differentiator versus opaque credit-scoring systems.
- **Agentic tool-calling chat**: the assistant can autonomously execute actions (run a risk assessment, pre-fill and submit forms, navigate the user, generate a roadmap) rather than just answering questions — a more advanced interaction model than a typical FAQ chatbot, albeit with the reliability caveat noted above.
- **Runtime auto-translation engine** for arbitrary UI text (batching + caching + retry) is a non-trivial piece of frontend engineering beyond a simple static i18n dictionary.
- **No backend persistence** is a deliberate-looking architectural simplification (likely appropriate for a hackathon/demo-stage project) rather than an oversight — it keeps the system stateless and easy to deploy, at the cost of no real user accounts or history.
