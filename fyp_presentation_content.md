# Vital Guardian — FYP Final Evaluation Slide Content

---

## SLIDE 1 — Title Slide

**Title:** Vital Guardian
**Subtitle:** AI-Powered ICU Patient Monitoring System
**Team:** [Your Names]
**Supervisor:** [Supervisor Name]
**Date:** [Presentation Date]

---

## SECTION 1 — FYP Overview (1–2 minutes)

### SLIDE 2 — What is Vital Guardian?

**Headline:** An AI system that watches over ICU patients so nurses don't have to watch every screen

**3 Core Goals:**
- 🎯 **Detect** — Automatically detect patient falls and seizures in real-time using computer vision
- 🔔 **Alert** — Instantly notify clinical staff with AI-verified, actionable alerts
- 🔒 **Protect** — Preserve patient privacy while maintaining safety

**One-liner:** *Vital Guardian is a real-time AI monitoring system that acts as a tireless, intelligent observer in the ICU — detecting emergencies the moment they happen.*

---

### SLIDE 3 — The Problem We Solve

**Headline:** ICU nurses cannot watch every patient, every second

**Left column — The Reality:**
- A single nurse can manage 2–4 ICU patients simultaneously
- Falls and seizures can occur in under 3 seconds
- Delayed response significantly worsens patient outcomes
- Manual camera monitoring is exhausting and unsustainable

**Right column — The Cost:**
- Falls are the #1 cause of preventable hospital injuries
- Seizures undetected for >5 minutes cause lasting neurological damage
- Traditional alert systems have a 60–70% false positive rate — nurses stop trusting them

**Transition line:** *We built Vital Guardian to solve this with AI that is accurate, fast, and trustworthy.*

---

### SLIDE 4 — System Architecture Overview

**Headline:** A full-stack clinical AI pipeline

**Visual flow (left → right):**
```
ICU Camera Feed
      ↓
MoViNet (Action Recognition)
+ YOLO (Person Detection)
      ↓
Alert Consolidator
      ↓
Gemini Cognitive Core (Verification + Enrichment)
      ↓
Clinical Dashboard → Nurse Alert Dashboard
```

**Key numbers (shown as stat tiles):**
- **98%** detection confidence on confirmed alerts
- **< 3s** time from event to initial alert
- **2 models** running in parallel — fall + seizure
- **3-tier** verification before a nurse is notified

---

## SECTION 2 — Faculty Feedback (1–2 minutes)

### SLIDE 5 — Feedback Received

**Headline:** Faculty Evaluation Feedback

**The Feedback:**
> *"Patient privacy must be protected. Showing live video feeds to all staff creates HIPAA/privacy compliance risks."*

**Why this matters:**
- ICU patients are in vulnerable states — continuous video exposure is a dignity concern
- Live feeds shared broadly increase risk of unauthorized viewing
- Medical AI systems must meet clinical-grade privacy standards to be deployable in real hospitals

**Our response:** *We took this feedback seriously and built a dedicated privacy-first workflow from the ground up.*

---

### SLIDE 6 — Our Answer: Decouple Monitoring from Alerts

**Headline:** Privacy by Design — Separating what the AI sees from what the nurse sees

**The Core Insight:**
> Nurses don't need to see video to respond to an alert. They need to know **who**, **what**, **where**, and **what to do**.

**Old flow (privacy risk):**
```
Camera Feed → Shown directly to all staff
```

**New flow (privacy-safe):**
```
Camera Feed → AI only → Verified Alert → Nurse sees:
             Patient ID · Ward · Incident Type · Clinical Guidance
             (No video. No identifiable footage. Ever.)
```

*This is not just a feature — it is a fundamental architectural decision.*

---

## SECTION 3 — Work Done Since Last Evaluation (2–4 minutes)

### SLIDE 7 — Overview of Deliverables

**Headline:** Four major deliverables since last evaluation

| # | What | Why |
|---|---|---|
| 1 | 🚨 Nurse Alert Dashboard | Privacy-preserving alert management |
| 2 | ⚡ Local GPU Inference | Real-time performance, no cloud dependency |
| 3 | 🎨 UI Enhancements | Professional-grade clinical interface |
| 4 | 📋 Patient History & Logs | Auditability and clinical record-keeping |

---

### SLIDE 8 — Deliverable 1: Nurse Alert Dashboard

**Headline:** 🚨 The Privacy-First Answer to Faculty Feedback

**What it is:**
A dedicated, separate web page exclusively for nursing staff that shows AI-confirmed alerts — with zero video exposure.

**What a nurse sees per alert:**
- Anonymised Patient ID (e.g. PT-00003)
- ICU Ward
- Incident Type: FALL / SEIZURE / AUDIO DISTRESS
- Severity Level: Critical / High / Moderate
- AI-generated clinical narrative
- Recommended actions (e.g. "Assess airway", "Check vitals")
- Time of alert
- Acknowledge button

**What a nurse does NOT see:**
- ❌ No live video
- ❌ No patient name
- ❌ No photograph
- ❌ No room-level visual surveillance

**Technical highlights:**
- Real-time updates — page updates instantly when a new confirmed alert arrives
- Filter by: All / Falls / Seizures / Audio Alerts / Unacknowledged
- Nurse can acknowledge alerts with their name recorded
- Pulsing red badge on the Patient Hub notifies nurses a new alert is waiting
- Audio chime on new alert arrival

---

### SLIDE 9 — Deliverable 2: Local GPU Inference

**Headline:** ⚡ From Cloud-Dependent to Fully Local — RTX 4050 Powered

**The shift:**
Previously the system depended on Kaggle cloud inference for MoViNet — introducing latency, internet dependency, and failure risk during demos.

**What we built:**
- Full MoViNet-A2 inference running locally on **NVIDIA RTX 4050 GPU**
- Asynchronous "fire-and-forget" inference — GPU works in background, video stream never stutters
- TensorFlow 2.x + cuDNN configured inside a dedicated virtual environment
- All CUDA libraries pre-loaded at startup

**Why it matters:**
| | Cloud (Before) | Local GPU (Now) |
|---|---|---|
| Latency | 8–25s per inference | < 1s per inference |
| Internet required | Yes | No |
| Demo reliability | Fragile | Fully offline |
| Data privacy | Frames sent to cloud | Frames never leave device |

---

### SLIDE 10 — Deliverable 3: UI Enhancements

**Headline:** 🎨 A Clinical Interface Built to Professional Standards

**Multi-Patient Hub:**
- Real-time grid showing all monitored patients simultaneously
- Per-patient status badges (Normal / Alert / Critical)
- Live vital sign indicators
- One-click patient deep-dive

**Active Monitoring Dashboard:**
- Live video stream with AI overlay
- Real-time fall risk + seizure risk gauges
- Gemini Cognitive Core panel showing verification status
- Alert log with confidence scores and timestamps
- Alert banner with audio alarm on confirmed events

**Respiratory Audio Monitoring:**
- YAMNet model detects: Cough, Breathing distress, Wheezing, Panting
- Accumulator timeline shown on nurse dashboard (e.g. "BREATHING → PANT → COUGH CONFIRMED")
- Dedicated audio alert cards with cyan visual distinction

**Desktop Application Launcher:**
- Double-click to launch — opens like a native desktop app
- GTK3 splash screen with animated progress bar
- Shows each startup stage (GPU init → model load → server ready)
- Auto-opens browser when server is live

---

### SLIDE 11 — Deliverable 4: Patient History & Clinical Logs

**Headline:** 📋 Full Auditability — Every Alert, Permanently Recorded

**What is stored per incident:**
- Patient ID and ward
- Incident type and confidence score
- Gemini-verified narrative (what happened, clinically described)
- Severity classification
- Recommended actions taken
- Timestamp + acknowledgement record (who acknowledged, when)

**History page features:**
- Filterable by incident type, severity, date range
- Gemini frame filmstrip (the 8 key frames that triggered the alert)
- Full audit trail exportable for clinical review
- Admin Command Center with system health monitoring

**Why this matters for clinical deployment:**
> In real hospitals, every alert must be logged for liability, compliance, and quality improvement. Vital Guardian is built for this from day one.

---

### SLIDE 12 — Live Demo Slide

**Headline:** 🎬 Let's see it in action

**Demo sequence to run:**
1. Open the desktop app (double-click → loading screen → browser opens)
2. Show Patient Hub → multi-patient grid
3. Start monitoring → show live video + AI gauges
4. Trigger a fall/seizure clip → alert fires → Gemini verifies
5. Switch to **Nurse Dashboard** → show alert appears with no video
6. Nurse acknowledges alert
7. Show Patient History — the logged record

**Talking point:**
> *"Notice that on the nurse dashboard, there is no video whatsoever — yet the nurse has everything they need to respond. That is the privacy-by-design principle in action."*

---

### SLIDE 13 — Summary & Impact

**Headline:** What Vital Guardian delivers

**Three pillars:**

| Safety | Privacy | Reliability |
|---|---|---|
| AI detects falls & seizures in real-time | Nurses see alerts, not surveillance | Fully local — no internet, no cloud |
| Gemini verifies before alerting | Patient IDs anonymised | Runs on commodity GPU hardware |
| Audio distress monitoring | No video on nurse-facing UI | Permanent clinical audit trail |

**Closing statement:**
> *"Vital Guardian demonstrates that AI in healthcare doesn't have to choose between safety and privacy. With the right architecture, you get both."*

---

### SLIDE 14 — Thank You / Q&A

**Title:** Thank You

**Tagline:** *Vital Guardian — Watching when it matters most.*

**Team members + roles**
**GitHub / Demo link** (if applicable)
**Questions welcome**

---

## PRESENTER TIMING GUIDE

| Section | Slides | Target Time |
|---|---|---|
| FYP Overview | 2, 3, 4 | 90 seconds |
| Faculty Feedback | 5, 6 | 90 seconds |
| Work Done | 7–11 | 3 minutes |
| Live Demo | 12 | 1–2 minutes |
| Summary + Q&A | 13, 14 | 30 seconds |
| **Total** | **14 slides** | **~8–9 minutes** |

---

## KEY TRANSITIONS TO USE

- **Slide 4 → 5:** *"That's the system we built. Now, in our last evaluation, the faculty raised an important concern..."*
- **Slide 5 → 6:** *"The feedback was clear — patient privacy. And here is exactly how we addressed it..."*
- **Slide 6 → 7:** *"This architectural decision became the foundation of everything we built since. Let me walk you through the four deliverables..."*
- **Slide 8 → 9:** *"The dashboard solves the privacy problem. But to make it work in real-time, we needed the AI to run fast — locally, without any cloud dependency..."*
- **Slide 11 → 12:** *"Let me now show you all of this working live..."*
