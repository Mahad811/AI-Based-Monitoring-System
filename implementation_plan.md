# Cognitive Core Implementation Plan

## Goal
Implement the **Cognitive Core**, the central intelligence of Vital Guardian.
It fuses **Vision** (Falls, Seizures, Poses) and **Audio** (Distress, Keywords) to generate:
1.  **Instant Alerts (<100ms):** Via the deterministic `ReflexEngine`.
2.  **Contextual Reports (2-5s):** Via the LLM-powered `ReasoningEngine` (Gemini).

---

## User Review Required
> [!IMPORTANT]
> **API Key:** Usage of `ReasoningEngine` requires a valid Google Gemini API key in `.env`.
> **Internet:** The demo machine MUST have internet access for Gemini.
> **Dependency:** We will add `google-generativeai` to `requirements.txt`.

---

## Proposed Changes

### 1. Cognitive Core Module (`cognitive_core/`)

#### [NEW] [reflex.py](file:///d:/project/FYP/cognitive_core/reflex.py)
*   **Purpose:** Fast, deterministic rule engine.
*   **Logic:**
    *   `Fall + Impact = Critical`
    *   `Fall + Silence = High (Unconscious)`
    *   `Seizure + Groan = Critical`
    *   `Normal + Help = Medium`
*   **Input:** `VisionState` (dict), `AudioEvent` (dict)
*   **Output:** `Alert` object (Level, Confidence, Message)

#### [NEW] [reasoning.py](file:///d:/project/FYP/cognitive_core/reasoning.py)
*   **Purpose:** LLM wrapper for Gemini 2.5.
*   **Trigger:** Only runs when `ReflexEngine` outputs `MEDIUM` or higher.
*   **Logic:** Sends last 30s of event log -> Returns natural language summary.
*   **Mock Mode:** Can return dummy strings if internet is down (Demo Safety).

#### [NEW] [simulator.py](file:///d:/project/FYP/cognitive_core/simulator.py)
*   **Purpose:** "Wizard of Oz" Audio Injector.
*   **Logic:** Listens for keypresses (`T`, `H`, `S`) and injects fake `AudioEvent`s into the pipeline.

### 2. Configuration (`config/`)

#### [MODIFY] [config.yaml](file:///d:/project/FYP/config/config.yaml)
*   Add `cognitive` section with fusion thresholds and API settings.

### 3. Demo Script (`scripts/`)

#### [NEW] [demo_playback.py](file:///d:/project/FYP/scripts/demo_playback.py)
*   Loads a video file.
*   Runs Vision Pipeline on frames.
*   Accepts Keyboard Input for Audio.
*   Displays overlay with "Cognitive State".

---

## Verification Plan

### Automated Tests
*   `test_reflex.py`: Unit test logic matrix (e.g., ensure `Fall`+`Thud`=`Critical`).
*   `test_gemini_connection.py`: Simple ping to Gemini API to verify key.

### Manual Verification
*   **The "Thud" Test:** Run `demo_playback.py`.
    *   Wait for visual fall.
    *   Press 'T'.
    *   Verify Alert goes RED.
*   **The "Essay" Test:**
    *   Trigger an alert.
    *   Verify console prints a Gemini-generated summary ("Subject fell at 10:00...").
