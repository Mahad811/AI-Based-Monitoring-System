# 🩺 Auditory Watchdog - LLM Audio Sensory Module (Offline Edition)

This repository contains the completely offline, real-time auditory sensory module for the FYP Watchdog system. It acts as the "Ears" of the system, autonomously filtering background noise, detecting physical distress, and cleanly transcribing patient speech into formatted JSON payloads designed specifically to be fed into your downstream LLM master-brain.

## 🚀 Key Architectural Features
1. **100% Offline Edge Inference:** All multi-gigabyte AI models (YAMNet, Silero, Faster-Whisper) have been permanently cached inside the `models/` directory. This module requires **zero internet connection** and never pings external servers.
2. **Dual-Gated Privacy Shield:** Uses advanced Voice Activity Detection (VAD) to preserve privacy. If a continuous conversation or TV is playing, the module engages "Visitor Mode" to pause speech-transcription—while still actively listening for coughs or gasps.
3. **Continuous Overlapping Distress (YAMNet):** A 1-second strided rolling chunk perfectly guarantees no cough, moan, or gasp is ever sliced in half or missed.
4. **Full-Sentence LLM Context (Faster-Whisper):** Instead of stuttering word-by-word, the module magically detects when the patient *finishes* their sentence, stitches it together seamlessly, and transcribes the entire context in one go. It natively handles mixed English/Urdu seamlessly.

---

## 🛠️ Setup Instructions (For Integration Partner)

Because the models are already bundled inside the `models/` directory, the setup takes less than two minutes:

1. **Extract the ZIP** anywhere on your machine.
2. **Open a Terminal** inside the `fyp_auditory_watchdog` folder.
3. **Create a fresh Python Virtual Environment** (Highly Recommended to avoid conflicting libraries):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   # source venv/bin/activate
   ```
4. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the Live Microphone Module!**
   ```bash
   python main.py
   ```

*(Note: The very first time you run the script, standard TensorFlow warnings may appear in the terminal, this is completely normal.)*

---

## 📡 The LLM Payload Structure

As the integration partner building the LLM logic, you will intercept the payloads fired by `core/action_engine.py`. 

The Action Engine outputs two strict event types:

### 1. Distress Events (Coughs, Gasps, Breathing)
When the patient physically struggles, YAMNet triggers a `Preverbal_Distress` payload:
```json
{
  "timestamp": "2026-03-20T19:30:15.123",
  "priority": "HIGH",
  "event_type": "Preverbal_Distress",
  "data": {
      "event_detected": true,
      "event_type": "Preverbal_Distress",
      "primary_sound": "Cough",
      "details": [
         {"sound": "Cough", "confidence": 0.85}
      ]
  }
}
```

### 2. Full-Sentence Speech Events
When the patient finishes a sentence (e.g., "Nurse, mujhe dard ho raha hai"), Whisper evaluates the full phrase and shoots a `Patient_Speech` payload for your LLM to decipher:
```json
{
  "timestamp": "2026-03-20T19:35:42.880",
  "priority": "MEDIUM (For LLM Evaluation)",
  "event_type": "Patient_Speech",
  "data": {
      "event_detected": true,
      "event_type": "Patient_Speech",
      "language": "ur",
      "text": "nurse mujhe dard ho raha hai jaldi aao"
  }
}
```

### Next Integration Steps
To connect this cleanly to your LLM module, open `core/action_engine.py`. Instead of the current `print_console_alert()` function, simply forward the `payload` JSON dictionary straight to your local LLM inference script, websocket server, or API!
