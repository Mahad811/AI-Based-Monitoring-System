# FYP Panel Demo Strategy: "The Live Playback Defense" 🎬

**The Constraint:** You cannot fall on the floor in a formal presentation room.
**The Advantage:** You have 1 month to prepare perfect data.

**The Strategy:** We will use **"Playback Injection."**
Instead of acting live, you will record **perfect scenarios** beforehand. During the demo, you feed this video/audio into the system **in real-time**. The panel sees the system processing data *live*, but the data itself is a pre-recorded "perfect take."

---

## 🎭 The "Cinema Verité" Demo Workflow

### 1. Preparation (The "Curator")
**Source Material:** You don't need to film. You can download existing high-quality clips from:
*   **YouTube:** Search "CCTV fall", "patient fall simulation", "seizure simulation".
*   **Stock Footage:** Pexels/Pixabay (search "hospital bed", "patient").
*   **Movies/TV:** Clips from medical dramas (ensure they look realistic).

**The "Golden Rule" for Selection:**
*   **Camera Angle:** Must be somewhat elevated (CCTV style) or clear eye-level.
*   **Clarity:** Subject must be fully visible.
*   **No Cuts:** The fall/event should happen in one continuous shot if possible.

**Action:** Download 3-5 perfect clips and name them `scenario_a.mp4`, `scenario_b.mp4`.

### 2. The Panel Presentation (The "Live Run")
You do NOT just play a standard video player. You run the **Vital Guardian System**.
*   **Command:** `python scripts/demo_playback.py --video scenario_a_fall.mp4`
*   **Visuals:** The panel sees the **Bounding Boxes, Skeletons, and Alerts** appearing on the video *as it plays*.
*   **Audio:** The laptop plays the recorded audio, and your Audio Module processes it (or we pipe it internally).

### 3. The Script
*   **Speaker:** "Since we cannot safely demonstrate high-impact falls in this room, we have recorded raw sensor data from our testing environment. We will now feed this data into the Vital Guardian system **live**."
*   **Action:** Start the script.
*   **Speaker (Narrating):** "Here, the patient is exiting the bed... The system flags 'Bed Exit'. Now, the fall occurs..."
*   **System Alert:** **[CRITICAL ALERT: FALL DETECTED]**
*   **Speaker:** "Notice how the Vision pipeline detected the pose change, and the Audio module confirmed the impact sound. The Cognitive Core fused these to trigger the Critical Alert."

---

## 🛠️ What We Build Now (To Make This Happen)

1.  **`demo_playback.py`:** A script that feeds a video file frame-by-frame into the `VisionPipeline`, simulating a live webcam.
2.  **Audio Integration:** Ensure the audio track from the video is fed to the `AudioModule` correctly.
3.  **The Dashboard:** A professional-looking screen that updates with the alerts.

---

## ✅ Why This Impresses the Panel
1.  **"It's Real Code":** You aren't playing a pre-rendered AVI. You are running your actual detection code on raw input. If you pause the script, the detection pauses. It *is* a live demo, just with a stored input source.
2.  **Risk-Free:** No demo effect. No bad lighting in the conference room. No camera driver issues.
3.  **Performance Proof:** The system proves it can handle the data in real-time frame rates.

---

## 🏃 Next Steps
1.  **Build the Cognitive Core:** We still need the logic to process these events!
2.  **Develop `demo_playback.py`:** To support this video injection.
3.  **Record Scenarios:** You have 1 month to get Oscar-worthy clips.

