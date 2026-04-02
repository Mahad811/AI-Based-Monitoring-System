"""
Quick standalone test for the Gemini 3 Flash Preview Verifier.
Run this to ensure your API key and model access are working!
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Load .env explicitly to guarantee GEMINI_API_KEY is found
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from cognitive_core.gemini_verifier import GeminiVerifier

def main():
    print("====================================================")
    print("🧪 TESTING GEMINI 3 FLASH PREVIEW INTEGRATION")
    print("====================================================")
    
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file.")
        return
    else:
        print(f"✅ Found API Key: {key[:8]}...{key[-4:]}")

    # Create dummy frame (black with some text so it's not totally blank)
    print("📸 Creating dummy camera frame...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Patient on floor", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    print("🤖 Initializing GeminiVerifier...")
    verifier = GeminiVerifier(mock_mode=False)
    
    if verifier.mock_mode:
        print("❌ Verifier fell back to mock mode. Is the 'google-genai' package installed?")
        return
        
    print("🚀 Sending request to Gemini 3 Flash Preview (this takes ~2-5s)...")
    
    # Simulate a fall alert that the VisionPipeline just flagged
    response = verifier.verify_alert(
        event_type="fall",
        confidence=0.88,
        patient_id="Patient A",
        frame=frame
    )
    
    print("\n📩 GEMINI RESPONSE RECEIVED:")
    print("-" * 50)
    print(f"Decision : {response.get('decision')}")
    print(f"Headline : {response.get('headline')}")
    print(f"Narrative: {response.get('narrative')}")
    print("Actions  :")
    for i, act in enumerate(response.get('actions', [])):
        print(f"  {i+1}. {act}")
    print("-" * 50)
    print("\n✅ Test Complete!")

if __name__ == "__main__":
    main()
