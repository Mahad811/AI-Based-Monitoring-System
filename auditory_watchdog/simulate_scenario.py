import numpy as np
import time
from core.privacy_shield import PrivacyShield
from config import SAMPLE_RATE

def simulate_conversation_pauses():
    shield = PrivacyShield()
    
    print("\n[TEST 1] Testing Conversation Pause Reset Loophole with 1-second strides")
    print("Simulating a 21-second conversation where the person takes a breath/pauses every 9 seconds...\n")
    
    # 9 chunks speech (9s), 3 chunks silence (3s), 9 chunks speech (9s)
    
    events = (
        ["Speech"] * 9 +
        ["Silence"] * 3 +
        ["Speech"] * 9
    )
    
    dummy_speech = np.random.normal(0, 0.5, SAMPLE_RATE * 3).astype(np.float32)
    dummy_silence = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    
    # We monkeypatch get_speech_timestamps to predictably force Silero VAD response for speed
    import core.privacy_shield
    def mock_vad(tensor, model, threshold, sampling_rate, return_seconds):
        # We pass side effects via a temporary flag we set below
        if hasattr(mock_vad, 'force_speech') and mock_vad.force_speech:
            return [{'start': 0, 'end': 1}] # fake speech
        return []
    core.privacy_shield.get_speech_timestamps = mock_vad
    
    for idx, event in enumerate(events):
        print(f"Time {idx}s to {idx+1}s -> {event}")
        if event == "Speech":
            mock_vad.force_speech = True
            should_monitor, speech_clip = shield.analyze_chunk(dummy_speech)
        else:
            mock_vad.force_speech = False
            should_monitor, speech_clip = shield.analyze_chunk(dummy_silence)
            
        is_flushed = "Yes!" if speech_clip is not None else "No"
        print(f"   Privacy Shield returned: Monitor={should_monitor}, VisitorModeActive={shield.in_visitor_mode}, FlushedSentence={is_flushed}")
        
    if not shield.in_visitor_mode:
        print("\n[FAILED] The conversation lasted 21 seconds but Visitor Mode NEVER triggered because a 3-second pause reset the counter entirely!")
    else:
        print("\n[PASSED] Visitor Mode triggered correctly.")

if __name__ == "__main__":
    simulate_conversation_pauses()
