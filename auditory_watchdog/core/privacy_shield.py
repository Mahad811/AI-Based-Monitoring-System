import time
import logging
import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps
from auditory_watchdog.config import (
    SAMPLE_RATE, VAD_THRESHOLD, HEURISTIC_LONG_SPEECH_SEC, 
    VISITOR_MODE_COOLDOWN_SEC, AUDIO_STRIDE_MS,
    HEURISTIC_PAUSE_TOLERANCE_SEC, SENTENCE_PAUSE_FLUSH_SEC
)

logger = logging.getLogger(__name__)

class PrivacyShield:
    def __init__(self):
        """
        Initializes Gate 1 (VAD) and Gate 2 (Visitor Heuristics).
        Uses silero-vad for highly efficient CPU-based VAD.
        """
        logger.info("Loading Silero VAD model...")
        self.vad_model = load_silero_vad(onnx=True)
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
        self.in_visitor_mode = False
        self.visitor_mode_until = 0.0
        self.speech_buffer = []

        # Based on config, how many continuous chunks mean a "long conversation"
        # Now based on AUDIO_STRIDE_MS since that's how often chunks are evaluated
        self.long_speech_chunk_limit = int((HEURISTIC_LONG_SPEECH_SEC * 1000) / AUDIO_STRIDE_MS)
        if self.long_speech_chunk_limit < 2:
            self.long_speech_chunk_limit = 2
            
        self.pause_tolerance_limit = int((HEURISTIC_PAUSE_TOLERANCE_SEC * 1000) / AUDIO_STRIDE_MS)
        self.sentence_flush_limit = int((SENTENCE_PAUSE_FLUSH_SEC * 1000) / AUDIO_STRIDE_MS)

    def analyze_chunk(self, new_stride_chunk: np.ndarray) -> tuple:
        """
        Determines if the audio should be monitored, and buffers speech for complete sentence transcription.
        
        Returns: (should_monitor, completed_speech_clip) 
                 should_monitor is False if Visitor Mode is active.
                 completed_speech_clip is a numpy array of a full spoken sentence, or None.
        """
        current_time = time.time()

        # Gate 2 Check: Are we already locked in Visitor Mode?
        if self.in_visitor_mode:
            if current_time < self.visitor_mode_until:
                # Still in visitor mode, drop everything
                self.speech_buffer = []
                return False, None
            else:
                logger.info("Privacy Shield: Exiting Visitor Mode. Resuming monitoring.")
                self.in_visitor_mode = False
                self.consecutive_speech_chunks = 0
                self.consecutive_silence_chunks = 0
                self.speech_buffer = []

        # Convert chunk to tensor for Silero
        if new_stride_chunk.ndim > 1:
            new_stride_chunk = new_stride_chunk.squeeze()
        audio_tensor = torch.from_numpy(new_stride_chunk).float()

        # Gate 1: Check for speech
        # get_speech_timestamps returns a list of dictionaries with 'start' and 'end' of speech
        try:
            speech_timestamps = get_speech_timestamps(
                audio_tensor, 
                self.vad_model, 
                threshold=VAD_THRESHOLD,
                sampling_rate=SAMPLE_RATE,
                return_seconds=True
            )
            has_speech = len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"VAD failed: {e}")
            has_speech = False

        completed_speech_clip = None

        if has_speech:
            self.consecutive_speech_chunks += 1
            self.consecutive_silence_chunks = 0
            self.speech_buffer.append(new_stride_chunk)
            
            # Check if this speech has been going on for too long (likely a conversation)
            if self.consecutive_speech_chunks >= self.long_speech_chunk_limit:
                logger.warning(f"Privacy Shield: Continuous speech detected for > {HEURISTIC_LONG_SPEECH_SEC}s. Activating Visitor Mode.")
                self.in_visitor_mode = True
                self.visitor_mode_until = current_time + VISITOR_MODE_COOLDOWN_SEC
                self.speech_buffer = [] # Trash the conversation string to preserve privacy
                return False, None
                
            return True, None
        else:
            # Silence or non-speech noise detected. 
            self.consecutive_silence_chunks += 1
            
            if len(self.speech_buffer) > 0:
                self.speech_buffer.append(new_stride_chunk)
                
                # Check if they have paused long enough to consider the sentence "finished"
                if self.consecutive_silence_chunks >= self.sentence_flush_limit:
                    completed_speech_clip = np.concatenate(self.speech_buffer)
                    self.speech_buffer = []
                    logger.info("Privacy Shield: Sentence completed. Flushing to Whisper.")
            
            # Reset conversation if silence holds for long enough
            if self.consecutive_silence_chunks > self.pause_tolerance_limit:
                if self.consecutive_speech_chunks > 0:
                    logger.debug("Privacy Shield: Extended silence detected. Resetting conversation counter.")
                self.consecutive_speech_chunks = 0
                
            # It should still be monitored for distress (coughs, glass breaking, etc.)
            return True, completed_speech_clip

if __name__ == "__main__":
    # Small test
    shield = PrivacyShield()
    dummy_silent = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    dummy_noisy = np.random.normal(0, 0.5, SAMPLE_RATE * 3).astype(np.float32)
    
    # Needs a real voice clip for true positive testing, but we can verify it doesn't crash
    print(f"Silent chunk passed? {shield.analyze_chunk(dummy_silent)}")
    print(f"Noisy chunk passed? {shield.analyze_chunk(dummy_noisy)}")
