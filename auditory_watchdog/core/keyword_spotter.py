import logging
from faster_whisper import WhisperModel
from auditory_watchdog.config import WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_DOWNLOAD_ROOT
import numpy as np

logger = logging.getLogger(__name__)

class KeywordSpotter:
    def __init__(self):
        """
        Initializes the faster-whisper model for zero-shot keyword spotting.
        The model processes raw audio chunks and determines if keywords are present.
        """
        logger.info(f"Loading faster-whisper model '{WHISPER_MODEL_SIZE}'...")
        # device dictates CPU vs GPU load
        try:
            self.model = WhisperModel(
                WHISPER_MODEL_SIZE, 
                device=WHISPER_DEVICE, 
                compute_type=WHISPER_COMPUTE_TYPE,
                download_root=WHISPER_DOWNLOAD_ROOT
            )
            logger.info("Faster-Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}")
            raise

    def analyze_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Transcribes the full speech sentence to feed to the future LLM module.
        """
        # Ensure audio is float32 1D array as expected by whisper
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.squeeze()
        audio_chunk = audio_chunk.astype(np.float32)
        
        try:
            # We don't force a language, whisper auto-detects English or Urdu
            # We also set word_timestamps=False to speed it up
            # Provide a broad medical context prompt instead of specific keywords
            initial_prompt = "Patient in distress: help aid nurse doctor. Urdu: مدد پانی درد تکلیف نرس ڈاکٹر"
            segments, info = self.model.transcribe(
                audio_chunk, 
                beam_size=7, 
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400)
            )
            
            # Combine all transcribed segments into one string
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            if not transcribed_text:
                return {"event_detected": False, "reason": "No speech recognized"}

            logger.debug(f"Speech Transcribed ({info.language}): '{transcribed_text}'")

            return {
                "event_detected": True,
                "event_type": "Patient_Speech",
                "language": info.language,
                "text": transcribed_text
            }

        except Exception as e:
            logger.error(f"KWS Analysis failed: {e}")
            return {"event_detected": False, "reason": "Whisper exception"}

if __name__ == "__main__":
    print("Testing Whisper KWS module loading...")
    kws = KeywordSpotter()
    print("Loading successful!")
