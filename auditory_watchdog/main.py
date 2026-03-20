import time
import logging
from concurrent.futures import ThreadPoolExecutor

# Force TensorFlow to suppress info/warnings to keep console clean
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from core.audio_capture import AudioStream
from core.privacy_shield import PrivacyShield
from core.keyword_spotter import KeywordSpotter
from core.distress_classifier import DistressClassifier
from core.action_engine import ActionEngine

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AuditoryWatchdog")

def main():
    print("\n" + "="*50)
    print("      INITIALIZING AUDITORY WATCHDOG MODULE       ")
    print("="*50 + "\n")

    # Initialize all modules
    audio_stream = AudioStream()
    privacy_shield = PrivacyShield()
    
    # We want these heavy models loaded before we start the mic
    kws_module = KeywordSpotter()
    distress_module = DistressClassifier()
    
    action_engine = ActionEngine()
    
    # Thread pool for running KWS and YAMNet in parallel to maintain real-time
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        audio_stream.start_stream()
        logger.info("System is ACTIVE and listening...")
        
        while True:
            # 1. Get the latest 3-second audio chunk and new 1-second stride
            chunk_data = audio_stream.get_latest_chunk(timeout=1.0)
            if chunk_data is None:
                continue
                
            chunk, new_stride = chunk_data
                
            # 2. Privacy Shield
            should_analyze, speech_clip = privacy_shield.analyze_chunk(new_stride)
            
            if should_analyze:
                # 3. Parallel Analysis
                # Distress Classifier always runs on the 3-second overlapping window
                future_distress = executor.submit(distress_module.analyze_chunk, chunk)
                
                # Keyword Spotter only runs if the patient finished a complete sentence
                if speech_clip is not None:
                    future_kws = executor.submit(kws_module.analyze_chunk, speech_clip)
                    # Non-blocking callback: when whisper finishes, send alert automatically!
                    future_kws.add_done_callback(lambda f: action_engine.dispatch_alert(f.result()))
                
                # Check distress result immediately (block wait for YAMNet, which is < 0.1s)
                distress_result = future_distress.result()
                
                # 4. Action Engine (Distress only, KWS is async)
                action_engine.dispatch_alert(distress_result)
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}")
    finally:
        audio_stream.terminate()
        executor.shutdown()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
