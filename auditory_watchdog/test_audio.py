import time
import logging
import argparse
import numpy as np
import wave
import sys

# Force TensorFlow to suppress info/warnings to keep console clean
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from config import SAMPLE_RATE, CHUNK_SIZE
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
logger = logging.getLogger("OfflineTester")

def load_wav_file(filepath: str) -> np.ndarray:
    """Loads a .wav file and converts it to a 1D float32 numpy array at 16kHz."""
    try:
        with wave.open(filepath, 'rb') as wf:
            framerate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            
            raw_data = wf.readframes(n_frames)
            
            if sampwidth == 2:
                # 16-bit audio
                audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                logger.error(f"Unsupported sample width: {sampwidth} bytes. Please use 16-bit WAV.")
                sys.exit(1)
                
            # Convert to mono if stereo
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
            # NOTE: We assume the wav file is already 16kHz. 
            # In production we would resample, but for testing we assume user provides 16kHz wavs.
            if framerate != SAMPLE_RATE:
                logger.warning(f"File sample rate is {framerate}Hz, expected {SAMPLE_RATE}Hz. Analysis may be skewed.")
                
            return audio_data
    except Exception as e:
        logger.error(f"Failed to load '{filepath}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test Auditory Watchdog Offline using a .wav file.")
    parser.add_argument("wav_file", help="Path to the .wav file to analyze")
    args = parser.parse_args()
    
    if not os.path.exists(args.wav_file):
        logger.error(f"File not found: {args.wav_file}")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("      INITIALIZING OFFLINE TESTING MODULE       ")
    print("="*50 + "\n")

    # Initialize all modules (No audio_stream needed)
    privacy_shield = PrivacyShield()
    kws_module = KeywordSpotter()
    distress_module = DistressClassifier()
    action_engine = ActionEngine()
    
    # Load the audio into memory
    logger.info(f"Loading '{args.wav_file}'...")
    audio_full = load_wav_file(args.wav_file)
    logger.info(f"Loaded {(len(audio_full)/SAMPLE_RATE):.2f} seconds of audio.")
    
    print("\n" + "="*50)
    print("      STARTING ANALYSIS       ")
    print("="*50 + "\n")

    # Split into 3-second chunks and process sequentially
    total_chunks = len(audio_full) // CHUNK_SIZE
    if len(audio_full) % CHUNK_SIZE > 0:
        total_chunks += 1
        
    for i in range(total_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, len(audio_full))
        
        chunk = audio_full[start_idx:end_idx]
        
        # If the last chunk is too small, pad it with zeros up to 3 seconds
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)), 'constant')
            
        print(f"\n--- Analyzing Chunk {i+1}/{total_chunks} [{(i*3):.1f}s - {((i+1)*3):.1f}s] ---")
        
        # 1. Privacy Shield
        should_analyze, speech_clip = privacy_shield.analyze_chunk(chunk)
        
        if should_analyze:
            # 2. Sequential Analysis (Offline = no need for ThreadPool)
            distress_result = distress_module.analyze_chunk(chunk)
            action_engine.dispatch_alert(distress_result)
            
            if speech_clip is not None:
                kws_result = kws_module.analyze_chunk(speech_clip)
                action_engine.dispatch_alert(kws_result)
        else:
            logger.info(f"Privacy Shield blocked chunk {i+1}")
            
    print("\n" + "="*50)
    print("      FILE PROCESSING COMPLETE       ")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
