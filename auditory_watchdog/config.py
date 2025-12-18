import os

# --- OFFLINE BUNDLING (CACHE HIJACKING) ---
# We force all AI libraries to store and load their multi-gigabyte models right here
# in the project folder, so the entire folder can be zipped and sent to a partner offline!
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# We must set these environment variables sequentially BEFORE any other library imports
os.environ["TFHUB_CACHE_DIR"] = os.path.join(MODELS_DIR, "tfhub")
os.environ["TORCH_HOME"] = os.path.join(MODELS_DIR, "torch")
os.environ["HF_HOME"] = os.path.join(MODELS_DIR, "huggingface")

# --- AUDIO CAPTURE SETTINGS ---
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 3000  # 3 seconds rolling buffer window
CHUNK_SIZE = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000.0))
AUDIO_STRIDE_MS = 1000    # Move forward by 1 second at a time (2 second overlap)
AUDIO_STRIDE_SAMPLES = int(SAMPLE_RATE * (AUDIO_STRIDE_MS / 1000.0))
NUM_CHANNELS = 1

# --- PRIVACY SHIELD (VAD) SETTINGS ---
VAD_THRESHOLD = 0.5  # Confidence threshold for human speech
MIN_SPEECH_DURATION_MS = 250  # Minimum duration to be considered valid speech
VISITOR_MODE_COOLDOWN_SEC = 10 # Seconds to ignore audio after a long conversation is detected
HEURISTIC_LONG_SPEECH_SEC = 15  # If there is continuous speech for this long, trigger visitor mode
HEURISTIC_PAUSE_TOLERANCE_SEC = 4 # Seconds of silence allowed before the entire visitor mode conversation resets
SENTENCE_PAUSE_FLUSH_SEC = 2 # Seconds of silence before we assume the patient finished their sentence and send to Whisper

# --- KEYWORD SPOTTING (KWS) SETTINGS ---
WHISPER_MODEL_SIZE = "tiny"
WHISPER_DOWNLOAD_ROOT = os.path.join(MODELS_DIR, "whisper")
# Using "cpu" or "int8" to avoid CUDA 12 driver issues if they aren't installed globally
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_DEVICE = "cpu"

# --- DISTRESS CLASSIFICATION SETTINGS ---
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
# In ESC-50/YAMNet, these classes represent sounds we might care about
# e.g., 16 is 'Breathing', 22 is 'Cough', 25 is 'Throat clearing', 28 is 'Crying, sobbing'
# We will refine these based on YAMNet's specific class map later.
DISTRESS_CLASSES = ['Cough', 'Crying, sobbing', 'Breathing', 'Baby cry, infant cry', 'Wail, moan', 'Groan', 'Gasp'] 
DISTRESS_CONFIDENCE_THRESHOLD = 0.15

# --- SYSTEM SETTINGS ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
