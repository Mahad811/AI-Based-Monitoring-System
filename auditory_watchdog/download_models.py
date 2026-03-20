import os
import logging
from config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predownload_all_models():
    logger.info(f"Starting Offline Model Packager...")
    logger.info(f"All models will be downloaded directly to: {MODELS_DIR}\n")
    
    # 1. YAMNet (TF Hub)
    logger.info("--- 1. Downloading YAMNet (Distress Classifier) ---")
    import tensorflow_hub as hub
    from config import YAMNET_MODEL_HANDLE
    logger.info(f"Fetching from {YAMNET_MODEL_HANDLE} into {os.environ.get('TFHUB_CACHE_DIR')}")
    hub.load(YAMNET_MODEL_HANDLE)
    logger.info("YAMNet downloaded successfully.\n")
    
    # 2. Silero VAD (Torch Hub)
    logger.info("--- 2. Downloading Silero VAD (Privacy Shield) ---")
    from silero_vad import load_silero_vad
    logger.info(f"Fetching into {os.environ.get('TORCH_HOME')}")
    load_silero_vad(onnx=True)
    logger.info("Silero VAD downloaded successfully.\n")
    
    # 3. Faster-Whisper (HuggingFace/CTranslate2)
    logger.info("--- 3. Downloading Faster-Whisper (Speech Transcriber) ---")
    from core.keyword_spotter import KeywordSpotter
    from config import WHISPER_DOWNLOAD_ROOT
    logger.info(f"Fetching into {WHISPER_DOWNLOAD_ROOT}")
    # Initializing the class triggers the download
    KeywordSpotter()
    logger.info("Faster-Whisper downloaded successfully.\n")

    logger.info("="*50)
    logger.info("✅ PACKAGING COMPLETE ✅")
    logger.info(f"You can now safely zip the entire project folder.")
    logger.info(f"When unzipped on any computer, it will find the models in the 'models' folder and run 100% offline!")
    logger.info("="*50)

if __name__ == "__main__":
    predownload_all_models()
