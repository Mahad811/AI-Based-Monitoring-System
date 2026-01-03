import logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv
import urllib.request
import os
from auditory_watchdog.config import YAMNET_MODEL_HANDLE, DISTRESS_CLASSES, DISTRESS_CONFIDENCE_THRESHOLD, LOG_DIR

logger = logging.getLogger(__name__)

class DistressClassifier:
    def __init__(self):
        """
        Initializes the YAMNet model for detecting non-verbal distress sounds.
        """
        logger.info(f"Loading YAMNet model from {YAMNET_MODEL_HANDLE}...")
        try:
            self.model = hub.load(YAMNET_MODEL_HANDLE)
            logger.info("YAMNet loaded successfully.")
            
            # YAMNet requires the class map to translate predictions (0-520) into strings
            self.class_map_path = self.model.class_map_path().numpy().decode('utf-8')
            self.class_names = self._load_class_map(self.class_map_path)
            
            # Find the numeric indices for our target classes
            self.target_indices = []
            for idx, name in enumerate(self.class_names):
                for target in DISTRESS_CLASSES:
                    if target.lower() in name.lower():
                        self.target_indices.append(idx)
                        logger.debug(f"Mapped Target '{target}' to YAMNet index: {idx} ({name})")
                        
            # Use a list since we iterate over it later but make items unique
            self.target_indices = list(set(self.target_indices))

        except Exception as e:
            logger.error(f"Failed to load YAMNet: {e}")
            raise

    def _load_class_map(self, class_map_csv):
        """Reads the YAMNet CSV class map."""
        with tf.io.gfile.GFile(class_map_csv) as csvfile:
            reader = csv.DictReader(csvfile)
            class_names = [row['display_name'] for row in reader]
        return class_names

    def analyze_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Runs YAMNet prediction on the audio chunk.
        Checks if the top predictions correspond to distress classes.
        """
        # Ensure 1D float32 mapped correctly
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.squeeze()
        audio_chunk = audio_chunk.astype(np.float32)

        # Robust RMS Normalization
        # Calculate Root Mean Square to measure average power
        rms = np.sqrt(np.mean(audio_chunk**2))
        if rms > 0:
            # Scale the audio so the RMS is around 0.1 (a healthy volume level)
            target_rms = 0.1
            
            # CRITICAL: Prevent over-amplifying background hiss during silence
            # Cap the multiplier so we don't boost a quiet room into a jet engine
            multiplier = min(target_rms / rms, 3.0) 
            audio_chunk = audio_chunk * multiplier
            
            # Clip to [-1.0, 1.0] to prevent clipping distortion
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

        try:
            # YAMNet inference
            scores, embeddings, spectrogram = self.model(audio_chunk)
            
            # Scores is shape [num_frames (>num chunks), 521]
            # Take the MAX score across frames. 
            # This captures short, strong transient sounds (coughs, fast breaths)
            # that might only exist in a single 0.48s YAMNet frame.
            agg_scores = np.max(scores.numpy(), axis=0)
            
            # Get the top class
            top_class_idx = np.argmax(agg_scores)
            top_class_name = self.class_names[top_class_idx]
            top_score = agg_scores[top_class_idx]
            
            # Debugging top classes
            logger.info(f"YAMNet Top Sound: '{top_class_name}' ({top_score:.2f})")

            # Check if any of our target indices exceed the threshold
            detected_distress = []
            for target_idx in self.target_indices:
                score = agg_scores[target_idx]
                if score >= DISTRESS_CONFIDENCE_THRESHOLD:
                    detected_distress.append({
                        "sound": self.class_names[target_idx],
                        "confidence": float(score)
                    })

            if len(detected_distress) > 0:
                # Sort by confidence
                detected_distress.sort(key=lambda x: x['confidence'], reverse=True)
                primary_sound = detected_distress[0]['sound']
                logger.warning(f"Distress Audio Detected: {primary_sound} (Confidence: {detected_distress[0]['confidence']:.2f})")
                
                return {
                    "event_detected": True,
                    "event_type": "Preverbal_Distress",
                    "details": detected_distress,
                    "primary_sound": primary_sound
                }

            return {"event_detected": False, "reason": f"No distress crossed threshold. Top non-distress sound: {top_class_name} ({top_score:.2f})"}

        except Exception as e:
            logger.error(f"Distress Classification failed: {e}")
            return {"event_detected": False, "reason": "YAMNet exception"}

if __name__ == "__main__":
    print("Testing YAMNet module loading...")
    classifier = DistressClassifier()
    print("Loading successful! Known indices:", classifier.target_indices)
