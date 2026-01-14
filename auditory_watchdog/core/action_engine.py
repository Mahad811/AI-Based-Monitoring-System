import json
import logging
from datetime import datetime
import os

from auditory_watchdog.config import LOG_DIR

logger = logging.getLogger(__name__)

class ActionEngine:
    """
    Executes post-activity actions proportional to the event detected.
    In a full production environment, this would forward alerts to a dashboard
    (e.g., via WebSockets or MQTT). For the prototype, it logs to a local file
    and prints formatted console alerts.
    """
    def __init__(self):
        # We will keep a daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_filepath = os.path.join(LOG_DIR, f"alerts_{date_str}.log")
        logger.info(f"Action Engine initialized. Logging to {self.log_filepath}")

    def dispatch_alert(self, event_data: dict):
        """
        Receives an event dict from KWS or Distress Classifier and handles it.
        """
        if not event_data.get("event_detected", False):
            return

        timestamp = datetime.now().isoformat()
        event_type = event_data.get("event_type", "Unknown")
        
        # Determine Priority based on event type
        priority = "CRITICAL"
        
        if event_type == "Preverbal_Distress":
            if "cough" in event_data.get("primary_sound", "").lower() or "breathing" in event_data.get("primary_sound", "").lower():
                priority = "HIGH"
        elif event_type == "Patient_Speech":
            priority = "MEDIUM (For LLM Evaluation)"
        
        payload = {
            "timestamp": timestamp,
            "priority": priority,
            "event_type": event_type,
            "data": event_data
        }

        self._log_to_file(payload)
        self._print_console_alert(payload)

    def _log_to_file(self, payload: dict):
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.error(f"Failed to log alert to file: {e}")

    def _print_console_alert(self, payload: dict):
        """
        Prints a highly visible console alert.
        """
        print("\n" + "="*50)
        print(f"🚨🚨 CRITICAL ALERT DETECTED [{payload['priority']}] 🚨🚨")
        print(f"Time: {payload['timestamp']}")
        print(f"Type: {payload['event_type']}")
        
        if payload['event_type'] == "Patient_Speech":
            lang = payload['data']['language']
            text = payload['data']['text']
            print(f"Detected Language: {lang.upper()}")
            print(f"Raw Transcript: '{text}'")
            print("▶️ ACTION: Forwarding Transcript to LLM Core...")
            
        elif payload['event_type'] == "Preverbal_Distress":
            sound = payload['data']['primary_sound']
            conf = payload['data']['details'][0]['confidence']
            print(f"Sound Detected: {sound}")
            print(f"Confidence: {conf*100:.1f}%")
            print("▶️ ACTION: Forwarding Distress Event to LLM Core...")

        print("="*50 + "\n")
