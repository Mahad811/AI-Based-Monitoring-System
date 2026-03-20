"""
Quick vision module test: webcam/video → unified events + latency printouts.
"""

import sys
import time
import cv2
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_guardian.tracker import PatientTracker


def load_config():
    with open(str(Path(__file__).resolve().parents[1] / 'config' / 'config.yaml'), 'r') as f:
        return yaml.safe_load(f)


def main(video_path: str | None = None, cam_index: int = 0):
    config = load_config()
    tracker = PatientTracker(config)

    cap = cv2.VideoCapture(video_path if video_path else cam_index)
    if not cap.isOpened():
        print('Failed to open video source')
        return

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            event = tracker.process_frame(frame)
            elapsed_ms = (time.time() - start) * 1000.0

            print({
                'event_type': event['event_type'],
                'confidence': round(event['confidence'], 3),
                'latency_ms': round(elapsed_ms, 1),
                'extras': {
                    'pose_state': event.get('metadata', {}).get('pose_state'),
                    'dangerous': event.get('metadata', {}).get('dangerous_movement'),
                    'bed_exit': event.get('metadata', {}).get('bed_exit_check')
                }
            })

            # Optional display with bbox
            md = event.get('metadata', {})
            bbox = md.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, event['event_type'], (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Visual Guardian', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # To test with a file: python scripts/test_visual_guardian.py
    # Then edit below to set a path, or pass via environment in your IDE.
    main(video_path=None, cam_index=0)


