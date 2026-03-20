"""
Live Camera Demo for Fall Detection

Real-time demonstration of the vision pipeline using webcam.
Displays bounding box, probabilities, and alerts.
"""

import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visual_guardian import VisionPipeline


class LiveDemo:
    """Live camera demo for fall detection"""
    
    def __init__(self, config_path, camera_id=0):
        """
        Args:
            config_path: Path to config.yaml
            camera_id: Camera device ID (default: 0)
        """
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        
        # Initialize pipeline
        print("Initializing vision pipeline...")
        self.pipeline = VisionPipeline(config['vision'])
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {camera_id} opened successfully")
        print("\nControls:")
        print("  'q' or ESC: Quit")
        print("  'r': Reset pipeline buffers")
        print("\nStarting live demo...\n")
    
    def draw_bbox(self, frame, bbox, color, thickness=2):
        """Draw bounding box on frame"""
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    def draw_probability_bar(self, frame, label, probability, x, y, width=200, height=20):
        """Draw a horizontal probability bar"""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        
        # Filled portion
        fill_width = int(width * probability)
        if probability >= 0.6:
            color = (0, 0, 255)  # Red for high probability
        elif probability >= 0.3:
            color = (0, 165, 255)  # Orange for medium
        else:
            color = (0, 255, 0)  # Green for low
        
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
        
        # Label and percentage
        label_text = f"{label}: {probability*100:.1f}%"
        cv2.putText(frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    def draw_alert(self, frame, event_type):
        """Draw alert overlay"""
        h, w = frame.shape[:2]
        
        if event_type == 'fall':
            # Red alert box
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Alert text
            cv2.putText(frame, "FALL DETECTED!", (w//2 - 150, 50), 
                       cv2.FONT_HERSHEY_BOLD, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        elif event_type == 'seizure':
            # Orange alert box
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 165, 255), -1)  # Orange BGR
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Alert text
            cv2.putText(frame, "SEIZURE DETECTED!", (w//2 - 180, 50), 
                       cv2.FONT_HERSHEY_BOLD, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        elif event_type == 'no_person':
            # Grey overlay
            cv2.putText(frame, "No person detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2, cv2.LINE_AA)
    
    def run(self):
        """Run live demo loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame from camera")
                    break
                
                # Process frame
                event = self.pipeline.process_frame(frame)
                
                # Draw visualization
                display_frame = frame.copy()
                
                # Draw person bounding box
                if event['person_bbox']:
                    if event['event_type'] == 'fall':
                        bbox_color = (0, 0, 255)  # Red for fall
                    elif event['event_type'] == 'seizure':
                        bbox_color = (0, 165, 255)  # Orange for seizure
                    else:
                        bbox_color = (0, 255, 0)  # Green for normal
                    self.draw_bbox(display_frame, event['person_bbox'], bbox_color, 3)
                
                # Draw probability bars
                h, w = display_frame.shape[:2]
                bar_x = w - 220
                bar_y = 20
                
                self.draw_probability_bar(
                    display_frame, 
                    "Fall (Raw)", 
                    event['fall_confidence'], 
                    bar_x, bar_y
                )
                
                self.draw_probability_bar(
                    display_frame, 
                    "Fall (Smoothed)", 
                    event['fall_smoothed'], 
                    bar_x, bar_y + 40
                )
                
                # Future: Seizure probability bar
                if event['seizure_confidence'] > 0:
                    self.draw_probability_bar(
                        display_frame, 
                        "Seizure", 
                        event['seizure_smoothed'], 
                        bar_x, bar_y + 80
                    )
                
                # Draw alert overlay
                self.draw_alert(display_frame, event['event_type'])
                
                # Draw info text
                info_text = [
                    f"Event: {event['event_type']}",
                    f"FPS: {self.cap.get(cv2.CAP_PROP_FPS):.0f}",
                    "Press 'q' to quit, 'r' to reset"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(display_frame, text, (10, h - 70 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show frame
                cv2.imshow('Fall Detection Demo', display_frame)
                
                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):
                    self.pipeline.reset()
                    print("Pipeline reset")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\nDemo ended")


def main():
    parser = argparse.ArgumentParser(description='Live fall detection demo')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    args = parser.parse_args()
    
    try:
        demo = LiveDemo(args.config, args.camera)
        demo.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
