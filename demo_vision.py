"""
Vital Guardian - Vision Module Version 3.0 DEMO
Run this script to demonstrate the system to supervisors.

Features Visualized:
1. State Machine (IN_BED -> EXITING -> FALLEN)
2. Seizure Detection (Rhythm Verification)
3. Sleep Restlessness (Digital Actigraphy)
4. Safety Net (Fallen Posture)
"""

import cv2
import numpy as np
import time
import yaml
import mediapipe as mp
from visual_guardian.pipeline import VisionPipeline

# Visualization Config
COLOR_OK = (0, 255, 0)       # Green
COLOR_WARN = (0, 165, 255)   # Orange
COLOR_ALERT = (0, 0, 255)    # Red
COLOR_TEXT = (255, 255, 255) # White

def draw_ui(frame, event, fps):
    h, w = frame.shape[:2]
    
    # 1. Overlay Skeleton (if landmarks present)
    if event.get('landmarks') is not None:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        # Convert norm landmarks back to proto for drawing util?
        # Actually easier to draw manually or construct proto keypoints.
        # For demo speed, let's just draw lines between key joints if available.
        # But we need pixel coords.
        landmarks = event['landmarks']
        
        # Draw Connections (Simplified Skeleton)
        connections = mp_pose.POSE_CONNECTIONS
        for conn in connections:
            start_idx = conn[0]
            end_idx = conn[1]
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_pt = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                end_pt = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                cv2.line(frame, start_pt, end_pt, (0, 255, 255), 2)
                
    # 2. Draw Bounding Box
    if event.get('person_bbox'):
        bbox = event['person_bbox']
        color = COLOR_ALERT if event['event_type'] in ['fall', 'seizure', 'force_fall'] else COLOR_OK
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

    # 3. Sidebar UI (Left Panel)
    panel_w = 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # 4. Status Text
    y = 40
    line_h = 30
    
    def put_text(text, color=COLOR_TEXT, scale=0.7):
        nonlocal y
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += line_h

    # Header
    put_text("VITAL GUARDIAN v3.0", (255, 255, 0), 0.8)
    y += 10
    
    # State
    state = event.get('state', 'UNKNOWN')
    state_color = COLOR_OK
    if state == 'EXITING': state_color = COLOR_WARN
    if state == 'FALLEN': state_color = COLOR_ALERT
    put_text(f"State: {state}", state_color)
    
    # Events
    etype = event['event_type']
    e_color = COLOR_TEXT
    if etype in ['fall', 'seizure', 'force_fall', 'restlessness']:
        e_color = COLOR_ALERT
        # Flash effect
        if int(time.time() * 5) % 2 == 0:
            cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 10)
            
    put_text(f"Event: {etype.upper()}", e_color)
    
    y += 10
    # Metrics
    put_text(f"Fall Conf: {event.get('fall_smoothed', 0.0):.2f}")
    
    # Seizure Info
    sz_conf = event.get('seizure_smoothed', 0.0)
    sz_color = COLOR_ALERT if sz_conf > 0.5 else COLOR_TEXT
    put_text(f"Seizure Conf: {sz_conf:.2f}", sz_color)
    
    # Debug Info (Rhythm/Restlessness)
    debug_info = event.get('debug_info', '')
    if debug_info:
        # Split debug info if too long
        words = debug_info.split()
        lines = []
        curr = ""
        for word in words:
            if len(curr + word) > 25:
                lines.append(curr)
                curr = word + " "
            else:
                curr += word + " "
        lines.append(curr)
        
        y += 10
        put_text("Debug:", (200, 200, 200), 0.5)
        for line in lines:
            put_text(line, (200, 200, 200), 0.5)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        root_config = yaml.safe_load(f)
        
    # Extract Vision Config
    vision_config = root_config['vision']
        
    # Force Enable Features for Demo
    vision_config['bed_exit']['enabled'] = True
    print("Initializing Pipeline...")
    pipeline = VisionPipeline(vision_config)
    
    cap = cv2.VideoCapture(0)
    # Set to reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\nStarting Demo. Press 'q' to exit.")
    print("Direct the camera at a bed/person.")
    
    prev_time = time.time()
    
    frame_count = 0
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read camera")
                break
                
            # Process
            event = pipeline.process_frame(frame)
            
            # FPS Calc
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time > 1.0:
                fps = frame_count / (curr_time - prev_time)
                frame_count = 0
                prev_time = curr_time
            
            # Draw
            vis_frame = draw_ui(frame, event, fps)
            
            cv2.imshow("Vital Guardian Demo", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
