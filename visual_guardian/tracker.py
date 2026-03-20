"""
Patient Tracker using YOLOv8 (+ optional MediaPipe Pose integration)
Detects and tracks patient position and flags bed-exit/unusual movement
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from .pose_analyzer import PoseAnalyzer


class PatientTracker:
    """Tracks patient using YOLOv8 object detection"""
    
    def __init__(self, config):
        """
        Initialize tracker
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config['vision']
        self.model = YOLO(self.config['model'])
        self.confidence_threshold = self.config['confidence_threshold']
        self.pose_enabled = bool(self.config.get('pose_detection', True))
        self.pose_analyzer = PoseAnalyzer(config) if self.pose_enabled else None
        
    def detect_patient(self, frame):
        """
        Detect patient in frame
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            dict: Detection results with bounding box and confidence
        """
        results = self.model(frame, conf=self.confidence_threshold, classes=[0])  # class 0 = person/person-like
        
        if len(results[0].boxes) == 0:
            return None
            
        # Get first detected person (assuming single patient)
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        
        return {
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(confidence),
            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        }
    
    def check_fall_risk(self, detection, frame_height):
        """
        Check if patient position indicates fall risk
        
        Args:
            detection: Detection dictionary from detect_patient
            frame_height: Height of video frame
            
        Returns:
            dict: Fall risk assessment
        """
        if detection is None:
            return {'fall_risk': False, 'reason': 'No patient detected'}
        
        bbox = detection['bbox']
        patient_height = bbox[3] - bbox[1]
        relative_height = patient_height / frame_height
        
        # Check if patient is too low (potential fall)
        fall_threshold = self.config['fall_detection']['height_threshold']
        
        if relative_height < fall_threshold:
            return {
                'fall_risk': True,
                'reason': 'Patient position abnormally low',
                'severity': 'high'
            }
        
        return {'fall_risk': False, 'reason': 'Normal position'}

    def process_frame(self, frame):
        """Run detection + optional pose analysis and return unified event dict."""
        h, w = frame.shape[:2]
        detection = self.detect_patient(frame)
        timestamp = datetime.utcnow().isoformat()

        if detection is None:
            return {
                'source': 'vision',
                'event_type': 'no_patient',
                'confidence': 0.0,
                'timestamp': timestamp,
                'metadata': {}
            }

        metadata = {'bbox': detection['bbox'], 'center': detection['center']}

        # Pose analysis (if enabled)
        pose_data = None
        bed_exit = {'bed_exit': False}
        dangerous = {'dangerous': False}
        if self.pose_enabled and self.pose_analyzer is not None:
            pose_data = self.pose_analyzer.analyze_pose(frame)
            if pose_data is not None:
                dangerous = self.pose_analyzer.detect_dangerous_movement(pose_data)
                bed_exit = self.pose_analyzer.check_bed_exit(
                    pose_data,
                    frame_height=h,
                    boundary_margin_px=self.config.get('bed_exit', {}).get('boundary_margin_px', 40),
                    min_cross_frames=self.config.get('bed_exit', {}).get('min_cross_frames', 10),
                )
                metadata.update({
                    'pose_state': pose_data.get('pose_state'),
                    'motion': pose_data.get('motion'),
                    'dangerous_movement': dangerous,
                    'bed_exit_check': bed_exit,
                })

        # Determine event type and confidence
        if bed_exit.get('bed_exit'):
            return {
                'source': 'vision',
                'event_type': 'bed_exit',
                'confidence': 0.9,
                'timestamp': timestamp,
                'metadata': metadata
            }

        if dangerous.get('dangerous'):
            return {
                'source': 'vision',
                'event_type': 'unusual_movement',
                'confidence': 0.8,
                'timestamp': timestamp,
                'metadata': metadata
            }

        fall_risk = self.check_fall_risk(detection, h)
        metadata['fall_risk'] = fall_risk

        return {
            'source': 'vision',
            'event_type': 'patient_detected',
            'confidence': float(detection['confidence']),
            'timestamp': timestamp,
            'metadata': metadata
        }

