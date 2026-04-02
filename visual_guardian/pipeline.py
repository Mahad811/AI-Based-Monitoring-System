"""
Vision Pipeline Module

Main orchestrator that integrates all vision components for real-time monitoring.
Processes incoming video frames and generates structured event dictionaries.
"""

from datetime import datetime
from pathlib import Path
from .person_detector import PersonDetector
from .temporal_encoder import TemporalEncoder
from .fall_classifier import FallClassifier
from .smoother import SlidingWindowSmoother
import cv2
import numpy as np
from collections import deque


class VisionPipeline:
    """
    Complete vision pipeline for fall and seizure detection.
    
    Architecture:
        Camera → [TemporalEncoder → FallClassifier] + [SeizureClassifier] → Smoothers → Events
    
    PersonDetector is shared by both fall and seizure classifiers for consistent bbox cropping.
    """
    
    def __init__(self, config):
        """
        Args:
            config: dict with vision configuration (from config.yaml)
        """
        self.config = config
        
        # Initialize components
        print("Initializing Vision Pipeline...")
        
        # Shared person detector
        self.person_detector = PersonDetector(
            model_path=config['person_detector']['model'],
            confidence=config['person_detector']['confidence'],
            device=config['person_detector'].get('device', 'cpu')
        )
        print("✓ Person detector loaded")
        
        # Fall detection branch (optional - skip if model path missing)
        self.temporal_encoder = TemporalEncoder(
            buffer_size=config['temporal_encoder']['buffer_size'],
            frame_size=config['temporal_encoder']['frame_size']
        )
        
        self.fall_classifier = None
        self.fall_smoother = None
        self.fall_threshold = 0.6
        
        # Bed-exit detection (optional - for filtering in-bed patients)
        self.pose_analyzer = None
        self.bed_exit_enabled = False
        if 'bed_exit' in config and config['bed_exit'].get('enabled', False):
            try:
                from .pose_analyzer import PoseAnalyzer
                # PoseAnalyzer expects config dict with 'vision' key
                pose_config = {'vision': config}
                self.pose_analyzer = PoseAnalyzer(pose_config)
                self.bed_exit_enabled = True
                print("✓ Bed-exit detection enabled")
            except Exception as e:
                print(f"⚠ Bed-exit detection disabled: {e}")
        
        if 'fall_classifier' in config:
            try:
                model_path = Path(config['fall_classifier']['model'])
                if model_path.exists():
                    self.fall_classifier = FallClassifier(
                        model_path=config['fall_classifier']['model'],
                        device='auto'
                    )
                    self.fall_smoother = SlidingWindowSmoother(
                        window_size=config['fall_classifier']['window_size']
                    )
                    self.fall_threshold = config['fall_classifier']['threshold']
                    print("✓ Fall classifier loaded")
                else:
                    print("⚠ Fall classifier skipped (model path not found)")
            except Exception as e:
                print(f"⚠ Fall classifier skipped: {e}")
        
        # Seizure detection branch (optional - only if model weights exist)
        self.seizure_classifier = None
        self.seizure_smoother = None
        self.seizure_threshold = 0.6
        self.seizure_frame_counter = 0
        self.seizure_stride_frames = 15  # Classify every 0.5 seconds @ 30fps
        
        if 'seizure_classifier' in config:
            try:
                from .seizure_classifier import SeizureClassifier
                
                # Convert time-based config to frames
                window_frames = int(config['seizure_classifier']['window_seconds'] * 30)  # Assume 30fps
                stride_frames = int(config['seizure_classifier']['stride_seconds'] * 30)
                
                self.seizure_classifier = SeizureClassifier(
                    model_path=config['seizure_classifier']['model'],
                    window_frames=window_frames,
                    device='auto'
                )
                
                self.seizure_smoother = SlidingWindowSmoother(
                    window_size=config['seizure_classifier']['window_size']
                )
                
                self.seizure_threshold = config['seizure_classifier']['threshold']
                self.seizure_stride_frames = stride_frames
                
                print("✓ Seizure classifier loaded")
            except Exception as e:
                import traceback
                print(f"⚠ Seizure classifier not loaded: {e}")
                traceback.print_exc()
                print("  (Fall detection will still work)")
        
        # State Machine Tracking
        self.patient_state = 'IN_BED' # IN_BED, EXITING, OUT_OF_BED, FALLEN
        self.state_timer = 0.0
        self.bed_exit_cooldown = config.get('bed_exit', {}).get('cooldown', 5.0)
        self.last_process_time = datetime.now()
        
        # Robustness Tracking
        self.last_person_seen_time = datetime.now()
        self.darkness_threshold = config.get('robustness', {}).get('darkness_threshold', 30.0)
        self.inactivity_timeout = config.get('robustness', {}).get('inactivity_timeout_sec', 600)
        
        # Vision 3.0: Sleep Monitor
        self.restlessness_timer = 0.0
        self.restlessness_trigger_sec = 5.0 # Need 5s of thrashing to alert
        
        # Seizure 2.0: Pose History Buffer for Rhythm Analysis
        self.pose_history = deque(maxlen=60) # 2 seconds history
        
        print("✓ Vision Pipeline ready\n")
    
    def process_frame(self, frame):
        """
        Process a single video frame and return event detection results
        
        Args:
            frame: numpy array (H, W, 3) in BGR format
            
        Returns:
            dict with keys:
                - 'source': 'vision'
                - 'event_type': 'fall' | 'seizure' | 'normal' | 'no_person'
                - 'fall_confidence': float (0-1)
                - 'seizure_confidence': float (0-1, default 0.0 until seizure added)
                - 'fall_smoothed': float (0-1)
                - 'seizure_smoothed': float (0-1, default 0.0)
                - 'person_bbox': [x1, y1, x2, y2] or None
                - 'timestamp': ISO format string
        """
        timestamp = datetime.now().isoformat()
        dt = (datetime.now() - self.last_process_time).total_seconds()
        self.last_process_time = datetime.now()
        
        # 0. Environmental Check: Darkness
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < self.darkness_threshold:
            return {
                'source': 'vision',
                'event_type': 'darkness',
                'timestamp': timestamp,
                'debug_info': f'brightness={brightness:.1f} < {self.darkness_threshold}'
            }

        # Update temporal encoder buffer
        self.temporal_encoder.update(frame)
        
        # Initialize event dict with defaults
        event = {
            'source': 'vision',
            'event_type': 'normal',
            'fall_confidence': 0.0,
            'seizure_confidence': 0.0,
            'fall_smoothed': 0.0,
            'seizure_smoothed': 0.0,
            'person_bbox': None,
            'timestamp': timestamp,
            'state': self.patient_state,  # Add state to output
            'debug_info': ''
        }
        # ── 1. Centralized YOLO Person Detection ──
        # Run exactly ONCE per frame. Re-used by Fall, Seizure, and Pose components 
        # to prevent redundant CPU workloads causing FPS drops.
        detection = self.person_detector.detect(frame, padding=0.2)
        if detection:
            event['person_bbox'] = list(detection['bbox'])

        # Fall detection (only if fall classifier loaded)
        if self.fall_classifier is not None and self.temporal_encoder.is_ready():
            # Bed-exit filter AND Context Manager
            
            skip_fall_detection = False
            current_threshold = self.fall_threshold
            
            if self.pose_analyzer is not None:
                # Inactivity / Missing Patient Check
                if detection:
                    self.last_person_seen_time = datetime.now()
                else:
                    time_missing = (datetime.now() - self.last_person_seen_time).total_seconds()
                    if time_missing > self.inactivity_timeout:
                         event['event_type'] = 'missing_patient'
                         event['debug_info'] = f'Missing for {time_missing:.0f}s'
                         return event

                if detection:
                    pose_data = self.pose_analyzer.analyze_pose(frame)
                    if pose_data is not None:
                        # Append to history for Seizure 2.0
                        self.pose_history.append(pose_data)
                        
                        # Expose landmarks for Demo/Visualization
                        event['landmarks'] = pose_data['landmarks_norm']
                        
                        # A. Check Safety Net (Fallen State) - Priority 1
                        safety_res = self.pose_analyzer.check_fallen_state(pose_data, frame.shape[0])
                        if safety_res['fallen']:
                            event['event_type'] = 'force_fall' # Special flag for safety net
                            event['debug_info'] = 'Safety Net Triggered'
                            self.patient_state = 'FALLEN'
                            # We can return early or force the probability high
                            event['fall_confidence'] = 1.0
                            event['fall_smoothed'] = 1.0
                            event['person_bbox'] = list(detection['bbox'])
                            return event

                        # B. Check Bed Exit - Priority 2
                        bed_exit_result = self.pose_analyzer.check_bed_exit(
                            pose_data,
                            frame.shape[0],
                            boundary_margin_px=self.config.get('bed_exit', {}).get('boundary_margin_px', 50),
                            min_cross_frames=self.config.get('bed_exit', {}).get('min_frames', 15)
                        )
                        
                        # State Transitions
                        if bed_exit_result.get('bed_exit', False):
                            self.patient_state = 'EXITING'
                            self.state_timer = self.bed_exit_cooldown # Reset cooldown
                        
                        # Handle State Logic
                        if self.patient_state == 'EXITING':
                            # High Sensitivity Mode
                            current_threshold = 0.4 # LOWER threshold to catch falls
                            event['debug_info'] = 'High Sensitivity (Bed Exit)'
                            self.state_timer -= dt
                            if self.state_timer <= 0:
                                self.patient_state = 'OUT_OF_BED'
                                
                        elif self.patient_state == 'IN_BED':
                            # In Bed Mode -> Skip falls or Very High Threshold
                            skip_fall_detection = True
                            event['event_type'] = 'in_bed'
                            event['person_bbox'] = list(detection['bbox'])
                            
                            # Vision 3.0: Sleep Restlessness Monitor
                            restless_res = self.pose_analyzer.check_sleep_restlessness(list(self.pose_history))
                            if restless_res['restless']:
                                self.restlessness_timer += dt
                                event['debug_info'] = f"Restless: {self.restlessness_timer:.1f}s (Energy: {restless_res['energy']:.4f})"
                                if self.restlessness_timer >= self.restlessness_trigger_sec:
                                    event['event_type'] = 'restlessness'
                                    # Don't Auto-reset immediately, let it stream 'restlessness' while happening?
                                    # Or trigger once? Let's stream it but maybe cap timer to avoid overflow if needed.
                            else:
                                # Cooldown / Reset if patient is still
                                self.restlessness_timer = max(0, self.restlessness_timer - dt)
                            
                        # If just sitting up (not exiting yet), we might still be 'IN_BED' but 
                        # bed_exit_result returns False. We rely on the sticky 'EXITING' state.
                        
            # Only run fall detection if we are NOT securely in bed
            if not skip_fall_detection:
                # Encode temporal RGB using the centralized detection
                temporal_rgb = self.temporal_encoder.encode(detection=detection)
                
                if temporal_rgb is not None:
                    # Classify
                    fall_result = self.fall_classifier.classify(temporal_rgb)
                    
                    if fall_result is not None:
                        event['fall_confidence'] = fall_result['fall_prob']
                        
                        # Smooth probability
                        if self.fall_smoother is not None:
                            event['fall_smoothed'] = self.fall_smoother.update(fall_result['fall_prob'])
                        else:
                            event['fall_smoothed'] = fall_result['fall_prob']
                        
                        # Determine event type based on RAW probability (for high recall)
                        # Smoothed prob is used for visualization stability only.
                        if event['fall_confidence'] >= current_threshold:
                            event['event_type'] = 'fall'
                            self.patient_state = 'FALLEN' # Latch state
                        else:
                            event['event_type'] = 'normal'
                    else:
                        event['event_type'] = 'no_person'
                else:
                    event['event_type'] = 'no_person'
        
        # Seizure detection (runs at lower frequency than fall detection)
        if self.seizure_classifier is not None:
            # Update seizure buffer every frame
            self.seizure_classifier.update(frame)
            
            # Classify only at specified stride (e.g., every 0.5 seconds)
            self.seizure_frame_counter += 1
            if self.seizure_frame_counter >= self.seizure_stride_frames:
                self.seizure_frame_counter = 0
                
                if self.seizure_classifier.is_ready():
                    # Classify current window
                    seizure_result = self.seizure_classifier.classify(detection=detection)
                    
                    if seizure_result is not None:
                        event['seizure_confidence'] = seizure_result['seizure_prob']
                        
                        # Smooth probability
                        event['seizure_smoothed'] = self.seizure_smoother.update(
                            seizure_result['seizure_prob']
                        )
                        
                        # Update event type only if not already fall (fall takes priority)
                        # USE RAW CONFIDENCE for trigger (matches offline evaluation max-aggregation)
                        if event['seizure_confidence'] >= self.seizure_threshold:
                            if event['event_type'] != 'fall':
                                # Seizure 2.0: Rhythm Verification
                                is_confirmed = True
                                if self.pose_analyzer is not None:
                                    rhythm_res = self.pose_analyzer.check_seizure_rhythm(list(self.pose_history))
                                    if not rhythm_res['confirmed']:
                                        # Suppression Logic: If rhythm is NOT chaotic, it's likely false positive
                                        # We downgrade the confidence or tag it
                                        event['debug_info'] = f"Seizure Suppressed (Rhythm Score: {rhythm_res.get('score',0):.2f})"
                                        event['seizure_confidence'] *= 0.5 # Penalty
                                        event['seizure_smoothed'] *= 0.5   # Penalty
                                        if event['seizure_confidence'] < self.seizure_threshold:
                                            is_confirmed = False
                                    else:
                                        event['debug_info'] = f"Seizure Confirmed (Rhythm: {rhythm_res['score']:.2f})"

                                if is_confirmed:
                                    event['event_type'] = 'seizure'
        
        return event
    
    def reset(self):
        """Reset all buffers, smoothers, and stateful detection history."""
        self.temporal_encoder.reset()
        if self.fall_smoother is not None:
            self.fall_smoother.reset()
        if self.seizure_classifier is not None:
            self.seizure_classifier.reset()
        if self.seizure_smoother is not None:
            self.seizure_smoother.reset()
        self.seizure_frame_counter = 0

        # Reset person detector cache so the first frames of the next segment
        # are not cropped using a stale bounding box from the previous clip.
        self.person_detector.reset()

        # Reset all stateful fields that persist between segments
        self.pose_history.clear()
        self.restlessness_timer       = 0.0
        self.last_person_seen_time    = datetime.now()
