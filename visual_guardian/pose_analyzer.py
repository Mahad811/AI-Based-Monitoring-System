"""
Pose Analyzer using MediaPipe
Analyzes patient posture and movements (CPU-friendly)
"""

import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime


class PoseAnalyzer:
    """Analyzes patient pose and detects dangerous movements"""
    
    def __init__(self, config):
        """
        Initialize pose analyzer
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config['vision']
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        # Motion tracking
        self.prev_landmarks = None
        self.prev_velocity = None
        self.velocity_window = []
        self.jerk_window = []
        
    def analyze_pose(self, frame):
        """
        Analyze patient pose in frame
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            dict: Pose analysis results
        """
        # Optional downscale to 480p for CPU
        h, w = frame.shape[:2]
        target_h = 480
        scale = target_h / float(h) if h > target_h else 1.0
        if scale < 1.0:
            frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame_small = frame

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
            
        landmarks = results.pose_landmarks.landmark
        # Normalize to original frame space ratios
        norm_points = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)

        # Motion metrics
        velocity = self._compute_average_velocity(norm_points)
        jerk = self._compute_average_jerk(velocity)

        # Pose state (simple rules using shoulders/hips and nose)
        state = self._estimate_pose_state(norm_points)

        return {
            'detected': True,
            'timestamp': datetime.utcnow().isoformat(),
            'landmarks_norm': norm_points,  # normalized [0..1]
            'pose_state': state,            # 'lying' | 'sitting' | 'standing' | 'unknown'
            'motion': {
                'avg_velocity': float(velocity) if velocity is not None else 0.0,
                'avg_jerk': float(jerk) if jerk is not None else 0.0,
            }
        }
    
    def detect_dangerous_movement(self, pose_data):
        """
        Detect dangerous movements like seizures or erratic motion
        
        Args:
            pose_data: Pose data from analyze_pose
            
        Returns:
            dict: Movement analysis
        """
        if pose_data is None:
            return {'dangerous': False, 'reason': 'No pose detected'}

        angle_threshold = self.config['fall_detection']['angle_threshold']
        velocity_threshold = self.config['unusual_movement']['velocity_threshold']
        jerk_threshold = self.config['unusual_movement']['jerk_threshold']

        # Torso angle using shoulders/hips from normalized landmarks
        lm = pose_data['landmarks_norm']
        left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        hip_mid = (left_hip + right_hip) / 2.0
        angle = np.degrees(np.arctan2(hip_mid[1] - shoulder_mid[1], hip_mid[0] - shoulder_mid[0]))

        motion = pose_data['motion']
        velocity = motion['avg_velocity']
        jerk = motion['avg_jerk']

        if abs(angle) > angle_threshold:
            return {'dangerous': True, 'reason': 'Abnormal body position', 'angle': float(angle), 'severity': 'medium'}

        if velocity >= velocity_threshold or jerk >= jerk_threshold:
            return {'dangerous': True, 'reason': 'Erratic movement', 'velocity': float(velocity), 'jerk': float(jerk), 'severity': 'high'}

        return {'dangerous': False, 'reason': 'Normal posture/motion'}

    def check_bed_exit(self, pose_data, frame_height, boundary_margin_px=40, min_cross_frames=10):
        """Detect bed-exit: hips below bed boundary for N consecutive frames."""
        if pose_data is None:
            return {'bed_exit': False, 'reason': 'No pose detected'}

        # Config defaults
        margin = self.config.get('bed_exit', {}).get('boundary_margin_px', boundary_margin_px)
        min_frames = self.config.get('bed_exit', {}).get('min_cross_frames', min_cross_frames)

        boundary_norm = 1.0 - (margin / max(frame_height, 1))  # bottom margin -> normalized y

        lm = pose_data['landmarks_norm']
        left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP][1]
        right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP][1]
        hips_below = (left_hip > boundary_norm) and (right_hip > boundary_norm)

        # Maintain a small counter
        if not hasattr(self, '_bed_cross_count'):
            self._bed_cross_count = 0

        self._bed_cross_count = self._bed_cross_count + 1 if hips_below else 0

        if self._bed_cross_count >= min_frames:
            return {'bed_exit': True, 'reason': 'Hips crossed boundary', 'frames': int(self._bed_cross_count)}

        return {'bed_exit': False, 'reason': 'Within boundary', 'frames': int(self._bed_cross_count)}

    def check_fallen_state(self, pose_data, frame_height):
        """
        Safety Net: Detect if person is in a 'Fallen State' (Lying on floor).
        Logic: Torso is horizontal AND Center of Mass is in the 'Floor Zone'.
        """
        if pose_data is None:
            return {'fallen': False, 'reason': 'No pose detected'}

        # Config defaults
        margin = self.config.get('safety_net', {}).get('floor_margin_px', 50)
        angle_thresh = self.config.get('safety_net', {}).get('torso_angle_threshold', 45)
        
        # 1. Check Torso Angle (Horizontal?)
        state = pose_data['pose_state'] # 'lying', 'sitting', 'standing'
        # We re-calculate precise angle if needed, but 'lying' is a good proxy from _estimate_pose_state
        is_horizontal = (state == 'lying')

        # 2. Check Location (On Floor?)
        lm = pose_data['landmarks_norm']
        # Use hips as center of mass proxy
        left_hip_y = lm[self.mp_pose.PoseLandmark.LEFT_HIP][1]
        right_hip_y = lm[self.mp_pose.PoseLandmark.RIGHT_HIP][1]
        hip_y = (left_hip_y + right_hip_y) / 2.0
        
        # Floor boundary: Normalized Y > (1.0 - margin_fraction)
        # Higher Y = Lower in image
        floor_y_norm = 1.0 - (margin / max(frame_height, 1))
        
        on_floor = (hip_y > floor_y_norm)


        if is_horizontal and on_floor:
            return {'fallen': True, 'reason': 'Horizontal on floor', 'confidence': 1.0}
        
        return {'fallen': False, 'reason': f'State: {state}, OnFloor: {on_floor}', 'confidence': 0.0}

    def check_seizure_rhythm(self, pose_history, fps=30):
        """
        Seizure 2.0: Verify seizure using Motion Rhythm Analysis.
        
        Args:
            pose_history: List of pose_data dicts from previous frames.
            fps: Frames per second (default 30).
            
        Returns:
            dict: {'confirmed': bool, 'score': float, 'reason': str}
        """
        if not pose_history or len(pose_history) < 15: # Need at least 0.5s
            return {'confirmed': True, 'score': 0.5, 'reason': 'Insufficient history — pass through'}

        # SUPPRESSION ONLY: Only suppress if patient is completely motionless.
        # We do NOT require rhythmic motion — seizures can be subtle.
        # Threshold: near-zero variance AND near-zero mean displacement.
        # This catches false alarms where the model fires on a still patient.
        suppression_var_threshold  = 0.0001   # Extremely low variance = no movement at all
        suppression_mean_threshold = 0.003    # Extremely low mean displacement = still patient
        
        left_wrist_diffs = []
        right_wrist_diffs = []
        
        for i in range(1, len(pose_history)):
            curr = pose_history[i]['landmarks_norm']
            prev = pose_history[i-1]['landmarks_norm']
            
            # Skip if landmarks missing
            if curr is None or prev is None:
                continue
            
            try:
                # Calculate displacement magnitude
                lw_d = np.sqrt((curr[15][0]-prev[15][0])**2 + (curr[15][1]-prev[15][1])**2) # LEFT_WRIST=15
                rw_d = np.sqrt((curr[16][0]-prev[16][0])**2 + (curr[16][1]-prev[16][1])**2) # RIGHT_WRIST=16
                left_wrist_diffs.append(lw_d)
                right_wrist_diffs.append(rw_d)
            except (IndexError, TypeError):
                continue
            
        if not left_wrist_diffs:
             return {'confirmed': True, 'score': 0.5, 'reason': 'No wrist data — pass through'}

        lw_var  = np.var(left_wrist_diffs)
        rw_var  = np.var(right_wrist_diffs)
        lw_mean = np.mean(left_wrist_diffs)
        rw_mean = np.mean(right_wrist_diffs)
        
        max_var  = max(lw_var, rw_var)
        max_mean = max(lw_mean, rw_mean)
        
        # Suppress ONLY if patient is completely still (no movement at all)
        is_completely_still = (max_var < suppression_var_threshold and 
                               max_mean < suppression_mean_threshold)
        
        # Score: 0 = no movement (suppress), 1 = lots of movement (confirm)
        score = min(max_mean / suppression_mean_threshold, 1.0)
        
        if is_completely_still:
            return {
                'confirmed': False, 
                'score': float(score), 
                'reason': f'Patient still: MeanDisp={max_mean:.5f}, Var={max_var:.6f}'
            }
        
        return {
            'confirmed': True, 
            'score': float(score), 
            'reason': f'Movement detected: MeanDisp={max_mean:.5f}, Var={max_var:.6f}'
        }

    def check_sleep_restlessness(self, pose_history):
        """
        Vision 3.0: Digital Actigraphy (Sleep Restlessness).
        Measures 'Gross Body Movement' (Torso) to detect tossing/turning.
        
        Args:
            pose_history: List of pose_data.
            
        Returns:
            dict: {'restless': bool, 'energy': float}
        """
        if not pose_history or len(pose_history) < 15:
            return {'restless': False, 'energy': 0.0}

        # Threshold for "Significant Movement" (e.g., turning over)
        # 0.005 is approx 0.5% of screen per frame (avg).
        # Tossing/turning generates high consistent displacement.
        actigraphy_threshold = 0.005
        
        torso_diffs = []
        
        for i in range(1, len(pose_history)):
            curr = pose_history[i]['landmarks_norm']
            prev = pose_history[i-1]['landmarks_norm']
            
            if not curr or not prev:
                continue
            
            # Torso Keypoints: Shoulders(11,12) + Hips(23,24)
            # We treat them as a "Core Block"
            indices = [11, 12, 23, 24]
            frame_disp = 0.0
            valid_pts = 0
            
            for idx in indices:
                # Check bounds (Mediapipe pose has 33 landmarks)
                if idx in curr and idx in prev: # Indices are keys in dist? No, landmarks_norm can be list or dict? 
                    # Wrapper returns dict usually? 
                    # Let's check how landmarks_norm was stored. 
                    # It's usually a dict: {0: (x,y), ...} or list [(x,y), ...].
                    # Let's assume list or dict, safe access.
                    try:
                       c = curr[idx]
                       p = prev[idx]
                       dist = np.sqrt((c[0]-p[0])**2 + (c[1]-p[1])**2)
                       frame_disp += dist
                       valid_pts += 1
                    except:
                       pass
                       
            if valid_pts > 0:
                torso_diffs.append(frame_disp / valid_pts)
                
        if not torso_diffs:
            return {'restless': False, 'energy': 0.0}
            
        avg_energy = np.mean(torso_diffs)
        
        return {
            'restless': (avg_energy > actigraphy_threshold),
            'energy': float(avg_energy)
        }
    
    def __del__(self):
        """Cleanup resources"""
        self.pose.close()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _compute_average_velocity(self, norm_points: np.ndarray):
        window = self.config.get('unusual_movement', {}).get('window_size', 5)
        if self.prev_landmarks is None:
            self.prev_landmarks = norm_points
            return 0.0
        delta = norm_points - self.prev_landmarks
        self.prev_landmarks = norm_points
        # L2 per-point, then average
        per_point_speed = np.linalg.norm(delta, axis=1)
        avg_speed = float(np.mean(per_point_speed))
        self.velocity_window.append(avg_speed)
        if len(self.velocity_window) > window:
            self.velocity_window.pop(0)
        # Simple smoothed velocity
        return float(np.mean(self.velocity_window))

    def _compute_average_jerk(self, smoothed_velocity: float):
        window = self.config.get('unusual_movement', {}).get('window_size', 5)
        if smoothed_velocity is None:
            return 0.0
        if self.prev_velocity is None:
            self.prev_velocity = smoothed_velocity
            return 0.0
        jerk = abs(smoothed_velocity - self.prev_velocity)
        self.prev_velocity = smoothed_velocity
        self.jerk_window.append(jerk)
        if len(self.jerk_window) > window:
            self.jerk_window.pop(0)
        return float(np.mean(self.jerk_window))

    def _estimate_pose_state(self, norm_points: np.ndarray):
        try:
            ls = norm_points[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = norm_points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lh = norm_points[self.mp_pose.PoseLandmark.LEFT_HIP]
            rh = norm_points[self.mp_pose.PoseLandmark.RIGHT_HIP]
            nose = norm_points[self.mp_pose.PoseLandmark.NOSE]
        except Exception:
            return 'unknown'

        shoulder_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0
        torso_vec = hip_mid - shoulder_mid
        angle = abs(np.degrees(np.arctan2(torso_vec[1], torso_vec[0])))

        # Heuristic: vertical torso -> standing/sitting; horizontal -> lying
        if angle < 25:
            return 'standing'
        if 25 <= angle <= 65:
            # Distinguish by nose height vs hips
            return 'sitting' if nose[1] < hip_mid[1] else 'unknown'
        return 'lying'

