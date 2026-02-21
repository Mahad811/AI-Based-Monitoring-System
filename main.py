"""
Vital Guardian - Main Integration Script
Integrates Vision, Audio, and Cognitive Core for real-time monitoring
"""

import os
from dotenv import load_dotenv
load_dotenv()

import cv2
import yaml
import numpy as np
import time
from collections import deque
import queue
from concurrent.futures import ThreadPoolExecutor

from visual_guardian.pipeline import VisionPipeline
from auditory_watchdog.core.audio_capture import AudioStream
from auditory_watchdog.core.privacy_shield import PrivacyShield
from auditory_watchdog.core.distress_classifier import DistressClassifier
from auditory_watchdog.core.keyword_spotter import KeywordSpotter
from cognitive_core.models import AudioEvent, VisionEvent
from cognitive_core.core import CognitiveCore


class VitalGuardian:
    """Main system integrating all monitoring modules"""
    
    def __init__(self, config_path='config/config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize modules
        print("Initializing Visual Guardian...")
        self.vision_pipeline = VisionPipeline(self.config.get('vision', {}))
        
        print("Initializing Auditory Watchdog...")
        self.audio_stream = AudioStream()
        self.privacy_shield = PrivacyShield()
        self.distress_classifier = DistressClassifier()
        self.keyword_spotter = KeywordSpotter()
        self.audio_executor = ThreadPoolExecutor(max_workers=2)
        
        print("Initializing Cognitive Core...")
        self.cognitive_core = CognitiveCore(self.config)
        
        # Thread-safe queue for async audio events
        self.audio_queue = queue.Queue()
        
        # Performance tracking
        self.performance_config = self.config.get('performance', {'target_fps': 30, 'max_latency_ms': 100})
        self.frame_times = deque(maxlen=30)

    def _handle_distress_result(self, future):
        try:
            result = future.result()
            if result.get("event_detected"):
                event = AudioEvent(
                    event_type="distress",
                    sound_type=result.get("primary_sound"),
                    confidence=result.get("details", [{}])[0].get("confidence", 1.0)
                )
                self.audio_queue.put(event)
        except Exception as e:
            print(f"Error handling distress async: {e}")

    def _handle_kws_result(self, future):
        try:
            result = future.result()
            if result.get("event_detected"):
                event = AudioEvent(
                    event_type="keyword",
                    keyword=result.get("text"),
                    language=result.get("language")
                )
                self.audio_queue.put(event)
        except Exception as e:
            print(f"Error handling kws async: {e}")
    
    def run_video_monitoring(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return
        
        print("Starting video and async audio monitoring...")
        self.audio_stream.start_stream()
        
        try:
            while True:
                start_time = time.time()
                
                # 1. Non-blocking audio chunk polling
                chunk_data = self.audio_stream.get_latest_chunk(timeout=0.0)
                if chunk_data is not None:
                    chunk, new_stride = chunk_data
                    should_analyze, speech_clip = self.privacy_shield.analyze_chunk(new_stride)
                    
                    if should_analyze:
                        future_distress = self.audio_executor.submit(self.distress_classifier.analyze_chunk, chunk)
                        future_distress.add_done_callback(self._handle_distress_result)
                        
                        if speech_clip is not None:
                            future_kws = self.audio_executor.submit(self.keyword_spotter.analyze_chunk, speech_clip)
                            future_kws.add_done_callback(self._handle_kws_result)

                # 2. Camera Frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 3. Vision Detection
                vision_dict = self.vision_pipeline.process_frame(frame)
                vision_event = VisionEvent.from_pipeline_event(vision_dict)
                
                # 4. Audio Events Pop
                audio_event = None
                try:
                    audio_event = self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # 5. Cognitive Core Fusion
                # Passes everything to the core, which returns a ReflexAlert if triggered.
                alert = self.cognitive_core.process(vision_event, audio_event, frame)
                if alert:
                    print(f"\n🚨 ALERT: {alert.reflex.level} | {alert.reflex.message}")
                
                # Render basic UI bounding box if person detected
                if vision_event.person_bbox:
                    bbox = vision_event.person_bbox
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    
                cv2.putText(frame, f"State: {vision_event.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Track performance
                processing_time = (time.time() - start_time) * 1000
                self.frame_times.append(processing_time)
                
                # Display frame
                cv2.imshow('Vital Guardian Master Module', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.audio_stream.stop_stream()
            self.audio_stream.terminate()
            self.audio_executor.shutdown(wait=False)
            self.cognitive_core.shutdown()
            cap.release()
            cv2.destroyAllWindows()
    
    def get_performance_stats(self):
        if len(self.frame_times) == 0:
            return None
        
        avg_time = np.mean(self.frame_times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_processing_time_ms': round(avg_time, 2),
            'fps': round(fps, 2),
            'target_fps': self.performance_config.get('target_fps', 30),
            'max_latency_ms': self.performance_config.get('max_latency_ms', 100),
            'meets_latency_target': avg_time < self.performance_config.get('max_latency_ms', 100)
        }

def main():
    print("="*60)
    print("VITAL GUARDIAN - AI Patient Monitoring System")
    print("="*60)
    
    system = VitalGuardian()
    print("\nPress 'q' inside the video window to quit\n")
    system.run_video_monitoring(video_source=0)
    
    stats = system.get_performance_stats()
    if stats:
        print("\n" + "="*60)
        print("Performance Statistics:")
        print(f"  Average Processing Time: {stats['avg_processing_time_ms']} ms")
        print(f"  FPS: {stats['fps']}")
        print(f"  Latency Target Met: {stats['meets_latency_target']}")
        print("="*60)

if __name__ == '__main__':
    main()
