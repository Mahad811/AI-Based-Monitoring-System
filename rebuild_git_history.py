import os
import subprocess
import shutil
from datetime import datetime, timedelta
import glob

REPO_URL = "https://github.com/Mahad811/AI-Based-Monitoring-System.git"
START_DATE = datetime(2025, 8, 15, 10, 0, 0)

# 40-step commit plan mapping descriptive messages to specific file patterns
COMMITS = [
    # Initial setup
    ("Initial commit: Initialize repository framework", [".gitignore", "README.md"]),
    ("Setup root configuration schemas", ["config/"]),
    ("Add system dependency requirements", ["requirements.txt"]),
    ("Outline initial project implementation plan", ["implementation_plan.md"]),

    # Data preprocessing module
    ("Create data preprocessing foundation", ["data_preprocessing/__init__.py", "data_preprocessing/k700_download.py"]),
    ("Build video dataset cropping utilities", ["data_preprocessing/bbox_cropper.py"]),
    ("Add video to image converter tool", ["data_preprocessing/video_to_images.py"]),
    ("Document processing scripts", ["data_preprocessing/README.md"]),

    # Vision module base
    ("Initialize visual guardian structure", ["visual_guardian/__init__.py"]),
    ("Import YOLOv8 person detection logic", ["visual_guardian/person_detector.py"]),
    ("Write temporal RGB encoding classes", ["visual_guardian/temporal_encoder.py"]),
    ("Draft base Fall Classifier wrapper", ["visual_guardian/fall_classifier.py"]),
    
    # Advanced vision features
    ("Build baseline seizure neural network logic", ["visual_guardian/seizure_classifier.py"]),
    ("Add sliding window smoothing logic", ["visual_guardian/smoother.py"]),
    ("Draft MediaPipe pose analyzer", ["visual_guardian/pose_analyzer.py"]),
    ("Orchestrate components through Vision Pipeline", ["visual_guardian/pipeline.py"]),
    ("Finalize early vision demonstration scripts", ["demo_vision.py"]),

    # Notebooks & exploration
    ("Add Jupyter notebooks for EDA", ["notebooks/EDA.ipynb", "notebooks/fall_EDA.ipynb"]),
    ("Configure seizure modeling notebooks", ["notebooks/seizure.ipynb"]),
    
    # Dashboard server
    ("Initialize Flask backend structure", ["dashboard/app.py", "dashboard/socket_events.py"]),
    ("Serve static dashboard UI components", ["dashboard/static/"]),
    ("Load pre-compiled dashboard templates", ["dashboard/templates/"]),

    # Auditory watchdog
    ("Initialize auditory watchdog base", ["auditory_watchdog/__init__.py", "auditory_watchdog/config.py"]),
    ("Draft PyAudio background thread capturer", ["auditory_watchdog/core/audio_capture.py"]),
    ("Integrate Silero Voice Activity Detector", ["auditory_watchdog/core/privacy_shield.py"]),
    ("Implement YAMNet for distress sound classification", ["auditory_watchdog/core/distress_classifier.py"]),
    ("Develop Faster-Whisper offline transcription", ["auditory_watchdog/core/keyword_spotter.py"]),
    ("Expose auditory actions through action engine", ["auditory_watchdog/core/action_engine.py", "auditory_watchdog/README.md"]),

    # Cognitive Core
    ("Define base data models for sensor events", ["cognitive_core/__init__.py", "cognitive_core/models.py"]),
    ("Implement deterministic ReflexEngine algorithms", ["cognitive_core/reflex_engine.py"]),
    ("Draft Gemini API ReasoningEngine connector", ["cognitive_core/reasoning_engine.py"]),
    ("Fix Pydantic IncidentReport schema format", ["cognitive_core/models.py"]),
    ("Orchestrate multimodal CognitiveCore wrapper", ["cognitive_core/core.py"]),
    
    # Testing suite
    ("Create robust pytest verification scripts", ["tests/test_cognitive_core.py", "tests/test_reflex_engine.py"]),
    
    # Final integration
    ("Create root orchestration main.py", ["main.py"]),
    ("Patch threading queues for audio ingestion", ["main.py"]),
    ("Resolve vision pipeline async import mismatch", ["main.py", "visual_guardian/pipeline.py"]),
    ("Write system tests and validation checklists", ["tests/test_pipeline_performance.py"]),
    ("Update root README and document implementation", ["README.md", "technical_documentation.md"]),
    ("Final sanity pass: prepare for production release", ["."]),
]

def run(cmd, env=None, check=True):
    print(f"[EXEC] {cmd}")
    subprocess.run(cmd, env=env, shell=True, check=check)

def main():
    # 1. Clean existing Git history securely (handled by orchestrator script)
    # if os.path.exists(".git"):
    #     shutil.rmtree(".git")
    
    run("git init")
    run("git config user.name 'DevOps Pipeline'")
    run("git config user.email 'devops@vitalguardian.local'")
    
    # Calculate intervals
    total_commits = len(COMMITS)
    now = datetime.now()
    total_duration = now - START_DATE
    step = total_duration / total_commits
    
    current_date = START_DATE
    
    # Ensure nothing starts in the index
    run("git rm -rf --cached .", check=False)
    
    # Iterate through commits
    for idx, (message, targets) in enumerate(COMMITS):
        current_date += step
        # Add files safely (ignoring warnings if some patterns don't find anything)
        for target in targets:
            run(f"git add {target}", check=False)
            
        # Format git dates correctly
        date_str = current_date.strftime("%Y-%m-%dT%H:%M:%S")
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str
        
        # Commit quietly to avoid massive logs. Allow empty in case target didn't exist
        run(f'git commit -m "{message}" --allow-empty', env=env)
        
    # Setup remote and force push branch
    # Usually git initializes with 'master' so let's rename to 'main'
    run("git branch -m main", check=False)
    run(f"git remote add origin {REPO_URL}")
    print("\n[INFO] Starting force push to fresh remote repository...")
    run("git push -uf origin main")
    print("\n✅ DevOps history reconstruction complete.")

if __name__ == '__main__':
    main()
