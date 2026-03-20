import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Attempting to import VisionPipeline...")
try:
    from visual_guardian.pipeline import VisionPipeline
    print("SUCCESS: VisionPipeline imported.")
except ImportError as e:
    print(f"FAILURE: ImportError: {e}")
except Exception as e:
    print(f"FAILURE: Exception: {e}")
