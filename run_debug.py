import sys
import traceback
from pathlib import Path

# Ensure project root is added to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.orchestrator import execute_pipeline

if __name__ == "__main__":
    # Update this path to point to your real .SAFE directory or .zip file
    safe_file = PROJECT_ROOT / "backend" / "data" / "scenes" / "huntington_beach_2021"/ "S1B_IW_GRDH_1SDV_20211003T014927_20211003T014952_028965_0374DA_3EE4.SAFE"

    print(f"Checking SAFE path: {safe_file} (Exists: {safe_file.exists()})")

    try:
        res = execute_pipeline(
            scene_id="huntington_beach_2021",
            run_mode="live",
            safe_path=str(safe_file)  # Pass the SAFE folder directly
        )
        print("\nPipeline succeeded:", res)
    except Exception as e:
        print("\n--- CRASH DETECTED ---")
        traceback.print_exc()