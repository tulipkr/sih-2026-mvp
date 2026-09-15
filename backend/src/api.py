import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.src.orchestrator import execute_pipeline
from backend.src.run_manager import (
    generate_run_id, 
    load_json_safely, 
    init_run_directory, 
    get_utc_now_iso
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("API")

app = FastAPI(
    title="Oil Spill Pipeline Integration API",
    description="Orchestration service serving Zeba's frontend dashboard",
    version="1.0.0",
)

# --- Enable CORS Middleware for Frontend Dashboard Access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = Path(PROJECT_ROOT) / "backend" / "configs" / "config.yaml"

DEFAULT_SCENE_PATHS = {
    "00052": {
        "vv_path": os.path.join(PROJECT_ROOT, "module2", "data", "00052_vv.tif"),
        "vh_path": os.path.join(PROJECT_ROOT, "module2", "data", "00052_vh.tif"),
        "safe_path": None,
    },
    "huntington_beach_2021": {
        "vv_path": os.path.join(PROJECT_ROOT, "backend", "data", "scenes", "huntington_beach_2021", "huntington_vv.tif"),
        "vh_path": os.path.join(PROJECT_ROOT, "backend", "data", "scenes", "huntington_beach_2021", "huntington_vh.tif"),
        "safe_path": None,
    }
}


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config at {CONFIG_PATH}: {e}")
    return {
        "demo_scenes": [{"scene_id": "00052"}, {"scene_id": "huntington_beach_2021"}],
        "run_timeout_seconds": 600,
        "api_host": "0.0.0.0",
        "api_port": 8000,
    }


class PipelineRunRequest(BaseModel):
    scene_id: str = Field(..., description="Target scene ID")
    run_mode: str = Field("live", description="'live' or 'precomputed'")


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/pipeline/run")
def trigger_pipeline_run(req: PipelineRunRequest, background_tasks: BackgroundTasks):
    current_config = load_config()
    demo_scenes = current_config.get("demo_scenes", [])
    allowed_scenes = [s["scene_id"] for s in demo_scenes if isinstance(s, dict) and "scene_id" in s]

    if req.scene_id not in allowed_scenes:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scene_id '{req.scene_id}'. Allowed scenes: {allowed_scenes}",
        )

    base_outputs_dir = os.path.join(PROJECT_ROOT, "backend", "outputs")
    runs_dir = Path(base_outputs_dir) / "runs"

    # Precomputed run logic
    if req.run_mode == "precomputed":
        if runs_dir.exists():
            for run_path in runs_dir.iterdir():
                res_file = run_path / "run_result.json"
                if res_file.exists():
                    res_data = load_json_safely(str(res_file))
                    if res_data and res_data.get("scene_id") == req.scene_id:
                        logger.info(f"Serving precomputed run ID: {res_data['run_id']}")
                        return {
                            "run_id": res_data["run_id"],
                            "scene_id": req.scene_id,
                            "run_mode": "precomputed",
                            "status": "completed",
                            "message": "Serving precomputed cached results.",
                        }

    # Generate run_id and pre-initialize directory & initial manifest state
    run_id = generate_run_id()
    run_dir = init_run_directory(base_outputs_dir, run_id)

    # Initialize a temporary run_result.json to prevent 404 errors while processing
    initial_manifest = {
        "run_id": run_id,
        "scene_id": req.scene_id,
        "run_mode": req.run_mode,
        "started_at_utc": get_utc_now_iso(),
        "completed_at_utc": None,
        "overall_status": "processing",
        "stages": {},
        "failed_stage": None,
        "results": {}
    }
    
    with open(run_dir / "run_result.json", "w", encoding="utf-8") as f:
        json.dump(initial_manifest, f, indent=4)

    scene_defaults = DEFAULT_SCENE_PATHS.get(req.scene_id, {})
    vv_path = scene_defaults.get("vv_path")
    vh_path = scene_defaults.get("vh_path")
    safe_path = scene_defaults.get("safe_path")

    try:
        background_tasks.add_task(
            execute_pipeline,
            scene_id=req.scene_id,
            run_mode=req.run_mode,
            run_id=run_id,
            vv_path=vv_path,
            vh_path=vh_path,
            safe_path=safe_path,
            timeout_seconds=current_config.get("run_timeout_seconds", 600),
            base_outputs_dir=base_outputs_dir,
            config=current_config,
        )
        return {
            "run_id": run_id,
            "scene_id": req.scene_id,
            "run_mode": req.run_mode,
            "status": "processing",
            "message": "Pipeline execution started in background.",
        }
    except Exception as e:
        logger.error(f"Failed to initiate pipeline execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/status/{run_id}")
def get_pipeline_status(run_id: str):
    run_file = Path(PROJECT_ROOT) / "backend" / "outputs" / "runs" / run_id / "run_result.json"
    
    if not run_file.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Run ID '{run_id}' not found."
        )

    res_data = load_json_safely(str(run_file))
    if not res_data:
        raise HTTPException(status_code=500, detail="Corrupted run result file.")

    return {
        "run_id": res_data.get("run_id"),
        "scene_id": res_data.get("scene_id"),
        "run_mode": res_data.get("run_mode"),
        "started_at_utc": res_data.get("started_at_utc"),
        "completed_at_utc": res_data.get("completed_at_utc"),
        "overall_status": res_data.get("overall_status"),
        "stages": res_data.get("stages"),
        "failed_stage": res_data.get("failed_stage"),
    }


@app.get("/results/{run_id}")
@app.get("/api/runs/{run_id}")
def get_pipeline_results(run_id: str):
    run_file = Path(PROJECT_ROOT) / "backend" / "outputs" / "runs" / run_id / "run_result.json"
    
    if not run_file.exists():
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found.")

    res_data = load_json_safely(str(run_file))
    if not res_data:
        raise HTTPException(status_code=500, detail="Failed to parse run results.")

    return res_data


if __name__ == "__main__":
    import uvicorn
    startup_config = load_config()
    host = startup_config.get("api_host", "0.0.0.0")
    port = startup_config.get("api_port", 8000)
    uvicorn.run("backend.src.api:app", host=host, port=port, reload=True)