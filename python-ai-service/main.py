import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.config import load_config
from core.engine import AIEngine
from core.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_settings() -> dict:
    path = Path(__file__).parent / "appsettings.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


settings = _load_settings()
engine_cfg = settings.get("engine", {})

gpu_available = _gpu_available()
config = load_config()

engine = AIEngine(config=config)
app = FastAPI(title="PhotoSelector AI Service", version="3.2.0")


class AnalyzeRequest(BaseModel):
    image_path: str


class AnalyzeBatchRequest(BaseModel):
    image_paths: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    image_path: str
    manual_tag: str
    note: str = ""
    predicted_is_waste: bool = False
    predicted_style: str = "unknown"
    predicted_face_count: int = 0


class PersonRenameRequest(BaseModel):
    old_label: str
    new_label: str


class PersonAssignRequest(BaseModel):
    image_path: str
    new_label: str


@app.on_event("startup")
async def startup() -> None:
    engine.load_plugins()
    await engine.startup()
    logger.info("Started with %d plugins: %s", len(engine.plugins), engine.plugin_names)


@app.on_event("shutdown")
async def shutdown() -> None:
    await engine.shutdown()
    logger.info("Engine shutdown complete")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "plugins": engine.plugin_names,
        "scheduler": engine.scheduler.queue_stats,
        "models": ModelRegistry.get_status(),
    }


@app.get("/loading/progress")
async def loading_progress() -> dict:
    statuses = ModelRegistry.get_status()
    total = len(statuses)
    loaded = sum(1 for s in statuses if s["loaded"])
    return {
        "total": total,
        "loaded": loaded,
        "models": statuses,
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> dict:
    try:
        return await engine.analyze(req.image_path)
    except FileNotFoundError as ex:
        raise HTTPException(status_code=404, detail=str(ex)) from ex
    except Exception as ex:
        logger.exception("Analyze failed for %s", req.image_path)
        raise HTTPException(status_code=500, detail=f"Analyze failed: {ex}") from ex


@app.post("/analyze/batch")
async def analyze_batch(req: AnalyzeBatchRequest) -> dict:
    if not req.image_paths:
        return {"items": []}
    try:
        items = await engine.analyze_many(req.image_paths)
        return {"items": items}
    except FileNotFoundError as ex:
        raise HTTPException(status_code=404, detail=str(ex)) from ex
    except Exception as ex:
        logger.exception("Batch analyze failed")
        raise HTTPException(status_code=500, detail=f"Batch analyze failed: {ex}") from ex


@app.post("/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    try:
        learning_state = engine.apply_feedback(
            manual_tag=req.manual_tag,
            predicted_is_waste=req.predicted_is_waste,
            predicted_face_count=req.predicted_face_count,
        )
        return {
            "status": "ok",
            "message": "feedback accepted",
            "learning_state": learning_state,
        }
    except Exception as ex:
        logger.exception("Feedback failed")
        raise HTTPException(status_code=500, detail=f"Feedback failed: {ex}") from ex


@app.post("/person/rename")
async def rename_person(req: PersonRenameRequest) -> dict:
    try:
        result = engine.rename_person(req.old_label, req.new_label)
        return {"status": "ok", "result": result}
    except Exception as ex:
        logger.exception("Rename failed")
        raise HTTPException(status_code=500, detail=f"Rename failed: {ex}") from ex


@app.post("/person/assign")
async def assign_person(req: PersonAssignRequest) -> dict:
    try:
        result = engine.assign_person(req.image_path, req.new_label)
        return {"status": "ok", "result": result}
    except Exception as ex:
        logger.exception("Assign failed")
        raise HTTPException(status_code=500, detail=f"Assign failed: {ex}") from ex
