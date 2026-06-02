from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

import config
from leader_store import LeaderStore, cosine_similarity
from model_service import FunASRService
from post_processor import TranscriptPostProcessor


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
correction_executor = ThreadPoolExecutor(max_workers=config.CORRECTION_TASK_WORKERS)
correction_tasks: dict[str, dict[str, Any]] = {}
correction_tasks_lock = threading.Lock()
asr_service = FunASRService()
leader_store = LeaderStore(config.LEADER_DB_PATH)
post_processor = TranscriptPostProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, asr_service.load)
    if config.POSTPROCESS_ENABLED and config.POSTPROCESS_PRELOAD:
        await loop.run_in_executor(executor, post_processor.load)
    yield


app = FastAPI(
    title="FunASR Leader ASR",
    description="Speech recognition, speaker diarization, and leader voiceprint identification.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    for schema in openapi_schema.get("components", {}).get("schemas", {}).values():
        properties = schema.get("properties", {})
        for prop in properties.values():
            items = prop.get("items")
            if isinstance(items, dict) and items.get("contentMediaType") == "application/octet-stream":
                items.pop("contentMediaType", None)
                items["format"] = "binary"
            if prop.get("contentMediaType") == "application/octet-stream":
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def ok(data: Any = None) -> dict[str, Any]:
    return {"code": 0, "message": "success", "data": data}


def transcribe_ok(data: Any = None, **extra: Any) -> dict[str, Any]:
    response = {"code": 200, "message": "success", "data": data}
    response.update(extra)
    return response


def now_ts() -> float:
    return round(time.time(), 3)


def public_correction_task(task: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task.items() if key != "future"}


def get_correction_task(task_id: str) -> dict[str, Any] | None:
    with correction_tasks_lock:
        task = correction_tasks.get(task_id)
        if task is None:
            return None
        return public_correction_task(task)


def segment_text_chars(segments: list[dict[str, Any]]) -> int:
    return sum(len(str(segment.get("text", ""))) for segment in segments)


def run_correction_task(task_id: str, segments: list[dict[str, Any]]) -> None:
    with correction_tasks_lock:
        task = correction_tasks.get(task_id)
        if task is not None:
            task["status"] = "running"
            task["updated_at"] = now_ts()
    try:
        result = post_processor.correct_segments([dict(segment) for segment in segments])
        with correction_tasks_lock:
            task = correction_tasks.get(task_id)
            if task is not None:
                task["status"] = "completed"
                task["updated_at"] = now_ts()
                task["result"] = result
    except Exception as exc:
        logger.exception("ASR correction task failed: %s", task_id)
        with correction_tasks_lock:
            task = correction_tasks.get(task_id)
            if task is not None:
                task["status"] = "failed"
                task["updated_at"] = now_ts()
                task["error"] = str(exc)


def submit_correction_task(segments: list[dict[str, Any]], mode: str) -> tuple[str, Any]:
    task_id = uuid.uuid4().hex
    task = {
        "task_id": task_id,
        "status": "queued",
        "mode": mode,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "result": None,
        "error": None,
    }
    with correction_tasks_lock:
        correction_tasks[task_id] = task
    future = correction_executor.submit(run_correction_task, task_id, [dict(segment) for segment in segments])
    with correction_tasks_lock:
        correction_tasks[task_id]["future"] = future
    return task_id, future


def correction_pending_fields(task_id: str, mode: str) -> dict[str, Any]:
    return {
        "correction_status": "running",
        "correction_mode": mode,
        "correction_task_id": task_id,
        "poll_url": f"/corrections/{task_id}",
    }


def normalize_correction_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode in {"", "auto"}:
        return "auto"
    if mode in {"sync", "async", "none"}:
        return mode
    raise HTTPException(status_code=400, detail="correction_mode must be auto, sync, async, or none")


async def save_upload(file: UploadFile, prefix: str) -> str:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty audio file")
    if len(content) > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"file is larger than {config.MAX_FILE_SIZE} bytes")
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    path = os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(content)
    return path


def parse_embedding(raw: str) -> list[float]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="embedding must be valid JSON") from exc
    if isinstance(value, dict):
        value = value.get("embedding")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="embedding must be a numeric array")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="embedding contains non-numeric values") from exc


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": asr_service.model is not None,
        "device": asr_service.device,
        "leader_count": len(leader_store.list()),
        "models_dir": str(config.MODELS_DIR),
        "postprocess_enabled": config.POSTPROCESS_ENABLED,
        "postprocess_model": config.POSTPROCESS_MODEL if config.POSTPROCESS_ENABLED else None,
        "postprocess_device": config.POSTPROCESS_DEVICE if config.POSTPROCESS_ENABLED else None,
        "postprocess_workers": config.POSTPROCESS_WORKERS if config.POSTPROCESS_ENABLED else 0,
        "postprocess_sync_max_chars": config.POSTPROCESS_SYNC_MAX_CHARS if config.POSTPROCESS_ENABLED else 0,
        "postprocess_sync_timeout_seconds": config.POSTPROCESS_SYNC_TIMEOUT_SECONDS if config.POSTPROCESS_ENABLED else 0,
        "correction_task_workers": config.CORRECTION_TASK_WORKERS if config.POSTPROCESS_ENABLED else 0,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(default=None),
    identify_leaders: bool = Form(default=True),
    leader_threshold: float = Form(default=config.LEADER_THRESHOLD),
    return_leader_scores: bool = Form(default=False),
    post_process: bool = Form(default=config.POSTPROCESS_ENABLED),
    correct_text: bool | None = Form(default=None),
    correction_mode: str = Form(default="auto"),
):
    path = await save_upload(file, "asr")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, asr_service.transcribe, path, num_speakers)
        if identify_leaders:
            result = await loop.run_in_executor(
                executor,
                asr_service.annotate_leaders,
                path,
                result,
                leader_store,
                leader_threshold,
                return_leader_scores,
            )
        result = asr_service._merge_adjacent_segments(result)
        should_correct_text = post_process if correct_text is None else correct_text
        mode = normalize_correction_mode(correction_mode)
        if mode == "none":
            should_correct_text = False
        if should_correct_text:
            if mode == "async" or (
                mode == "auto" and segment_text_chars(result) > config.POSTPROCESS_SYNC_MAX_CHARS
            ):
                task_id, _ = submit_correction_task(result, mode)
                return transcribe_ok(result, **correction_pending_fields(task_id, mode))

            task_id, future = submit_correction_task(result, mode)
            try:
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=config.POSTPROCESS_SYNC_TIMEOUT_SECONDS)
                task = get_correction_task(task_id)
                if task and task.get("status") == "completed":
                    return transcribe_ok(task.get("result", result))
                if task and task.get("status") == "failed":
                    logger.warning("ASR correction task failed, returning raw ASR: %s", task.get("error"))
                    return transcribe_ok(result)
            except asyncio.TimeoutError:
                return transcribe_ok(result, **correction_pending_fields(task_id, "timeout"))
        return transcribe_ok(result)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.get("/corrections/{task_id}")
async def correction_result(task_id: str):
    task = get_correction_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="correction task not found")
    return ok(task)


@app.post("/leaders/enroll")
async def enroll_leader(
    files: List[UploadFile] = File(..., description="Upload at least two voiceprint audio files."),
):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="at least two voiceprint files are required")

    paths: list[str] = []
    try:
        loop = asyncio.get_event_loop()
        embeddings: list[tuple[list[float], str | None]] = []
        for file in files:
            path = await save_upload(file, "leader")
            paths.append(path)
            embedding = await loop.run_in_executor(executor, asr_service.extract_voiceprint, path)
            embeddings.append((embedding, file.filename))
        item = leader_store.create_leader(embeddings)
        return ok(item)
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


@app.get("/leaders")
async def list_leaders():
    return ok(leader_store.list())


@app.delete("/leaders/{leader_id}")
async def delete_leader(leader_id: str):
    deleted = leader_store.delete(leader_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="leader not found")
    return ok({"deleted": True, "leader_id": leader_id})


@app.post("/voiceprint/extract")
async def voiceprint_extract(file: UploadFile = File(...)):
    path = await save_upload(file, "voiceprint")
    try:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(executor, asr_service.extract_voiceprint, path)
        return ok({"embedding": embedding, "dimension": len(embedding)})
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/voiceprint/verify")
async def voiceprint_verify(
    audio_a: UploadFile = File(...),
    audio_b: UploadFile | None = File(default=None),
    embedding: str | None = Form(default=None),
    threshold: float = Form(default=config.LEADER_THRESHOLD),
):
    if audio_b is None and embedding is None:
        raise HTTPException(status_code=400, detail="provide audio_b or embedding")
    if audio_b is not None and embedding is not None:
        raise HTTPException(status_code=400, detail="audio_b and embedding cannot both be provided")

    paths: list[str] = []
    try:
        loop = asyncio.get_event_loop()
        path_a = await save_upload(audio_a, "voiceprint")
        paths.append(path_a)
        embedding_a = await loop.run_in_executor(executor, asr_service.extract_voiceprint, path_a)

        if audio_b is not None:
            path_b = await save_upload(audio_b, "voiceprint")
            paths.append(path_b)
            embedding_b = await loop.run_in_executor(executor, asr_service.extract_voiceprint, path_b)
        else:
            embedding_b = parse_embedding(embedding or "")

        score = cosine_similarity(embedding_a, embedding_b)
        return ok(
            {
                "score": round(score, 5),
                "threshold": threshold,
                "same_speaker": score >= threshold,
                "dimension": len(embedding_a),
            }
        )
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT, workers=1)
