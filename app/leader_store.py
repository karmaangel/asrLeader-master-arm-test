from __future__ import annotations

import json
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("embedding dimensions are different")
    if not a:
        raise ValueError("embedding is empty")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("embedding cannot be a zero vector")
    return dot / (norm_a * norm_b)


class LeaderStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"leaders": {}})

    def _read(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                return {"leaders": {}}
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("leaders", {})
            return data

    def _write(self, data: dict[str, Any]) -> None:
        with self._lock:
            tmp_path = self.path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp_path.replace(self.path)

    def list(self) -> list[dict[str, Any]]:
        data = self._read()
        leaders = []
        for leader_id, item in sorted(data["leaders"].items()):
            leaders.append(
                {
                    "leader_id": leader_id,
                    "sample_count": len(item.get("samples", [])),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                }
            )
        return leaders

    def create_leader(self, embeddings: list[tuple[list[float], str | None]]) -> dict[str, Any]:
        leader_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        data = self._read()
        data["leaders"][leader_id] = {
            "created_at": now,
            "updated_at": now,
            "samples": [
                {
                    "embedding": embedding,
                    "source": source,
                    "created_at": now,
                    "dimension": len(embedding),
                }
                for embedding, source in embeddings
            ],
        }
        self._write(data)
        dimensions = sorted({len(embedding) for embedding, _ in embeddings})
        return {
            "leader_id": leader_id,
            "sample_count": len(embeddings),
            "dimensions": dimensions,
        }

    def add_sample(
        self,
        leader_id: str,
        name: str | None,
        embedding: list[float],
        source: str | None = None,
    ) -> dict[str, Any]:
        if not leader_id:
            raise ValueError("leader_id is required")
        now = datetime.now(timezone.utc).isoformat()
        data = self._read()
        leader = data["leaders"].setdefault(
            leader_id,
            {
                "name": name or leader_id,
                "created_at": now,
                "updated_at": now,
                "samples": [],
            },
        )
        if name:
            leader["name"] = name
        leader["updated_at"] = now
        leader.setdefault("samples", []).append(
            {
                "embedding": embedding,
                "source": source,
                "created_at": now,
                "dimension": len(embedding),
            }
        )
        self._write(data)
        return {
            "leader_id": leader_id,
            "name": leader.get("name", leader_id),
            "sample_count": len(leader.get("samples", [])),
            "dimension": len(embedding),
        }

    def delete(self, leader_id: str) -> bool:
        data = self._read()
        existed = leader_id in data["leaders"]
        if existed:
            del data["leaders"][leader_id]
            self._write(data)
        return existed

    def identify(self, embedding: list[float], threshold: float) -> dict[str, Any] | None:
        scores = self.score_all(embedding)
        best = scores[0] if scores else None
        if best is None or best["score"] < threshold:
            return None
        return best

    def score_all(self, embedding: list[float]) -> list[dict[str, Any]]:
        data = self._read()
        best_by_leader: dict[str, dict[str, Any]] = {}
        for leader_id, item in data["leaders"].items():
            for sample in item.get("samples", []):
                score = cosine_similarity(embedding, sample["embedding"])
                current = best_by_leader.get(leader_id)
                if current is None or score > current["score"]:
                    best_by_leader[leader_id] = {
                        "leader_id": leader_id,
                        "score": score,
                    }
        scores = sorted(best_by_leader.values(), key=lambda item: item["score"], reverse=True)
        for item in scores:
            item["score"] = round(item["score"], 5)
        return scores

    def has_samples(self) -> bool:
        return any(item.get("samples") for item in self._read()["leaders"].values())
