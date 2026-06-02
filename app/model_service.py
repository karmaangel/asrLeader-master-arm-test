from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from funasr import AutoModel

import config
from leader_store import LeaderStore


logger = logging.getLogger(__name__)


def detect_device() -> str:
    if config.DEVICE != "auto":
        return config.DEVICE
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu:0"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def embedding_to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "reshape"):
        value = value.reshape(-1).tolist()
    return [float(v) for v in value]


class FunASRService:
    def __init__(self) -> None:
        self.device = detect_device()
        self.model: AutoModel | None = None

    def load(self) -> None:
        logger.info("Loading FunASR models")
        logger.info("ASR=%s", config.ASR_MODEL)
        logger.info("VAD=%s", config.VAD_MODEL)
        logger.info("PUNC=%s", config.PUNC_MODEL)
        logger.info("SPK=%s", config.SPK_MODEL)
        logger.info("DEVICE=%s", self.device)
        self.model = AutoModel(
            model=config.ASR_MODEL,
            vad_model=config.VAD_MODEL,
            punc_model=config.PUNC_MODEL,
            spk_model=config.SPK_MODEL,
            device=self.device,
            disable_update=True,
        )
        logger.info("FunASR models are ready")

    def transcribe(self, audio_path: str, num_speakers: int | None = None) -> list[dict[str, Any]]:
        self._ensure_loaded()
        kwargs: dict[str, Any] = {
            "input": audio_path,
            "batch_size_s": config.BATCH_SIZE_S,
        }
        if num_speakers and num_speakers > 0:
            kwargs["spk_kwargs"] = {"num_speakers": num_speakers}

        raw_results = self.model.generate(**kwargs)
        full_text: list[str] = []
        segments: list[dict[str, Any]] = []

        for result in raw_results or []:
            full_text.append(result.get("text", ""))
            for item in result.get("sentence_info", []):
                start = round(float(item.get("start", 0)) / 1000.0, 3)
                end = round(float(item.get("end", 0)) / 1000.0, 3)
                segments.append(
                    {
                        "speaker": str(item.get("spk", 0)),
                        "start_time": round(start, 2),
                        "end_time": round(end, 2),
                        "text": item.get("text", ""),
                        "is_leader": False,
                        "leader_id": None,
                    }
                )

        return segments

    def extract_voiceprint(self, audio_path: str) -> list[float]:
        self._ensure_loaded()
        if self.model.spk_model is None:
            raise RuntimeError("speaker model is not loaded")
        spk_kwargs = dict(getattr(self.model, "spk_kwargs", {}) or {})
        spk_kwargs["device"] = self.device
        results = self.model.inference(audio_path, model=self.model.spk_model, kwargs=spk_kwargs)
        if not results or "spk_embedding" not in results[0]:
            raise RuntimeError("could not extract speaker embedding")
        return embedding_to_list(results[0]["spk_embedding"])

    def annotate_leaders(
        self,
        audio_path: str,
        segments: list[dict[str, Any]],
        leader_store: LeaderStore,
        threshold: float,
        return_scores: bool = False,
    ) -> list[dict[str, Any]]:
        if not leader_store.has_samples():
            return segments

        segment_scores: dict[int, list[dict[str, Any]]] = {}
        for index, segment in enumerate(segments):
            scores = self._score_segment(audio_path, segment, leader_store)
            segment_scores[index] = scores
            if return_scores:
                segment["leader_candidates"] = scores

        speaker_matches = self._identify_speakers(audio_path, segments, leader_store, segment_scores)
        for segment in segments:
            match = speaker_matches.get(segment["speaker"])
            if match:
                self._mark_leader(segment, match)
        return segments

    def _identify_speakers(
        self,
        audio_path: str,
        segments: list[dict[str, Any]],
        leader_store: LeaderStore,
        segment_scores: dict[int, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        speaker_segments: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, segment in enumerate(segments):
            if segment["end_time"] - segment["start_time"] >= config.LEADER_MIN_SEGMENT_SECONDS:
                speaker_segments.setdefault(segment["speaker"], []).append((index, segment))

        leader_candidates: dict[str, list[dict[str, Any]]] = {}
        for speaker, indexed_items in speaker_segments.items():
            items = [item for _, item in indexed_items]
            combined_path = self._extract_speaker_audio(audio_path, items)
            try:
                embedding = self.extract_voiceprint(combined_path)
                speaker_scores = leader_store.score_all(embedding)
                segment_evidence = self._speaker_segment_evidence(indexed_items, segment_scores)
                for score in speaker_scores:
                    leader_id = score["leader_id"]
                    evidence = max(score["score"], segment_evidence.get(leader_id, {}).get("top_avg", -1.0))
                    next_leader_score = self._next_score(speaker_scores, leader_id)
                    candidate = {
                        "speaker": speaker,
                        "leader_id": leader_id,
                        "score": round(evidence, 5),
                        "speaker_score": score["score"],
                        "segment_top_avg": segment_evidence.get(leader_id, {}).get("top_avg"),
                        "support_segments": segment_evidence.get(leader_id, {}).get("support", 0),
                        "speaker_margin": round(score["score"] - next_leader_score, 5),
                    }
                    if self._candidate_has_evidence(candidate):
                        leader_candidates.setdefault(leader_id, []).append(candidate)
            finally:
                try:
                    os.remove(combined_path)
                except OSError:
                    pass

        assigned_speakers: set[str] = set()
        matches: dict[str, dict[str, Any]] = {}
        winners: list[dict[str, Any]] = []
        for _, candidates in leader_candidates.items():
            candidates.sort(key=lambda item: item["score"], reverse=True)
            best = candidates[0]
            runner_up = candidates[1]["score"] if len(candidates) > 1 else -1.0
            best["meeting_margin"] = round(best["score"] - runner_up, 5)
            if self._candidate_wins_meeting(best, runner_up):
                winners.append(best)

        for candidate in sorted(winners, key=lambda item: item["score"], reverse=True):
            speaker = candidate["speaker"]
            if speaker in assigned_speakers:
                continue
            matches[speaker] = {
                "leader_id": candidate["leader_id"],
                "score": candidate["score"],
                "confidence": self._confidence_label(candidate),
            }
            assigned_speakers.add(speaker)
        return matches

    def _speaker_segment_evidence(
        self,
        indexed_segments: list[tuple[int, dict[str, Any]]],
        segment_scores: dict[int, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        by_leader: dict[str, list[float]] = {}
        for index, _ in indexed_segments:
            scores = segment_scores.get(index, [])
            if not scores:
                continue
            best = scores[0]
            runner_up = scores[1]["score"] if len(scores) > 1 else -1.0
            if best["score"] - runner_up >= config.LEADER_SCORE_MARGIN:
                by_leader.setdefault(best["leader_id"], []).append(best["score"])

        evidence: dict[str, dict[str, Any]] = {}
        for leader_id, values in by_leader.items():
            top_values = sorted(values, reverse=True)[:3]
            evidence[leader_id] = {
                "top_avg": round(sum(top_values) / len(top_values), 5),
                "support": len(values),
            }
        return evidence

    def _score_segment(
        self,
        audio_path: str,
        segment: dict[str, Any],
        leader_store: LeaderStore,
    ) -> list[dict[str, Any]]:
        if segment["end_time"] - segment["start_time"] < config.LEADER_MIN_SEGMENT_SECONDS:
            return []
        clip_path = self._extract_clip(audio_path, segment["start_time"], segment["end_time"])
        try:
            embedding = self.extract_voiceprint(clip_path)
            return leader_store.score_all(embedding)
        finally:
            try:
                os.remove(clip_path)
            except OSError:
                pass

    @staticmethod
    def _next_score(scores: list[dict[str, Any]], leader_id: str) -> float:
        for score in scores:
            if score["leader_id"] != leader_id:
                return score["score"]
        return -1.0

    def _candidate_has_evidence(self, candidate: dict[str, Any]) -> bool:
        if candidate["speaker_score"] >= 0.50:
            return True
        if candidate["speaker_margin"] < config.LEADER_SCORE_MARGIN:
            return False
        if candidate["speaker_score"] >= config.LEADER_SPEAKER_THRESHOLD:
            return True
        if candidate["support_segments"] >= 2 and candidate["score"] >= config.LEADER_SEGMENT_THRESHOLD:
            return True
        return False

    def _candidate_wins_meeting(self, candidate: dict[str, Any], runner_up: float) -> bool:
        if runner_up < 0:
            return True
        adaptive_gap = max(config.LEADER_SCORE_MARGIN, min(0.12, candidate["score"] * 0.18))
        if candidate["score"] - runner_up >= adaptive_gap:
            return True
        return candidate["support_segments"] >= 2 and candidate["score"] - runner_up >= config.LEADER_SCORE_MARGIN

    @staticmethod
    def _confidence_label(candidate: dict[str, Any]) -> str:
        if candidate["score"] >= 0.55 and candidate["speaker_margin"] >= 0.08:
            return "high"
        if candidate["score"] >= 0.40 and candidate["speaker_margin"] >= 0.04:
            return "medium"
        return "low"

    def _mark_leader(self, segment: dict[str, Any], match: dict[str, Any]) -> None:
        segment["is_leader"] = True
        segment["leader_id"] = match["leader_id"]

    def _extract_clip(self, audio_path: str, start: float, end: float) -> str:
        start = max(0.0, start - config.SEGMENT_PADDING_MS / 1000.0)
        duration = max(0.2, end - start + config.SEGMENT_PADDING_MS / 1000.0)
        out_path = str(Path(tempfile.gettempdir()) / f"leader_clip_{uuid.uuid4().hex}.wav")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            audio_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            out_path,
        ]
        subprocess.run(cmd, check=True)
        return out_path

    def _extract_speaker_audio(self, audio_path: str, segments: list[dict[str, Any]]) -> str:
        clip_paths = []
        list_path = str(Path(tempfile.gettempdir()) / f"leader_concat_{uuid.uuid4().hex}.txt")
        out_path = str(Path(tempfile.gettempdir()) / f"leader_speaker_{uuid.uuid4().hex}.wav")
        try:
            longest = sorted(segments, key=lambda item: item["end_time"] - item["start_time"], reverse=True)[:8]
            for segment in sorted(longest, key=lambda item: item["start_time"]):
                clip_paths.append(self._extract_clip(audio_path, segment["start_time"], segment["end_time"]))
            with open(list_path, "w", encoding="utf-8") as f:
                for path in clip_paths:
                    escaped = path.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                out_path,
            ]
            subprocess.run(cmd, check=True)
            return out_path
        finally:
            for path in clip_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                os.remove(list_path)
            except OSError:
                pass

    def _ensure_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("model is not loaded")

    @staticmethod
    def _merge_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for segment in segments:
            if (
                merged
                and merged[-1]["speaker"] == segment["speaker"]
                and FunASRService._leader_id(merged[-1]) == FunASRService._leader_id(segment)
            ):
                merged[-1]["text"] += segment["text"]
                merged[-1]["end_time"] = segment["end_time"]
                merged[-1]["is_leader"] = merged[-1].get("is_leader", False) or segment.get("is_leader", False)
                if segment.get("leader_id"):
                    merged[-1]["leader_id"] = segment["leader_id"]
            else:
                segment.setdefault("is_leader", False)
                segment.setdefault("leader_id", None)
                segment.pop("leader", None)
                merged.append(segment)
        return merged

    @staticmethod
    def _leader_id(segment: dict[str, Any]) -> str | None:
        return segment.get("leader_id")
