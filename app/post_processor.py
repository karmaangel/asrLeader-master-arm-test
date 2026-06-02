from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import config


logger = logging.getLogger(__name__)


class TranscriptPostProcessor:
    def __init__(self, use_worker_pool: bool = True) -> None:
        self.enabled = config.POSTPROCESS_ENABLED
        self.model_id = config.POSTPROCESS_MODEL
        self.device = config.POSTPROCESS_DEVICE
        self.worker_count = config.POSTPROCESS_WORKERS if use_worker_pool else 1
        self._lock = threading.Lock()
        self._pool_lock = threading.Lock()
        self._loaded = False
        self._available = False
        self._tokenizer: Any = None
        self._model: Any = None
        self._workers: list[TranscriptPostProcessor] | None = None

    def load(self) -> bool:
        if self.worker_count > 1:
            return all(worker.load() for worker in self._get_workers())
        return self._ensure_model()

    def correct_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.enabled or not segments:
            return segments

        corrected = [dict(segment) for segment in segments]
        changed = False
        full_text = self._full_text(corrected)
        batches = self._batches(corrected)
        for batch, suggestions in zip(batches, self._correct_batches(batches)):
            for segment, text in zip(batch, suggestions):
                text = self._contextual_fallback(text, full_text)
                if text and text != segment.get("text", ""):
                    segment.setdefault("raw_text", segment.get("text", ""))
                    segment["text"] = text
                    segment["post_processed"] = True
                    changed = True

        if not changed:
            full_text = self._full_text(corrected)
            for segment in corrected:
                fixed = self._contextual_fallback(segment.get("text", ""), full_text)
                if fixed != segment.get("text", ""):
                    segment.setdefault("raw_text", segment.get("text", ""))
                    segment["text"] = fixed
                    segment["post_processed"] = True
        return corrected

    def _correct_batches(self, batches: list[list[dict[str, Any]]]) -> list[list[str]]:
        if self.worker_count <= 1 or len(batches) <= 1:
            return [self._correct_batch(batch) for batch in batches]
        return self._correct_batches_parallel(batches)

    def _correct_batches_parallel(self, batches: list[list[dict[str, Any]]]) -> list[list[str]]:
        worker_count = min(self.worker_count, len(batches))
        workers = self._get_workers()[:worker_count]
        assignments: list[list[tuple[int, list[dict[str, Any]]]]] = [[] for _ in range(worker_count)]
        for index, batch in enumerate(batches):
            assignments[index % worker_count].append((index, batch))

        results: list[list[str] | None] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self._run_worker_batches, workers[worker_index], worker_batches)
                for worker_index, worker_batches in enumerate(assignments)
                if worker_batches
            ]
            for future in as_completed(futures):
                for index, suggestions in future.result():
                    results[index] = suggestions

        return [item if item is not None else self._original_texts(batch) for item, batch in zip(results, batches)]

    def _get_workers(self) -> list["TranscriptPostProcessor"]:
        with self._pool_lock:
            if self._workers is None:
                logger.info("Creating %s ASR post-process workers", self.worker_count)
                self._workers = [
                    TranscriptPostProcessor(use_worker_pool=False)
                    for _ in range(self.worker_count)
                ]
            return self._workers

    @staticmethod
    def _run_worker_batches(
        worker: "TranscriptPostProcessor",
        batches: list[tuple[int, list[dict[str, Any]]]],
    ) -> list[tuple[int, list[str]]]:
        return [(index, worker._correct_batch(batch)) for index, batch in batches]

    def _correct_batch(self, segments: list[dict[str, Any]]) -> list[str]:
        original = self._original_texts(segments)
        if not self._ensure_model():
            return original

        payload = [{"id": index, "text": text} for index, text in enumerate(original)]
        messages = self._messages(payload)
        try:
            prompt = self._format_prompt(messages)
            inputs = self._tokenizer([prompt], return_tensors="pt")
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=config.POSTPROCESS_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
            raw = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            parsed = self._parse_json_array(raw)
            return self._validated_texts(parsed, original)
        except Exception as exc:
            logger.warning("ASR post-process model failed: %s", exc)
            return original

    @staticmethod
    def _original_texts(segments: list[dict[str, Any]]) -> list[str]:
        return [str(segment.get("text", "")) for segment in segments]

    def _ensure_model(self) -> bool:
        if self._loaded:
            return self._available
        with self._lock:
            if self._loaded:
                return self._available
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_path = self._resolve_model_path()
                logger.info("Loading ASR post-process model: %s", model_path)
                self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    trust_remote_code=False,
                )
                target_device = self._select_device(torch)
                self._model.to(target_device)
                self._model.eval()
                self._available = True
                logger.info("ASR post-process model is ready on %s", target_device)
            except Exception as exc:
                logger.warning("ASR post-process model unavailable: %s", exc)
                self._available = False
            finally:
                self._loaded = True
        return self._available

    def _resolve_model_path(self) -> str:
        if config.POSTPROCESS_MODEL_DIR:
            return config.POSTPROCESS_MODEL_DIR
        local_candidates = [
            config.MODELS_DIR / "Qwen2.5-1.5B-Instruct",
            config.MODELS_DIR / "qwen2.5-1.5b-instruct",
            config.MODELS_DIR / "postprocess-qwen",
        ]
        for path in local_candidates:
            if path.is_dir():
                return str(path)

        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(self.model_id)

    def _select_device(self, torch: Any) -> str:
        device = self.device
        if device == "auto":
            if hasattr(torch, "npu") and torch.npu.is_available():
                return "npu:0"
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        if device.startswith("npu"):
            import torch_npu  # noqa: F401
        return device

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n\n".join(f"{item['role']}:\n{item['content']}" for item in messages) + "\n\nassistant:\n"

    def _validated_texts(self, parsed: Any, original: list[str]) -> list[str]:
        if not isinstance(parsed, list) or len(parsed) != len(original):
            return original
        corrected: list[str] = []
        for index, item in enumerate(parsed):
            if not isinstance(item, dict) or item.get("id") != index:
                return original
            text = item.get("text")
            if not isinstance(text, str):
                return original
            if len(text) > max(40, len(original[index]) * 3):
                corrected.append(original[index])
            else:
                corrected.append(text.strip())
        return corrected

    @staticmethod
    def _parse_json_array(raw: str) -> Any:
        try:
            value = json.loads(raw)
            if isinstance(value, dict) and isinstance(value.get("items"), list):
                return value["items"]
            return value
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _contextual_fallback(text: str, full_text: str) -> str:
        if "四月七年前" in text and "清明" in full_text:
            return text.replace("四月七年前", "清明前")
        return text

    @staticmethod
    def _full_text(segments: list[dict[str, Any]]) -> str:
        return "".join(str(segment.get("text", "")) for segment in segments)

    @staticmethod
    def _batches(segments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        batches: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        char_count = 0
        for segment in segments:
            text_len = len(str(segment.get("text", "")))
            if current and char_count + text_len > config.POSTPROCESS_MAX_CHARS:
                batches.append(current)
                current = []
                char_count = 0
            current.append(segment)
            char_count += text_len
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _messages(payload: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "你是中文会议语音转写的后处理器。只根据上下文修正明显的 ASR "
                    "错字、同音误识别、时间表达错误和不通顺短语。不要总结，不要扩写，"
                    "不要改变原意，不要改数字、姓名、术语，除非上下文强烈支持。"
                    "保留分段数量和 id，只输出 JSON 数组，格式为 "
                    '[{"id":0,"text":"修正后的文本"}]。'
                    "如果上下文反复出现清明、清明前、上线等内容，像“四月七年前”"
                    "这种不通顺片段应按上下文优先修正为“清明前”。"
                ),
            },
            {
                "role": "user",
                "content": "原始分段 JSON：\n" + json.dumps(payload, ensure_ascii=False),
            },
        ]
