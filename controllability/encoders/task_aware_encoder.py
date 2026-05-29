from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

import numpy as np

from controllability.encoders.base import EncodedTrajectory, FixedFeatureEncoder
from controllability.envs.trajectory_schema import Trajectory

TOOL_TYPES = ("grep", "glob_files", "read_file", "list_dir", "unknown")
SWE_DIR_CLUSTERS = (
    "root",
    "src",
    "tests",
    "docs",
    "app",
    "lib",
    "config",
    "indexer",
    "queue",
    "store",
    "dispatcher",
    "other",
)
MATH_REASONING_MARKERS = ("let", "therefore", "because", "suppose", "case", "hence", "thus", "so")
MATH_METHOD_GROUPS: dict[str, tuple[str, ...]] = {
    "calculus": ("differentiate", "derivative", "derivatives", "d/d", "slope"),
    "integration": ("integral", "integrate", "antiderivative", "\\int"),
    "algebra": ("algebra", "factor", "expand", "simplify", "substitute", "solve"),
    "trig": ("sin", "cos", "tan", "trig", "radian", "theta"),
    "geometry": ("geometry", "triangle", "angle", "circle", "area", "perimeter"),
    "probability": ("probability", "random", "expected", "counting", "choose"),
    "number_theory": ("modulo", "divisibility", "prime", "integer", "remainder"),
    "inequality": ("inequality", "maximum", "minimum", "bound", "positive"),
}


def _safe_log1p(value: float, scale: float = 1.0) -> float:
    return math.log1p(max(value, 0.0)) / scale


def _hash_index(text: str, size: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % size


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_./:-]*|[0-9]+", text.lower())


def _assistant_text(traj: Trajectory) -> str:
    return "\n".join(str(message.get("content", "")) for message in traj.messages if message.get("role") == "assistant")


def _all_text(traj: Trajectory) -> str:
    return "\n".join(str(message.get("content", "")) for message in traj.messages)


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"_raw": value}
    return parsed if isinstance(parsed, dict) else {"_raw": value}


def _tool_name(call: dict[str, Any]) -> str:
    name = str(call.get("name", "unknown"))
    return name if name in TOOL_TYPES else "unknown"


def _tool_calls(traj: Trajectory) -> list[dict[str, Any]]:
    return [call for call in traj.tool_calls if isinstance(call, dict)]


def _cluster_path(path: str) -> str:
    cleaned = path.strip().strip("./")
    if not cleaned or cleaned in {".", "/"}:
        return "root"
    parts = [part for part in cleaned.split("/") if part]
    first = parts[0].lower() if parts else "root"
    if first in {"test", "tests", "__tests__"}:
        return "tests"
    if first in {"doc", "docs", "documentation"}:
        return "docs"
    if first in SWE_DIR_CLUSTERS:
        return first
    joined = "/".join(part.lower() for part in parts[:3])
    for cluster in SWE_DIR_CLUSTERS:
        if cluster != "other" and cluster in joined:
            return cluster
    return "other"


def _paths_from_tool_call(call: dict[str, Any]) -> list[str]:
    args = _parse_arguments(call.get("arguments"))
    paths = []
    for key in ("path", "path_prefix", "file", "filename"):
        value = args.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    return paths


def _paths_from_tool_outputs(traj: Trajectory) -> list[str]:
    paths = []
    for message in traj.messages:
        if message.get("role") != "tool":
            continue
        content = str(message.get("content", ""))
        for line in content.splitlines():
            candidate = line.strip().split(":", 1)[0]
            if "/" in candidate or candidate.endswith((".py", ".md", ".toml", ".yaml", ".json", ".txt")):
                paths.append(candidate)
    return paths


def _query_terms(call: dict[str, Any]) -> list[str]:
    args = _parse_arguments(call.get("arguments"))
    text_parts = []
    for key in ("pattern", "query", "file_glob", "path", "path_prefix", "_raw"):
        value = args.get(key)
        if isinstance(value, str):
            text_parts.append(value)
    return _tokenize(" ".join(text_parts))


class TaskAwareBehavioralEncoderV3(FixedFeatureEncoder):
    encoder_version = "behavioral_encoder_v3"

    def __init__(
        self,
        env_id: str,
        *,
        swe_query_features: int = 32,
        swe_first_tools: int = 8,
        math_answer_hash_features: int = 8,
        math_lexical_hash_features: int = 12,
    ):
        super().__init__(env_id)
        self.swe_query_features = int(swe_query_features)
        self.swe_first_tools = int(swe_first_tools)
        self.math_answer_hash_features = int(math_answer_hash_features)
        self.math_lexical_hash_features = int(math_lexical_hash_features)
        if self.swe_query_features < 1:
            raise ValueError("swe_query_features must be positive")
        if self.swe_first_tools < 1:
            raise ValueError("swe_first_tools must be positive")
        if self.math_answer_hash_features < 1:
            raise ValueError("math_answer_hash_features must be positive")
        if self.math_lexical_hash_features < 1:
            raise ValueError("math_lexical_hash_features must be positive")
        self.feature_dim = self._feature_dim_for_env()

    def _feature_dim_for_env(self) -> int:
        if self._is_swe_grep:
            return (
                len(TOOL_TYPES)
                + self.swe_first_tools * len(TOOL_TYPES)
                + len(SWE_DIR_CLUSTERS)
                + self.swe_query_features
                + 10
            )
        if self._is_math:
            return (
                len(MATH_REASONING_MARKERS)
                + 8
                + 7
                + self.math_answer_hash_features
                + 6
                + len(MATH_METHOD_GROUPS)
                + self.math_lexical_hash_features
                + 5
            )
        return FixedFeatureEncoder.feature_dim

    @property
    def _is_swe_grep(self) -> bool:
        return "swe-grep" in self.env_id or "swe_grep" in self.env_id

    @property
    def _is_math(self) -> bool:
        return "math500" in self.env_id or "math" in self.env_id

    def encode(self, traj: Trajectory) -> EncodedTrajectory:
        if self._is_swe_grep:
            features, raw = self._encode_swe_grep(traj)
        elif self._is_math:
            features, raw = self._encode_math(traj)
        else:
            encoded = super().encode(traj)
            return EncodedTrajectory(
                trajectory_id=encoded.trajectory_id,
                env_id=encoded.env_id,
                behavior_features=encoded.behavior_features,
                quality=encoded.quality,
                coherence=encoded.coherence,
                raw_features={"fallback_encoder": encoded.raw_features},
                encoder_version=self.encoder_version,
            )
        return EncodedTrajectory(
            trajectory_id=traj.trajectory_id,
            env_id=traj.env_id,
            behavior_features=features.astype(np.float32),
            quality=float(traj.reward),
            coherence=1.0 if traj.messages else 0.0,
            raw_features=raw,
            encoder_version=self.encoder_version,
        )

    def _encode_swe_grep(self, traj: Trajectory) -> tuple[np.ndarray, dict[str, Any]]:
        calls = _tool_calls(traj)
        features: list[float] = []
        tool_counts = Counter(_tool_name(call) for call in calls)
        total_tools = max(len(calls), 1)
        features.extend(_safe_log1p(tool_counts[tool], math.log1p(16.0)) for tool in TOOL_TYPES)

        for position in range(self.swe_first_tools):
            one_hot = [0.0] * len(TOOL_TYPES)
            if position < len(calls):
                one_hot[TOOL_TYPES.index(_tool_name(calls[position]))] = 1.0
            features.extend(one_hot)

        paths = []
        for call in calls:
            paths.extend(_paths_from_tool_call(call))
        paths.extend(_paths_from_tool_outputs(traj))
        cluster_counts = Counter(_cluster_path(path) for path in paths)
        features.extend(min(cluster_counts[cluster], 8) / 8.0 for cluster in SWE_DIR_CLUSTERS)

        query_counter: Counter[str] = Counter()
        for call in calls:
            query_counter.update(_query_terms(call))
        query_features = np.zeros(self.swe_query_features, dtype=np.float32)
        if query_counter:
            norm = math.sqrt(sum(count * count for count in query_counter.values())) or 1.0
            for term, count in query_counter.items():
                query_features[_hash_index(term, self.swe_query_features)] += count / norm
        features.extend(query_features.tolist())

        nonempty_tool_outputs = [
            str(message.get("content", ""))
            for message in traj.messages
            if message.get("role") == "tool" and str(message.get("content", "")).strip()
        ]
        first_result_index = next(
            (
                index
                for index, message in enumerate(nonempty_tool_outputs)
                if "no matches" not in message.lower() and not message.lower().startswith("error:")
            ),
            len(nonempty_tool_outputs),
        )
        turns = max(len(traj.step_metadata), 1)
        completion_tokens = sum(
            int((step.get("usage") or {}).get("completion_tokens", 0))
            for step in traj.step_metadata
            if isinstance(step, dict)
        )
        prompt_tokens = sum(
            int((step.get("usage") or {}).get("prompt_tokens", 0))
            for step in traj.step_metadata
            if isinstance(step, dict)
        )
        assistant_messages = [message for message in traj.messages if message.get("role") == "assistant"]
        final_answer = str(assistant_messages[-1].get("content", "")) if assistant_messages else ""
        features.extend(
            [
                min(total_tools, 32) / 32.0,
                min(turns, 8) / 8.0,
                min(first_result_index, 16) / 16.0,
                min(len(nonempty_tool_outputs), 32) / 32.0,
                min(completion_tokens, 2048) / 2048.0,
                min(prompt_tokens, 8192) / 8192.0,
                min(len(final_answer), 2048) / 2048.0,
                min(traj.reward, 4.0) / 4.0,
                float(traj.success),
                min(traj.wall_clock_seconds, 120.0) / 120.0,
            ]
        )
        raw = {
            "tool_counts": dict(tool_counts),
            "directory_clusters": dict(cluster_counts),
            "query_terms": dict(query_counter.most_common(50)),
            "first_result_index": first_result_index,
            "num_turns": turns,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "feature_dim": len(features),
        }
        return np.asarray(features, dtype=np.float32), raw

    def _encode_math(self, traj: Trajectory) -> tuple[np.ndarray, dict[str, Any]]:
        assistant = _assistant_text(traj)
        text = assistant if assistant.strip() else _all_text(traj)
        lower = text.lower()
        tokens = _tokenize(text)
        features: list[float] = []

        marker_counts = {marker: lower.count(marker) for marker in MATH_REASONING_MARKERS}
        features.extend(_safe_log1p(marker_counts[marker], math.log1p(12.0)) for marker in MATH_REASONING_MARKERS)

        words = len(tokens)
        lines = text.count("\n") + 1 if text else 0
        features.extend(
            [
                min(len(text), 6000) / 6000.0,
                min(words, 1200) / 1200.0,
                min(lines, 120) / 120.0,
                min(text.count("."), 80) / 80.0,
                min(text.count(","), 120) / 120.0,
                min(text.count("\n\n"), 30) / 30.0,
                min(sum(ch.isdigit() for ch in text), 400) / 400.0,
                min(sum(ch in "+-*/=<>^" for ch in text), 400) / 400.0,
            ]
        )

        equation_features = {
            "dollar_pairs": text.count("$") // 2,
            "equals": text.count("="),
            "frac": lower.count("\\frac"),
            "sqrt": lower.count("\\sqrt"),
            "boxed": lower.count("\\boxed") + lower.count("boxed"),
            "align": lower.count("\\begin{align") + lower.count("\\begin{array"),
            "latex_commands": len(re.findall(r"\\[a-zA-Z]+", text)),
        }
        features.extend(_safe_log1p(value, math.log1p(30.0)) for value in equation_features.values())

        final_answer = _extract_final_answer(text)
        answer_features = np.zeros(self.math_answer_hash_features, dtype=np.float32)
        if final_answer:
            answer_features[_hash_index(_canonical_answer(final_answer), self.math_answer_hash_features)] = 1.0
        features.extend(answer_features.tolist())

        casework_terms = ("case", "cases", "suppose", "if", "otherwise", "without loss")
        features.extend(min(lower.count(term), 8) / 8.0 for term in casework_terms)

        method_counts = {
            group: sum(lower.count(term) for term in terms) for group, terms in MATH_METHOD_GROUPS.items()
        }
        features.extend(_safe_log1p(method_counts[group], math.log1p(10.0)) for group in MATH_METHOD_GROUPS)

        lexical_features = np.zeros(self.math_lexical_hash_features, dtype=np.float32)
        lexical_terms = [token for token in tokens if len(token) > 2]
        if lexical_terms:
            lexical_counts = Counter(lexical_terms)
            norm = math.sqrt(sum(count * count for count in lexical_counts.values())) or 1.0
            for term, count in lexical_counts.items():
                lexical_features[_hash_index(term, self.math_lexical_hash_features)] += count / norm
        features.extend(lexical_features.tolist())

        features.extend(
            [
                min(traj.reward, 1.0),
                float(traj.success),
                min(traj.wall_clock_seconds, 120.0) / 120.0,
                min(len(traj.step_metadata), 4) / 4.0,
                1.0 if final_answer else 0.0,
            ]
        )
        raw = {
            "marker_counts": marker_counts,
            "equation_features": equation_features,
            "final_answer_canonical": _canonical_answer(final_answer) if final_answer else "",
            "method_counts": method_counts,
            "feature_dim": len(features),
        }
        return np.asarray(features, dtype=np.float32), raw


def _extract_final_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        return boxed[-1]
    answer_match = re.findall(r"(?:final answer|answer)\s*[:is]*\s*(.+)", text, flags=re.IGNORECASE)
    if answer_match:
        return answer_match[-1].splitlines()[0]
    stripped = [line.strip() for line in text.splitlines() if line.strip()]
    return stripped[-1] if stripped else ""


def _canonical_answer(answer: str) -> str:
    return re.sub(r"\s+", "", answer.lower().strip().strip("."))
