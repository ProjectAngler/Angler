"""A narrow, lazy boundary around Cognee's public memory API.

Cognee is optional and never becomes Angler's source of truth.  The adapter
uses raw CHUNKS retrieval so Cognee proposes evidence candidates rather than
generating an answer for the procedural core.
"""

from __future__ import annotations

import os
from typing import Any

from .contracts import MemoryHit


class CogneeConfigurationError(RuntimeError):
    pass


def _cognee_result_entries(results: Any) -> tuple[Any, ...]:
    """Normalize Cognee's public CHUNKS compatibility envelopes.

    Cognee 1.5 returns a flat list of chunk dictionaries when backend access
    control is disabled.  With access control it returns one dataset envelope
    whose ``search_result`` contains that same list.  Older/test doubles may
    expose normalized result objects instead.  This boundary deliberately
    supports only those public shapes rather than guessing recursively.
    """

    if not isinstance(results, (list, tuple)):
        return ()
    entries: list[Any] = []
    for result in results:
        if isinstance(result, dict) and isinstance(result.get("search_result"), list):
            entries.extend(result["search_result"])
        elif isinstance(result, (list, tuple)):
            entries.extend(result)
        else:
            entries.append(result)
    return tuple(entries)


def _chunk_hit(result: Any) -> MemoryHit | None:
    if isinstance(result, dict):
        text = result.get("text")
        score = result.get("score")
        raw_ref = result.get("id") or result.get("document_id")
    else:
        text = getattr(result, "text", None)
        score = getattr(result, "score", None)
        metadata = getattr(result, "metadata", None)
        raw_ref = None
        if isinstance(metadata, dict):
            raw_ref = metadata.get("chunk_id") or metadata.get("data_id")
    if not isinstance(text, str):
        return None
    if score is not None and not isinstance(score, (int, float)):
        score = None
    return MemoryHit(
        document=text,
        score=None if score is None else float(score),
        backend_ref=None if raw_ref is None else str(raw_ref),
    )


class CogneeProjectionBackend:
    """Dataset-scoped Cognee backend with explicit model-call authority."""

    def __init__(
        self,
        dataset_name: str,
        *,
        cognee_module: Any | None = None,
        local_models_configured: bool = False,
        external_model_calls_authorized: bool = False,
        telemetry_authorized: bool = False,
    ) -> None:
        if type(dataset_name) is not str or not dataset_name.strip():
            raise ValueError("dataset_name must be non-empty text")
        self.dataset_name = dataset_name
        self._cognee = cognee_module
        self._local_models_configured = local_models_configured
        self._external_model_calls_authorized = external_model_calls_authorized
        self._telemetry_authorized = telemetry_authorized

    def _module(self) -> Any:
        if self._cognee is None:
            try:
                import cognee
            except ImportError as exc:
                raise CogneeConfigurationError(
                    "Cognee is optional; install project-angler[memory]"
                ) from exc
            self._cognee = cognee
        return self._cognee

    def _require_model_authority(self) -> None:
        if not (
            self._local_models_configured
            or self._external_model_calls_authorized
        ):
            raise CogneeConfigurationError(
                "Cognee ingestion may invoke configured embedding/LLM providers; "
                "declare local_models_configured or explicitly authorize external calls"
            )

    def _require_telemetry_authority(self) -> None:
        if not os.getenv("TELEMETRY_DISABLED") and not self._telemetry_authorized:
            raise CogneeConfigurationError(
                "Cognee telemetry is enabled by default; set TELEMETRY_DISABLED=1 "
                "or explicitly authorize telemetry"
            )

    async def remember(self, document: str) -> str | None:
        cognee = self._module()
        self._require_telemetry_authority()
        self._require_model_authority()
        result = await cognee.remember(
            document,
            dataset_name=self.dataset_name,
            self_improvement=False,
        )
        status = getattr(result, "status", None)
        if status not in (None, "completed"):
            raise RuntimeError(f"Cognee remember did not complete: {status}")
        items = getattr(result, "items", None) or ()
        if items and isinstance(items[0], dict):
            identifier = items[0].get("id")
            return None if identifier is None else str(identifier)
        return None

    async def remember_many(self, documents: tuple[str, ...]) -> tuple[str, ...]:
        """Index a bounded projection batch through one Cognee pipeline run."""

        if not documents or any(type(item) is not str or not item for item in documents):
            raise ValueError("documents must be a non-empty tuple of text")
        cognee = self._module()
        self._require_telemetry_authority()
        self._require_model_authority()
        result = await cognee.remember(
            list(documents),
            dataset_name=self.dataset_name,
            self_improvement=False,
        )
        status = getattr(result, "status", None)
        if status not in (None, "completed"):
            raise RuntimeError(f"Cognee remember did not complete: {status}")
        identifiers = []
        for item in getattr(result, "items", None) or ():
            if isinstance(item, dict) and item.get("id") is not None:
                identifiers.append(str(item["id"]))
        return tuple(identifiers)

    async def search(self, query: str, *, limit: int) -> tuple[MemoryHit, ...]:
        if type(query) is not str or not query.strip():
            raise ValueError("query must be non-empty text")
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        cognee = self._module()
        self._require_telemetry_authority()
        self._require_model_authority()
        results = await cognee.search(
            query_text=query,
            query_type=cognee.SearchType.CHUNKS,
            datasets=[self.dataset_name],
            top_k=limit,
        )
        hits = []
        for result in _cognee_result_entries(results):
            hit = _chunk_hit(result)
            if hit is not None:
                hits.append(hit)
        return tuple(hits)

    async def forget_dataset(self) -> None:
        cognee = self._module()
        self._require_telemetry_authority()
        await cognee.forget(dataset=self.dataset_name)


__all__ = ["CogneeConfigurationError", "CogneeProjectionBackend"]
