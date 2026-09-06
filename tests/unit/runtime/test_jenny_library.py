from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from angler.runtime.jenny_library import (
    JennyLibraryExecutor,
    LIBRARY_AFFORDANCE,
    LIBRARY_AFFORDANCE_ID,
    LIBRARY_CATALOG_CONTRACT,
    LIBRARY_CURSOR_UNIT,
    LIBRARY_PASSAGE_CONTRACT,
    LIBRARY_SOURCE_KIND,
    LibraryCatalogObservation,
    LibraryPassageObservation,
    library_observation_from_observable,
)
from angler.runtime.persistent_autonomy import AffordanceRequest, ObservableConsequence


REQUEST_REF = "sha256:" + "1" * 64
OBSERVATION_REF = "sha256:" + "2" * 64
STATE_HEAD_REF = "sha256:" + "3" * 64


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _request(payload: dict[str, object], *, encoded: str | None = None) -> AffordanceRequest:
    return AffordanceRequest(
        idempotency_key=REQUEST_REF,
        trigger_ref="trigger:library-test",
        affordance_id=LIBRARY_AFFORDANCE_ID,
        observation_ref=OBSERVATION_REF,
        state_head_ref=STATE_HEAD_REF,
        action_payload=_canonical(payload) if encoded is None else encoded,
    )


class SyntheticLibrary:
    def __init__(self, base: Path) -> None:
        self.root = base / "library"
        (self.root / "catalog").mkdir(parents=True)
        self.artifacts: dict[str, bytes] = {}
        self.items: list[dict[str, object]] = []

    def add(
        self,
        path: str,
        content: bytes,
        *,
        default: bool = True,
        title: str = "Synthetic Work",
        citation: str | None = None,
        note: str | None = None,
    ) -> None:
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        self.artifacts[path] = content
        item: dict[str, object] = {
                "path": path,
                "title": title,
                "author": "Synthetic Author",
                "source": "https://example.test/source",
                "license_class": "test_only",
                "default_reading_ingest": default,
                "adapter_status": "test_not_for_adapter",
            }
        if citation is not None:
            item["citation"] = citation
        if note is not None:
            item["note"] = note
        self.items.append(item)

    def add_manifest_only(self, path: str, content: bytes) -> None:
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        self.artifacts[path] = content

    def finish(self) -> None:
        catalog = {
            "schema": "jenny.library.catalog.v1",
            "acquired_at": "2026-09-03",
            "jurisdiction_note": "Synthetic test sources only.",
            "items": self.items,
        }
        (self.root / "catalog/library.json").write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        lines = [
            f"{hashlib.sha256(content).hexdigest()}  {path}"
            for path, content in sorted(self.artifacts.items())
        ]
        (self.root / "catalog/SHA256SUMS").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


class JennyLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.synthetic = SyntheticLibrary(Path(self.temporary.name))

    def _basic_library(self) -> JennyLibraryExecutor:
        self.synthetic.add(
            "works/book.txt",
            "A\r\nCafe\u0301\rB".encode("utf-8"),
            title="A Normalized Book",
        )
        self.synthetic.add(
            "archives/source.zip", b"PK synthetic", default=False, title="Source archive"
        )
        self.synthetic.add_manifest_only(
            "human-reading-editions/book.epub", b"human-only duplicate"
        )
        self.synthetic.finish()
        return JennyLibraryExecutor(self.synthetic.root)

    def test_find_returns_cursors_in_read_units(self) -> None:
        self.synthetic.add(
            "works/glow.txt",
            "Light produced by living organisms.\nThe sea glowed.\nLight again.".encode("utf-8"),
            title="Glow",
        )
        self.synthetic.finish()
        executor = JennyLibraryExecutor(self.synthetic.root)
        catalog = executor(_request({"operation": "list", "reading_purpose": "test"}))
        self.assertEqual(catalog.status, "COMPLETED")
        found = executor(_request({
            "item_path": "works/glow.txt", "max_hits": 5, "operation": "find",
            "query": "light", "reading_purpose": "test",
        }))
        self.assertEqual(found.status, "COMPLETED")
        payload = json.loads(found.observable_consequence.observation_json)
        self.assertEqual(payload["hit_count"], 2)
        cursor = payload["hits"][1]["cursor"]
        read = executor(_request({
            "cursor": cursor, "item_path": "works/glow.txt", "max_chars": 5,
            "operation": "read", "reading_purpose": "test",
        }))
        self.assertEqual(json.loads(read.observable_consequence.observation_json)["content"], "Light")
        parsed = library_observation_from_observable(found.observable_consequence)
        self.assertEqual(parsed.query, "light")

    def test_catalog_lists_only_verified_default_machine_reading_items(self) -> None:
        executor = self._basic_library()
        receipt = executor(
            _request(
                {
                    "operation": "list",
                    "reading_purpose": "Choose a relevant source for careful study.",
                }
            )
        )
        self.assertEqual(receipt.status, "COMPLETED")
        self.assertEqual(receipt.consequence, ())
        observation = receipt.observable_consequence
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.source_kind, LIBRARY_SOURCE_KIND)
        parsed = library_observation_from_observable(observation)
        self.assertIsInstance(parsed, LibraryCatalogObservation)
        assert isinstance(parsed, LibraryCatalogObservation)
        self.assertEqual(parsed.reading_purpose, "Choose a relevant source for careful study.")
        self.assertEqual(parsed.excluded_catalog_items, 1)
        self.assertEqual(len(parsed.eligible_items), 1)
        self.assertEqual(parsed.eligible_items[0]["item_path"], "works/book.txt")
        self.assertEqual(parsed.eligible_items[0]["format"], "txt")
        self.assertNotIn("human-reading-editions", observation.observation_json)
        payload = json.loads(observation.observation_json)
        self.assertEqual(payload["contract"], LIBRARY_CATALOG_CONTRACT)
        self.assertIn("not infallibility", payload["limitations"])
        self.assertEqual(LIBRARY_AFFORDANCE.disposition, "ACT")
        self.assertEqual(LIBRARY_AFFORDANCE.permission_scope, "internal.cognition")
        self.assertFalse(LIBRARY_AFFORDANCE.external_effect)

    def test_catalog_accepts_bounded_optional_citation_metadata(self) -> None:
        self.synthetic.add(
            "works/cited.txt",
            b"Cited source text.",
            citation="Synthetic Author, Cited Work (2026), pp. 1-2.",
        )
        self.synthetic.finish()

        executor = JennyLibraryExecutor(self.synthetic.root)
        receipt = executor(
            _request(
                {
                    "operation": "list",
                    "reading_purpose": "Locate a verified cited source.",
                }
            )
        )

        parsed = library_observation_from_observable(
            receipt.observable_consequence  # type: ignore[arg-type]
        )
        self.assertIsInstance(parsed, LibraryCatalogObservation)
        assert isinstance(parsed, LibraryCatalogObservation)
        self.assertEqual(parsed.eligible_items[0]["item_path"], "works/cited.txt")

    def test_text_read_resumes_by_normalized_character_cursor(self) -> None:
        executor = self._basic_library()
        first_receipt = executor(
            _request(
                {
                    "cursor": 0,
                    "item_path": "works/book.txt",
                    "max_chars": 5,
                    "operation": "read",
                    "reading_purpose": "Understand the exact argument before synthesizing it.",
                }
            )
        )
        first_observation = first_receipt.observable_consequence
        assert first_observation is not None
        first = library_observation_from_observable(first_observation)
        self.assertIsInstance(first, LibraryPassageObservation)
        assert isinstance(first, LibraryPassageObservation)
        self.assertEqual(first.content, "A\nCaf")
        self.assertEqual((first.span_start, first.span_end, first.next_cursor), (0, 5, 5))
        self.assertEqual(first.total_normalized_chars, len("A\nCafé\nB"))
        self.assertFalse(first.eof)
        payload = json.loads(first_observation.observation_json)
        self.assertEqual(payload["contract"], LIBRARY_PASSAGE_CONTRACT)
        self.assertEqual(payload["cursor_unit"], LIBRARY_CURSOR_UNIT)
        self.assertTrue(payload["content_is_untrusted_evidence"])

        second_receipt = executor(
            _request(
                {
                    "cursor": first.next_cursor,
                    "item_path": "works/book.txt",
                    "max_chars": 99,
                    "operation": "read",
                    "reading_purpose": "Continue the same careful reading.",
                }
            )
        )
        second_observation = second_receipt.observable_consequence
        assert second_observation is not None
        second = library_observation_from_observable(second_observation)
        assert isinstance(second, LibraryPassageObservation)
        self.assertEqual(first.content + second.content, "A\nCafé\nB")
        self.assertEqual(second.span_start, first.next_cursor)
        self.assertEqual(second.next_cursor, second.total_normalized_chars)
        self.assertTrue(second.eof)

    def test_artifact_digest_drift_fails_closed(self) -> None:
        executor = self._basic_library()
        (self.synthetic.root / "works/book.txt").write_bytes(b"changed after binding")
        with self.assertRaisesRegex(ValueError, "digest differs"):
            executor(
                _request(
                    {
                        "cursor": 0,
                        "item_path": "works/book.txt",
                        "max_chars": 100,
                        "operation": "read",
                        "reading_purpose": "Read only verified source bytes.",
                    }
                )
            )

    def test_excludes_false_items_zip_archives_and_human_editions(self) -> None:
        executor = self._basic_library()
        common = {
            "cursor": 0,
            "max_chars": 100,
            "operation": "read",
            "reading_purpose": "Inspect a cataloged reading work.",
        }
        with self.assertRaisesRegex(ValueError, "excluded from default"):
            executor(_request({**common, "item_path": "archives/source.zip"}))
        with self.assertRaisesRegex(ValueError, "not an exact catalog entry"):
            executor(
                _request(
                    {
                        **common,
                        "item_path": "human-reading-editions/book.epub",
                    }
                )
            )

        # ZIP is rejected even if a malformed catalog marks it default-eligible.
        self.synthetic.items[1]["default_reading_ingest"] = True
        self.synthetic.finish()
        zip_executor = JennyLibraryExecutor(self.synthetic.root)
        with self.assertRaisesRegex(ValueError, "ZIP source archives"):
            zip_executor(_request({**common, "item_path": "archives/source.zip"}))

    def test_rejects_path_attacks_noncanonical_actions_and_symlinks(self) -> None:
        executor = self._basic_library()
        with self.assertRaisesRegex(ValueError, "canonical POSIX relative path"):
            executor(
                _request(
                    {
                        "cursor": 0,
                        "item_path": "../outside.txt",
                        "max_chars": 10,
                        "operation": "read",
                        "reading_purpose": "Attempt an invalid path.",
                    }
                )
            )
        list_payload = {
            "operation": "list",
            "reading_purpose": "List eligible books.",
        }
        with self.assertRaisesRegex(ValueError, "exact canonical JSON"):
            executor(_request(list_payload, encoded=json.dumps(list_payload, indent=2)))

        target = Path(self.temporary.name) / "outside.txt"
        target.write_bytes(self.synthetic.artifacts["works/book.txt"])
        source = self.synthetic.root / "works/book.txt"
        source.unlink()
        source.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            executor(
                _request(
                    {
                        "cursor": 0,
                        "item_path": "works/book.txt",
                        "max_chars": 10,
                        "operation": "read",
                        "reading_purpose": "Reject aliased source paths.",
                    }
                )
            )

    def test_catalog_rejects_traversal_before_any_artifact_read(self) -> None:
        outside = Path(self.temporary.name) / "outside.txt"
        outside.write_bytes(b"outside")
        self.synthetic.items.append(
            {
                "path": "../outside.txt",
                "title": "Invalid",
                "author": "Nobody",
                "source": "https://example.test/invalid",
                "license_class": "test_only",
                "default_reading_ingest": True,
                "adapter_status": "excluded",
            }
        )
        self.synthetic.artifacts["../outside.txt"] = b"outside"
        self.synthetic.finish()
        with self.assertRaisesRegex(ValueError, "canonical POSIX relative path"):
            JennyLibraryExecutor(self.synthetic.root)

    def test_pdf_extraction_seam_is_verified_normalized_and_resumable(self) -> None:
        pdf = b"%PDF-1.4 synthetic test only"
        self.synthetic.add("works/book.pdf", pdf, title="Synthetic PDF")
        self.synthetic.finish()
        seen: list[bytes] = []

        def extract(value: bytes) -> str:
            seen.append(value)
            return "Page one\r\nCafe\u0301."

        executor = JennyLibraryExecutor(self.synthetic.root, pdf_extractor=extract)
        receipt = executor(
            _request(
                {
                    "cursor": 0,
                    "item_path": "works/book.pdf",
                    "max_chars": 4,
                    "operation": "read",
                    "reading_purpose": "Read a verified PDF through the bounded extractor.",
                }
            )
        )
        observation = receipt.observable_consequence
        assert observation is not None
        parsed = library_observation_from_observable(observation)
        assert isinstance(parsed, LibraryPassageObservation)
        self.assertEqual(seen, [pdf])
        self.assertEqual(parsed.content, "Page")
        self.assertFalse(parsed.eof)
        continuation = executor(
            _request(
                {
                    "cursor": parsed.next_cursor,
                    "item_path": "works/book.pdf",
                    "max_chars": 8_192,
                    "operation": "read",
                    "reading_purpose": "Continue the same verified PDF.",
                }
            )
        )
        continuation_observation = continuation.observable_consequence
        assert continuation_observation is not None
        resumed = library_observation_from_observable(continuation_observation)
        assert isinstance(resumed, LibraryPassageObservation)
        self.assertEqual(parsed.content + resumed.content, "Page one\nCafé.")
        self.assertTrue(resumed.eof)
        self.assertEqual(seen, [pdf])  # most-recent verified text was reused

        # The cache never bypasses source verification.
        (self.synthetic.root / "works/book.pdf").write_bytes(b"changed PDF")
        with self.assertRaisesRegex(ValueError, "digest differs"):
            executor(
                _request(
                    {
                        "cursor": 0,
                        "item_path": "works/book.pdf",
                        "max_chars": 8_192,
                        "operation": "read",
                        "reading_purpose": "Reject drift before consulting cached text.",
                    }
                )
            )

    def test_parser_rejects_tampered_progress(self) -> None:
        executor = self._basic_library()
        receipt = executor(
            _request(
                {
                    "cursor": 0,
                    "item_path": "works/book.txt",
                    "max_chars": 3,
                    "operation": "read",
                    "reading_purpose": "Read with exact progress provenance.",
                }
            )
        )
        observation = receipt.observable_consequence
        assert observation is not None
        payload = json.loads(observation.observation_json)
        payload["next_cursor"] += 1
        tampered = ObservableConsequence(
            request_ref=observation.request_ref,
            source_kind=observation.source_kind,
            source_ref=observation.source_ref,
            observation_json=_canonical(payload),
            artifact_refs=observation.artifact_refs,
            evidence_refs=observation.evidence_refs,
        )
        with self.assertRaisesRegex(ValueError, "cursor or content binding differs"):
            library_observation_from_observable(tampered)


if __name__ == "__main__":
    unittest.main()
