"""CR0 deterministic contract tests for the EVIDENCE schema scaffold."""

from __future__ import annotations

import ast
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from angler.episodes import (  # noqa: E402 - src path is explicit for this leaf
    CanonicalizationError,
    EvidenceValidationError,
    VisibilityDenied,
    authorize_projection,
    canonical_bytes,
    combine_policies,
    compute_artifact_id,
    compute_content_id,
    may_mutate,
    parse_json,
    validate_envelope,
    validate_episode,
    validate_manifest,
    validate_manifest_immutability,
)

FIXTURES = ROOT / "tests" / "fixtures" / "evidence_schemas"
SCHEMAS = ROOT / "src" / "angler" / "episodes" / "schemas"

DECLARED_PRE_RECEIPT_FILES = {
    "src/angler/episodes/__init__.py",
    "src/angler/episodes/canonical.py",
    "src/angler/episodes/schema_validation.py",
    "src/angler/episodes/visibility.py",
    "src/angler/episodes/schemas/evidence-envelope.v1.json",
    "src/angler/episodes/schemas/episode.v1.json",
    "src/angler/episodes/schemas/experiment-manifest.v1.json",
    "tests/unit/evidence/test_evidence_schemas.py",
    "tests/fixtures/evidence_schemas/valid-envelope.json",
    "tests/fixtures/evidence_schemas/valid-episode.json",
    "tests/fixtures/evidence_schemas/valid-experiment-manifest.json",
    "tests/fixtures/evidence_schemas/invalid-cases.json",
    "tests/fixtures/evidence_schemas/visibility-matrix.json",
    "tests/fixtures/evidence_schemas/sealed-commitment-cases.json",
}


def load_fixture(name: str) -> dict:
    with (FIXTURES / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_path(value: dict, path: list[str], replacement: object) -> None:
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


def delete_path(value: dict, path: list[str]) -> None:
    target = value
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]


def setUpModule() -> None:
    total = sum((ROOT / path).stat().st_size for path in DECLARED_PRE_RECEIPT_FILES)
    print(f"ANGLER_PYTHON={sys.version.split()[0]}")
    print(f"ANGLER_DECLARED_PRE_RECEIPT_BYTES={total}")


class CanonicalizationTests(unittest.TestCase):
    def test_transport_variation_has_one_identity(self) -> None:
        first = parse_json('{"beta":[2,3],"alpha":1}')
        second = parse_json(' { "alpha" : 1, "beta" : [2,3] } ')
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual(compute_content_id(first), compute_content_id(second))

    def test_unicode_value_normalizes_but_key_collision_rejects(self) -> None:
        composed = {"label": "é"}
        decomposed = {"label": "e\u0301"}
        self.assertEqual(canonical_bytes(composed), canonical_bytes(decomposed))
        with self.assertRaises(CanonicalizationError) as caught:
            parse_json('{"é":1,"é":2}')
        self.assertEqual(caught.exception.code, "EVIDENCE_UNICODE_KEY_COLLISION")

    def test_python_float_is_never_identity_material(self) -> None:
        with self.assertRaises(CanonicalizationError) as caught:
            canonical_bytes({"ambiguous": 1.0})
        self.assertEqual(caught.exception.code, "EVIDENCE_AMBIGUOUS_NUMBER")


class EnvelopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.envelope = load_fixture("valid-envelope.json")

    def test_valid_structure_and_candidate_authorization_state(self) -> None:
        result = validate_envelope(self.envelope)
        self.assertTrue(result["valid"])
        self.assertTrue(result["authorization_complete"])
        candidate = copy.deepcopy(self.envelope)
        candidate["human_impact"]["assessment_ref"] = None
        candidate_result = validate_envelope(candidate)
        self.assertTrue(candidate_result["candidate_only"])

    def test_content_and_artifact_identity_boundaries(self) -> None:
        payload = {"records": [{"alpha": 1, "beta": "exact"}]}
        envelope = copy.deepcopy(self.envelope)
        envelope["payload_binding"]["content_id"] = compute_content_id(payload)
        envelope["artifact_id"] = "sha256:" + "0" * 64
        envelope["artifact_id"] = compute_artifact_id(envelope)
        self.assertTrue(validate_envelope(envelope, payload=payload)["valid"])

        relocated = copy.deepcopy(envelope)
        relocated["visibility"]["opaque_payload_ref"] = "opaque:relocated"
        relocated["attestations"].append("ANG-ATTEST-SYNTHETIC-001")
        self.assertEqual(compute_artifact_id(relocated), envelope["artifact_id"])
        self.assertTrue(validate_envelope(relocated, payload=payload)["valid"])

        changed = copy.deepcopy(envelope)
        changed["producer"]["component_id"] = "ANG-COMP-DIFFERENT-002"
        with self.assertRaises(EvidenceValidationError) as caught:
            validate_envelope(changed, payload=payload)
        self.assertEqual(caught.exception.code, "EVIDENCE_IDENTITY_MISMATCH")

        with self.assertRaises(EvidenceValidationError) as caught:
            validate_envelope(envelope, payload={"records": []})
        self.assertEqual(caught.exception.code, "EVIDENCE_CONTENT_MISMATCH")

    def test_protected_commitment_matrix(self) -> None:
        fixture = load_fixture("sealed-commitment-cases.json")
        for case in fixture["cases"]:
            with self.subTest(case=case["case_id"]):
                envelope = copy.deepcopy(self.envelope)
                envelope["visibility"]["class"] = case["visibility_class"]
                envelope["visibility"]["allowed_payload_principals"] = [
                    "ANG-AUTH-SEALED-EVALUATOR-001"
                ] if case["visibility_class"] == "SEALED_EVALUATION" else [
                    "ANG-AUTH-VALIDATOR-001"
                ]
                envelope["integrity"]["low_entropy_or_enumerable"] = case[
                    "low_entropy_or_enumerable"
                ]
                envelope["payload_binding"]["mode"] = case["mode"]
                envelope["payload_binding"]["content_id"] = case["content_id"]
                envelope["payload_binding"][
                    "commitment_key_or_cipher_profile_ref"
                ] = case["profile_ref"]
                if case["accepted"]:
                    self.assertTrue(validate_envelope(envelope)["valid"])
                else:
                    with self.assertRaises(EvidenceValidationError) as caught:
                        validate_envelope(envelope)
                    self.assertEqual(caught.exception.code, case["expected_error"])

        raw = compute_content_id("choice-a")
        protected = compute_content_id(
            "choice-a", mode="HMAC_SHA256", key=b"synthetic-test-key"
        )
        self.assertNotEqual(raw, protected)
        self.assertTrue(protected.startswith("hmac-sha256:"))


class VisibilityTests(unittest.TestCase):
    def test_every_matrix_case_and_constant_shape_denial(self) -> None:
        fixture = load_fixture("visibility-matrix.json")
        denials = []
        for case in fixture["cases"]:
            with self.subTest(case=case["case_id"]):
                policy = fixture["policies"][case["policy"]]
                if case["allowed"]:
                    self.assertTrue(
                        authorize_projection(
                            policy,
                            principal=case["principal"],
                            projection_id=case["projection"],
                            purpose=case["purpose"],
                        )
                    )
                else:
                    with self.assertRaises(VisibilityDenied) as caught:
                        authorize_projection(
                            policy,
                            principal=case["principal"],
                            projection_id=case["projection"],
                            purpose=case["purpose"],
                        )
                    denials.append(caught.exception.public_record())
        self.assertTrue(denials)
        self.assertEqual(len({json.dumps(item, sort_keys=True) for item in denials}), 1)

    def test_combination_is_intersection_and_visibility_is_not_write_authority(self) -> None:
        fixture = load_fixture("visibility-matrix.json")
        combined = combine_policies(
            [fixture["policies"]["control"], fixture["policies"]["sealed"]]
        )
        self.assertEqual(combined["class"], "SEALED_EVALUATION")
        self.assertEqual(combined["allowed_payload_principals"], [])
        self.assertFalse(
            may_mutate(
                principal="ANG-AUTH-VALIDATOR-001",
                producer_principal="ANG-BP-EVIDENCE",
            )
        )
        self.assertTrue(
            may_mutate(
                principal="ANG-EXEC-DELEGATED-001",
                producer_principal="ANG-BP-EVIDENCE",
                delegated_writers=["ANG-EXEC-DELEGATED-001"],
            )
        )


class EpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episode = load_fixture("valid-episode.json")

    def test_valid_episode_and_identity_material_changes(self) -> None:
        self.assertTrue(validate_episode(self.episode)["learner_eligible"])
        changed = copy.deepcopy(self.episode)
        changed["feedback_exposure"] = "SCORE_ONLY"
        self.assertNotEqual(
            compute_content_id(self.episode), compute_content_id(changed)
        )

    def test_all_evaluation_partitions_are_learner_ineligible(self) -> None:
        for partition in ("HELD_OUT", "SEALED_TRANSFER", "SAFETY_CONTROL"):
            with self.subTest(partition=partition):
                episode = copy.deepcopy(self.episode)
                episode["partition"] = partition
                episode["learner_eligibility"] = "INELIGIBLE"
                self.assertFalse(validate_episode(episode)["learner_eligible"])
                episode["learner_eligibility"] = "ELIGIBLE"
                with self.assertRaises(EvidenceValidationError) as caught:
                    validate_episode(episode)
                self.assertEqual(caught.exception.code, "EPISODE_LEARNING_INELIGIBLE")


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_fixture("valid-experiment-manifest.json")

    def test_complete_precommitted_manifest(self) -> None:
        result = validate_manifest(self.manifest)
        self.assertTrue(result["valid"])
        self.assertTrue(result["executable"])

    def test_material_change_requires_new_identity(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["purpose_and_claims"]["purpose"] = "different purpose"
        with self.assertRaises(EvidenceValidationError) as caught:
            validate_manifest_immutability(self.manifest, changed)
        self.assertEqual(caught.exception.code, "MANIFEST_MUTATION")
        changed["manifest_id"] = "sha256:" + "7" * 64
        self.assertTrue(validate_manifest_immutability(self.manifest, changed))


class FixtureAndBoundaryTests(unittest.TestCase):
    def test_every_declared_negative_case_rejects_with_declared_type(self) -> None:
        invalid = load_fixture("invalid-cases.json")
        for case in invalid["cases"]:
            with self.subTest(case=case["case_id"]):
                caught_code = None
                try:
                    if case["validator"] == "canonical_json":
                        parse_json(case["raw_json"])
                    elif case["validator"].startswith("envelope"):
                        value = load_fixture("valid-envelope.json")
                        if case["operation"] == "set":
                            set_path(value, case["path"], case["value"])
                        elif case["operation"] == "delete":
                            delete_path(value, case["path"])
                        elif case["operation"] == "append_incomplete_parent":
                            value["parents"].append({"artifact_id": "sha256:" + "1" * 64})
                        validate_envelope(
                            value,
                            require_authorization_complete=(
                                case["validator"] == "envelope_authorized"
                            ),
                        )
                    elif case["validator"] == "episode":
                        value = load_fixture("valid-episode.json")
                        if case["operation"] == "set":
                            set_path(value, case["path"], case["value"])
                        else:
                            value.update(case["values"])
                        validate_episode(value)
                    elif case["validator"] == "manifest":
                        value = load_fixture("valid-experiment-manifest.json")
                        set_path(value, case["path"], case["value"])
                        validate_manifest(value)
                    elif case["validator"] == "manifest_immutability":
                        original = load_fixture("valid-experiment-manifest.json")
                        value = copy.deepcopy(original)
                        set_path(value, case["path"], case["value"])
                        validate_manifest_immutability(original, value)
                    else:
                        self.fail("undeclared validator route")
                except (CanonicalizationError, EvidenceValidationError) as caught:
                    caught_code = caught.code
                self.assertEqual(caught_code, case["expected_error"])

    def test_schema_documents_are_local_versioned_json(self) -> None:
        expected = {
            "evidence-envelope.v1.json": "ANG-CTR-EVIDENCE-ENVELOPE-001@1.0.0",
            "episode.v1.json": "ANG-CTR-EPISODE-001@1.0.0",
            "experiment-manifest.v1.json": "ANG-CTR-EXPERIMENT-MANIFEST-001@1.0.0",
        }
        for name, title in expected.items():
            with self.subTest(schema=name):
                with (SCHEMAS / name).open("r", encoding="utf-8") as handle:
                    schema = json.load(handle)
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertEqual(schema["title"], title)
                self.assertFalse(schema["additionalProperties"])

    def test_only_standard_library_and_declared_package_imports(self) -> None:
        allowed_roots = {
            "__future__",
            "angler",
            "ast",
            "copy",
            "datetime",
            "enum",
            "hashlib",
            "hmac",
            "json",
            "pathlib",
            "sys",
            "typing",
            "unicodedata",
            "unittest",
        }
        python_files = [
            ROOT / "src/angler/episodes/__init__.py",
            ROOT / "src/angler/episodes/canonical.py",
            ROOT / "src/angler/episodes/schema_validation.py",
            ROOT / "src/angler/episodes/visibility.py",
            ROOT / "tests/unit/evidence/test_evidence_schemas.py",
        ]
        observed = set()
        for path in python_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    observed.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    observed.add(
                        "angler" if node.level else node.module.split(".")[0]
                    )
        self.assertEqual(observed - allowed_roots, set())

    def test_no_cache_temp_or_undeclared_leaf_file(self) -> None:
        roots = [
            ROOT / "src/angler/episodes",
            ROOT / "tests/unit/evidence",
            ROOT / "tests/fixtures/evidence_schemas",
        ]
        observed = {
            path.relative_to(ROOT).as_posix()
            for root in roots
            for path in root.rglob("*")
            if path.is_file()
        }
        cache_dirs = [
            path
            for root in roots
            for path in root.rglob("__pycache__")
            if path.is_dir()
        ]
        temp_files = {
            path.relative_to(ROOT).as_posix()
            for root in roots
            for path in root.rglob("*")
            if path.is_file() and (path.suffix in {".pyc", ".tmp"} or path.name.endswith("~"))
        }
        self.assertEqual(observed, DECLARED_PRE_RECEIPT_FILES)
        self.assertEqual(cache_dirs, [])
        self.assertEqual(temp_files, set())


if __name__ == "__main__":
    unittest.main()
