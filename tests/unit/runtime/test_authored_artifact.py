from __future__ import annotations

import json
import socket
import unittest
from unittest import mock
import urllib.request

from angler.runtime.authored_artifact import (
    AUTHORED_ARTIFACT_AFFORDANCE,
    AUTHORED_ARTIFACT_AFFORDANCE_DESCRIPTION,
    AUTHORED_ARTIFACT_AFFORDANCE_ID,
    AUTHORED_ARTIFACT_OBSERVATION_CONTRACT,
    AUTHORED_ARTIFACT_PERMISSION_SCOPE,
    AUTHORED_ARTIFACT_PURPOSE,
    AUTHORED_ARTIFACT_VISIBILITY,
    AuthoredArtifactAction,
    AuthoredArtifactExecutor,
    action_from_observable_consequence,
    validate_artifact_successor,
)
from angler.runtime.persistent_autonomy import (
    AffordanceRequest,
    ObservableConsequence,
)


REQUEST_REF = "sha256:" + "1" * 64
OBSERVATION_REF = "sha256:" + "2" * 64
STATE_HEAD_REF = "sha256:" + "3" * 64
EVIDENCE_REF = "sha256:" + "4" * 64
EPISODE_REF = "sha256:" + "5" * 64
PREDECESSOR_REF = "sha256:" + "6" * 64


def _action(
    *,
    kind: str = "research synthesis",
    version: int = 1,
    supersedes_ref: str | None = None,
) -> AuthoredArtifactAction:
    return AuthoredArtifactAction(
        version=version,
        kind=kind,
        title="Attention as a limited workspace",
        body=(
            "The observations support a distinction between stored context and "
            "the smaller set of relationships currently available for comparison."
        ),
        purpose="Preserve a revisable synthesis for later comparison.",
        evidence_refs=(EVIDENCE_REF,),
        source_episode_refs=(EPISODE_REF,),
        supersedes_ref=supersedes_ref,
    )


def _request(
    action: AuthoredArtifactAction,
    *,
    affordance_id: str = AUTHORED_ARTIFACT_AFFORDANCE_ID,
) -> AffordanceRequest:
    return AffordanceRequest(
        idempotency_key=REQUEST_REF,
        trigger_ref="trigger:authored-artifact-test",
        affordance_id=affordance_id,
        observation_ref=OBSERVATION_REF,
        state_head_ref=STATE_HEAD_REF,
        action_payload=action.canonical_json(),
    )


class AuthoredArtifactTests(unittest.TestCase):
    def test_arbitrary_research_creative_and_diary_like_kinds_round_trip(self) -> None:
        kinds = (
            "research synthesis",
            "short speculative fiction",
            "private diary-like reflection",
            "a kind invented at runtime: luminous counterexample map",
        )
        for kind in kinds:
            with self.subTest(kind=kind):
                action = _action(kind=kind)
                self.assertEqual(
                    AuthoredArtifactAction.from_json(action.canonical_json()),
                    action,
                )
        self.assertIn(
            "no fixed topic or action taxonomy",
            AUTHORED_ARTIFACT_AFFORDANCE_DESCRIPTION,
        )
        self.assertNotIn("emotion", AUTHORED_ARTIFACT_AFFORDANCE_DESCRIPTION.lower())
        self.assertIn("finished work", AUTHORED_ARTIFACT_AFFORDANCE_DESCRIPTION)
        self.assertIn("not hidden reasoning", AUTHORED_ARTIFACT_AFFORDANCE_DESCRIPTION)

    def test_affordance_is_one_private_internal_cognition_boundary(self) -> None:
        self.assertEqual(
            AUTHORED_ARTIFACT_AFFORDANCE.affordance_id,
            "internal.authored-artifact",
        )
        self.assertEqual(
            AUTHORED_ARTIFACT_AFFORDANCE.permission_scope,
            AUTHORED_ARTIFACT_PERMISSION_SCOPE,
        )
        self.assertEqual(AUTHORED_ARTIFACT_PERMISSION_SCOPE, "internal.cognition")
        self.assertFalse(AUTHORED_ARTIFACT_AFFORDANCE.external_effect)
        self.assertEqual(AUTHORED_ARTIFACT_VISIBILITY, "PRIVATE")
        self.assertIn("not a feelings or consciousness claim", AUTHORED_ARTIFACT_PURPOSE)
        self.assertIn("neither reward nor permission", AUTHORED_ARTIFACT_PURPOSE)

    def test_executor_returns_content_addressed_immutable_observation(self) -> None:
        action = _action()
        receipt = AuthoredArtifactExecutor()(_request(action))
        self.assertEqual(receipt.status, "COMPLETED")
        self.assertEqual(receipt.consequence, ())
        observed = receipt.observable_consequence
        assert observed is not None
        self.assertEqual(observed.artifact_refs, (action.artifact_ref,))
        self.assertEqual(
            observed.evidence_refs,
            tuple(sorted((EVIDENCE_REF, EPISODE_REF))),
        )
        payload = json.loads(observed.observation_json)
        self.assertEqual(payload["contract"], AUTHORED_ARTIFACT_OBSERVATION_CONTRACT)
        self.assertEqual(payload["artifact_ref"], action.artifact_ref)
        self.assertEqual(payload["artifact"]["visibility"], "PRIVATE")
        self.assertEqual(payload["artifact"]["action"], action.canonical_payload())
        self.assertEqual(action_from_observable_consequence(observed), action)

    def test_content_and_receipt_hashes_are_deterministic(self) -> None:
        first_action = _action()
        second_action = AuthoredArtifactAction.from_json(first_action.canonical_json())
        first = AuthoredArtifactExecutor()(_request(first_action))
        second = AuthoredArtifactExecutor()(_request(second_action))
        self.assertEqual(first_action.artifact_ref, second_action.artifact_ref)
        self.assertEqual(first.observable_consequence, second.observable_consequence)
        self.assertEqual(first.receipt_ref, second.receipt_ref)
        self.assertRegex(first_action.artifact_ref, r"^sha256:[0-9a-f]{64}$")

    def test_version_and_supersedes_lineage_is_explicit(self) -> None:
        predecessor = _action()
        successor = _action(version=2, supersedes_ref=predecessor.artifact_ref)
        receipt = AuthoredArtifactExecutor()(_request(successor))
        observed = receipt.observable_consequence
        assert observed is not None
        self.assertIn(predecessor.artifact_ref, observed.evidence_refs)
        self.assertEqual(
            action_from_observable_consequence(observed).supersedes_ref,
            predecessor.artifact_ref,
        )
        validate_artifact_successor(predecessor, successor)
        with self.assertRaisesRegex(ValueError, "predecessor artifact_ref"):
            validate_artifact_successor(
                predecessor,
                _action(version=2, supersedes_ref=PREDECESSOR_REF),
            )
        version_two = _action(version=2, supersedes_ref=predecessor.artifact_ref)
        skipped = _action(version=4, supersedes_ref=version_two.artifact_ref)
        with self.assertRaisesRegex(ValueError, "increment predecessor by one"):
            validate_artifact_successor(version_two, skipped)
        with self.assertRaisesRegex(ValueError, "version 1"):
            _action(version=1, supersedes_ref=PREDECESSOR_REF)
        with self.assertRaisesRegex(ValueError, "later artifact versions"):
            _action(version=2)

    def test_body_accepts_exactly_twelve_kib_and_rejects_more(self) -> None:
        base = _action()
        accepted = AuthoredArtifactAction(
            version=1,
            kind=base.kind,
            title=base.title,
            body="x" * (12 * 1024),
            purpose=base.purpose,
            evidence_refs=(),
            source_episode_refs=(),
        )
        self.assertEqual(len(accepted.body.encode("utf-8")), 12 * 1024)
        accepted.canonical_json()
        observed = AuthoredArtifactExecutor()(
            _request(accepted)
        ).observable_consequence
        assert observed is not None
        self.assertEqual(action_from_observable_consequence(observed), accepted)
        with self.assertRaisesRegex(ValueError, "body exceeds"):
            AuthoredArtifactAction(
                version=1,
                kind=base.kind,
                title=base.title,
                body="x" * (12 * 1024 + 1),
                purpose=base.purpose,
                evidence_refs=(),
                source_episode_refs=(),
            )

    def test_malformed_noncanonical_and_extra_fields_fail_closed(self) -> None:
        action = _action()
        payload = action.canonical_payload()
        with self.assertRaisesRegex(ValueError, "exact canonical JSON"):
            AuthoredArtifactAction.from_json(json.dumps(payload, sort_keys=True))
        with self.assertRaisesRegex(ValueError, "fields differ"):
            AuthoredArtifactAction.from_json(
                json.dumps(
                    {**payload, "emotion": "programmed"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        duplicate_key = action.canonical_json().replace(
            '{"body":',
            '{"body":"duplicate","body":',
            1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            AuthoredArtifactAction.from_json(duplicate_key)

    def test_invalid_unsorted_duplicate_and_excess_references_fail_closed(self) -> None:
        invalid_sets = (
            ("sha256:" + "A" * 64,),
            (EVIDENCE_REF, EVIDENCE_REF),
            (EPISODE_REF, EVIDENCE_REF),
            tuple("sha256:" + f"{index:064x}" for index in range(7)),
        )
        for refs in invalid_sets:
            with self.subTest(refs=refs):
                with self.assertRaises((TypeError, ValueError)):
                    AuthoredArtifactAction(
                        version=1,
                        kind="unconstrained kind",
                        title="Title",
                        body="Body",
                        purpose="Purpose",
                        evidence_refs=refs,
                        source_episode_refs=(),
                    )
        payload = _action().canonical_payload()
        payload["evidence_refs"] = "not-an-array"
        with self.assertRaisesRegex(TypeError, "JSON arrays"):
            AuthoredArtifactAction.from_json(
                json.dumps(payload, separators=(",", ":"), sort_keys=True)
            )

    def test_binding_and_content_tampering_fail_closed(self) -> None:
        observed = AuthoredArtifactExecutor()(
            _request(_action())
        ).observable_consequence
        assert observed is not None
        payload = json.loads(observed.observation_json)
        payload["artifact_ref"] = "sha256:" + "f" * 64
        tampered = ObservableConsequence(
            request_ref=observed.request_ref,
            source_kind=observed.source_kind,
            source_ref=observed.source_ref,
            observation_json=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            artifact_refs=observed.artifact_refs,
            evidence_refs=observed.evidence_refs,
        )
        with self.assertRaisesRegex(ValueError, "content reference differs"):
            action_from_observable_consequence(tampered)
        with self.assertRaisesRegex(ValueError, "different affordance"):
            AuthoredArtifactExecutor()(
                _request(_action(), affordance_id="internal.other")
            )

    def test_execution_performs_no_filesystem_or_network_io(self) -> None:
        action = _action(kind="runtime-invented observation form")
        executor = AuthoredArtifactExecutor()
        with (
            mock.patch("builtins.open", side_effect=AssertionError("filesystem I/O")) as opened,
            mock.patch.object(socket, "socket", side_effect=AssertionError("socket I/O")) as sock,
            mock.patch.object(
                urllib.request,
                "urlopen",
                side_effect=AssertionError("network I/O"),
            ) as urlopen,
        ):
            receipt = executor(_request(action))
        self.assertEqual(receipt.status, "COMPLETED")
        opened.assert_not_called()
        sock.assert_not_called()
        urlopen.assert_not_called()

class PrivateWorkDeclarationTest(unittest.TestCase):
    def test_only_her_explicit_prefix_marks_private(self):
        from angler.runtime.authored_artifact import is_private_work
        for kind in ("private", "Private note", "private-thought", "private/letter", "PRIVATE: draft"):
            self.assertTrue(is_private_work(kind), kind)
        for kind in ("privateer log", "note", "story", "letter to Becca", "", None, 3):
            self.assertFalse(is_private_work(kind), repr(kind))


if __name__ == "__main__":
    unittest.main()
