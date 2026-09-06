"""Canonical acquisition renderers and rebuildable situated projection.

The schema-v3 cognitive acquisition stream is the sole authority used here.
Backends receive only a disposable reference projection, and every search hit
is rejoined to canonical acquisition bytes before any record is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import math
import re
from typing import Protocol, Sequence, runtime_checkable

from angler.cognition.contracts import (
    CognitiveMemoryKind,
    CognitiveMemoryRecord,
    CognitiveRelation,
    EpistemicStatus,
    RelationType,
)
from angler.cognition.prospective_origin import (
    ProspectiveDynamicsBatch,
    ProspectiveResolution,
    ResolutionDisposition,
)

from .cognitive_acquisition import (
    CognitiveAcquisition,
    CognitiveGraphProjectionV2,
)
from .moving_origin import MovingOriginIndex, TemporalPosition


_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_PRODUCER_ID = "angler.prospective-cycle"
_LOCAL_BRANCH_LIMIT = 64
MAX_ACQUISITION_NEIGHBORS = 384
MAX_ACQUISITION_PROVENANCE_REFS = 384
_ORDINARY_EPISTEMIC_STATUSES = frozenset(
    {EpistemicStatus.OBSERVED, EpistemicStatus.VALIDATED}
)


def _ref(value: object, label: str) -> str:
    if type(value) is not str or _REF.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256 reference")
    return value


def _text(value: object, label: str, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{label} must be non-empty text of at most {maximum} characters"
        )
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace")
    return value


def _ordinal(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _canonical_batch(value: object) -> ProspectiveDynamicsBatch:
    if type(value) is not ProspectiveDynamicsBatch:
        raise TypeError("batch must be an exact ProspectiveDynamicsBatch")
    if len(value.branches) > _LOCAL_BRANCH_LIMIT:
        raise ValueError("batch exceeds this leaf's 64-branch execution ceiling")
    encoded = value.canonical_bytes()
    restored = ProspectiveDynamicsBatch.from_json(encoded)
    if restored.canonical_bytes() != encoded or restored.batch_ref != value.batch_ref:
        raise ValueError("batch does not round-trip through its canonical contract")
    return restored


def _canonical_resolution(value: object) -> ProspectiveResolution:
    if type(value) is not ProspectiveResolution:
        raise TypeError("resolution must be an exact ProspectiveResolution")
    if len(value.reservation.batch.branches) > _LOCAL_BRANCH_LIMIT:
        raise ValueError(
            "resolution batch exceeds this leaf's 64-branch execution ceiling"
        )
    encoded = value.canonical_bytes()
    restored = ProspectiveResolution.from_json(encoded)
    if (
        restored.canonical_bytes() != encoded
        or restored.resolution_ref != value.resolution_ref
    ):
        raise ValueError("resolution does not round-trip through its canonical contract")
    return restored


def _relations(
    pairs: set[tuple[RelationType, str]],
) -> tuple[CognitiveRelation, ...]:
    return tuple(
        CognitiveRelation(relation_type, target_ref)
        for relation_type, target_ref in sorted(
            pairs, key=lambda item: (item[0].value, item[1])
        )
    )


def _assert_record_neighbor_bound(record: CognitiveMemoryRecord) -> None:
    total = (
        len(record.relations)
        + len(record.supersedes_refs)
        + len(record.tombstone_refs)
    )
    if total > MAX_ACQUISITION_NEIGHBORS:
        raise ValueError("canonical record exceeds the 384-neighbor local ceiling")


def _assert_record_provenance_bound(record: CognitiveMemoryRecord) -> None:
    if len(record.provenance_refs) > MAX_ACQUISITION_PROVENANCE_REFS:
        raise ValueError(
            "canonical record exceeds the 384-provenance-reference local ceiling"
        )


def prospective_batch_record(
    batch: ProspectiveDynamicsBatch, ordinal: int
) -> CognitiveMemoryRecord:
    """Render one bounded, explicitly hypothetical prospective batch record."""

    batch = _canonical_batch(batch)
    ordinal = _ordinal(ordinal, "ordinal")
    selected = batch.selected_branch
    content = (
        "Prospective dynamics batch\n"
        "Hypothetical status: PROPOSED (not observed, validated, or authorized).\n"
        "Request:\n"
        f"{batch.request[:4_096]}\n"
        f"Branch count: {len(batch.branches)}\n"
        "Selected proposed trace:\n"
        f"{selected.candidate_trace[:4_096]}"
    )
    provenance = tuple(
        sorted(
            {
                batch.batch_ref,
                batch.parent_lineage.lineage_ref,
                batch.decision_evidence_ref,
                *batch.recalled_refs,
            }
        )
    )
    relation_pairs = {
        (RelationType.PREDICTS, branch.branch_ref) for branch in batch.branches
    }
    relation_pairs.update(
        (RelationType.ABOUT, evidence_ref)
        for evidence_ref in {batch.decision_evidence_ref, *batch.recalled_refs}
    )
    return CognitiveMemoryRecord(
        kind=CognitiveMemoryKind.COUNTERFACTUAL,
        epistemic_status=EpistemicStatus.PROPOSED,
        content=content,
        provenance_refs=provenance,
        visibility="LEARNER_VISIBLE",
        producer_id=_PRODUCER_ID,
        producer_checkpoint_ref=batch.dynamics_checkpoint_ref,
        competence_ref=batch.parent_lineage.competence_state_digest,
        acquired_ordinal=ordinal,
        world_valid_from=None,
        world_valid_until=None,
        relations=_relations(relation_pairs),
        confidence_ppm=None,
    )


def prospective_resolution_record(
    resolution: ProspectiveResolution, ordinal: int
) -> CognitiveMemoryRecord:
    """Render an observed lifecycle fact without inventing outcome truth."""

    resolution = _canonical_resolution(resolution)
    ordinal = _ordinal(ordinal, "ordinal")
    selected = resolution.reservation.batch.selected_branch
    receipt = resolution.execution_receipt
    execution_status = "ABSENT" if receipt is None else receipt.status
    lines = [
        "Prospective lifecycle resolution",
        f"Disposition: {resolution.disposition.value}",
        "Selected trace:",
        selected.candidate_trace[:4_096],
        f"Executor status: {execution_status}",
    ]
    if resolution.disposition is ResolutionDisposition.OBSERVED:
        feedback = resolution.objective_feedback
        if feedback is None or resolution.child_lineage is None:
            raise ValueError("observed resolution lacks canonical outcome material")
        lines.extend(
            (
                f"Objective outcome: {feedback.outcome}",
                "Objective feedback:",
                feedback.feedback_text[:2_048],
                "Child competence update: PRESENT",
            )
        )
        competence_ref = resolution.child_lineage.competence_state_digest
    else:
        lines.extend(
            (
                "Objective outcome: ABSENT",
                "Child competence update: ABSENT",
            )
        )
        competence_ref = resolution.parent_lineage.competence_state_digest

    cited_refs = {
        resolution.resolution_ref,
        resolution.batch_ref,
        resolution.selected_branch_ref,
        *resolution.observation_evidence_refs,
    }
    if resolution.execution_request is not None:
        cited_refs.add(resolution.execution_request.execution_request_ref)
    if receipt is not None:
        cited_refs.add(receipt.execution_receipt_ref)
    if resolution.objective_feedback is not None:
        cited_refs.add(resolution.objective_feedback.feedback_ref)
        cited_refs.add(resolution.objective_feedback.feedback_source_ref)

    relation_pairs = {
        (RelationType.ABOUT, cited_ref)
        for cited_ref in cited_refs
        if cited_ref != resolution.resolution_ref
    }
    return CognitiveMemoryRecord(
        kind=CognitiveMemoryKind.EPISODIC,
        epistemic_status=EpistemicStatus.OBSERVED,
        content="\n".join(lines),
        provenance_refs=tuple(sorted(cited_refs)),
        visibility="LEARNER_VISIBLE",
        producer_id=_PRODUCER_ID,
        producer_checkpoint_ref=(
            resolution.reservation.batch.dynamics_checkpoint_ref
        ),
        competence_ref=competence_ref,
        acquired_ordinal=ordinal,
        world_valid_from=None,
        world_valid_until=None,
        relations=_relations(relation_pairs),
        confidence_ppm=None,
    )


@dataclass(frozen=True, slots=True)
class AcquisitionReferenceHit:
    """Untrusted backend references; canonical content is deliberately absent."""

    record_ref: str
    adjacent_record_refs: tuple[str, ...] = ()
    score: float | None = None
    backend_ref: str | None = None

    def __post_init__(self) -> None:
        _ref(self.record_ref, "record_ref")
        if type(self.adjacent_record_refs) is not tuple:
            raise TypeError("adjacent_record_refs must be a tuple")
        if len(self.adjacent_record_refs) > MAX_ACQUISITION_NEIGHBORS:
            raise ValueError(
                "adjacent_record_refs exceeds the 384-reference local ceiling"
            )
        for adjacent_ref in self.adjacent_record_refs:
            _ref(adjacent_ref, "adjacent_record_ref")
        if (
            tuple(sorted(set(self.adjacent_record_refs)))
            != self.adjacent_record_refs
        ):
            raise ValueError(
                "adjacent_record_refs must be unique and canonically sorted"
            )
        if self.score is not None and (
            type(self.score) not in (int, float)
            or not math.isfinite(float(self.score))
        ):
            raise ValueError("score must be finite or None")
        if self.backend_ref is not None:
            _text(self.backend_ref, "backend_ref", 512)


@runtime_checkable
class AcquisitionReferenceBackend(Protocol):
    async def project(self, projection: CognitiveGraphProjectionV2) -> str | None: ...

    async def search(
        self, query: str, *, limit: int
    ) -> Sequence[AcquisitionReferenceHit]: ...

    async def forget_namespace(self) -> None: ...


@runtime_checkable
class CanonicalAcquisitionSource(Protocol):
    def acquisition_head(self) -> object: ...

    def get_acquisition_item(self, record_ref: str) -> object: ...

    def acquisition_items(
        self, after_ordinal: int = -1, limit: int = 64
    ) -> Sequence[object]: ...

    def pending_acquisition_projections(
        self, limit: int = 64
    ) -> Sequence[object]: ...

    def ack_acquisition_projection(self, projection_ref: str) -> None: ...


@dataclass(frozen=True, slots=True)
class AcquisitionRecall:
    record: CognitiveMemoryRecord
    acquisition_ref: str
    source_contract: str
    source_ref: str
    position: TemporalPosition
    adjacent_records: tuple[CognitiveMemoryRecord, ...]
    backend_score: float | None
    backend_ref: str | None
    world_valid_at_query: bool | None

    @property
    def artifact_ref(self) -> str:
        return self.record.record_ref

    @property
    def record_ref(self) -> str:
        return self.record.record_ref

    @property
    def text(self) -> str:
        return self.record.content

    @property
    def visibility(self) -> str:
        return self.record.visibility

    @property
    def epistemic_status(self) -> str:
        return self.record.epistemic_status.value

    @property
    def kind(self) -> str:
        return self.record.kind.value

    @property
    def context(self) -> tuple[tuple[str, str], ...]:
        return (
            ("epistemic_status", self.record.epistemic_status.value),
            ("kind", self.record.kind.value),
            ("producer_checkpoint_ref", self.record.producer_checkpoint_ref),
        )

    @property
    def acquired_ordinal(self) -> int:
        return self.position.acquired_ordinal

    @property
    def age(self) -> int:
        return self.position.age

    @property
    def landmark_relations(self) -> tuple[tuple[str, str], ...]:
        return self.position.landmark_relations

    @property
    def world_valid_from(self) -> int | None:
        return self.position.world_valid_from

    @property
    def world_valid_until(self) -> int | None:
        return self.position.world_valid_until


@dataclass(frozen=True, slots=True)
class AcquisitionRecallBatch:
    items: tuple[AcquisitionRecall, ...]
    rejected: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AcquisitionProjectionReceipt:
    ordinal: int
    acquisition_ref: str
    source_ref: str
    record_ref: str
    projection_ref: str
    backend_ref: str | None


@dataclass(frozen=True, slots=True)
class _ProjectedBinding:
    ordinal: int
    acquisition_ref: str
    source_contract: str
    source_ref: str
    projection_ref: str


def _item_parts(
    item: object,
) -> tuple[int, str, str, CognitiveAcquisition]:
    try:
        ordinal = getattr(item, "ordinal")
        acquisition_ref = getattr(item, "acquisition_ref")
        record_ref = getattr(item, "record_ref")
        acquisition = getattr(item, "acquisition")
    except AttributeError as exc:
        raise TypeError(
            "item does not implement the canonical acquisition-item contract"
        ) from exc
    ordinal = _ordinal(ordinal, "acquisition ordinal")
    _ref(acquisition_ref, "acquisition_ref")
    _ref(record_ref, "record_ref")
    if type(acquisition) is not CognitiveAcquisition:
        raise TypeError("item acquisition must be an exact CognitiveAcquisition")
    _assert_record_neighbor_bound(acquisition.record)
    _assert_record_provenance_bound(acquisition.record)
    encoded = acquisition.canonical_bytes()
    restored = CognitiveAcquisition.from_json(encoded)
    if (
        restored.canonical_bytes() != encoded
        or restored.ordinal != ordinal
        or restored.acquisition_ref != acquisition_ref
        or restored.record.record_ref != record_ref
        or restored.record.acquired_ordinal != ordinal
    ):
        raise ValueError("acquisition item does not match canonical acquisition bytes")
    return ordinal, acquisition_ref, record_ref, restored


def _outbox_parts(
    item: object,
) -> tuple[int, str, str, str, CognitiveGraphProjectionV2]:
    try:
        ordinal = getattr(item, "ordinal")
        acquisition_ref = getattr(item, "acquisition_ref")
        record_ref = getattr(item, "record_ref")
        projection_ref = getattr(item, "projection_ref")
        projection = getattr(item, "projection")
    except AttributeError as exc:
        raise TypeError(
            "item does not implement the acquisition projection-item contract"
        ) from exc
    ordinal = _ordinal(ordinal, "projection ordinal")
    for label, value in (
        ("acquisition_ref", acquisition_ref),
        ("record_ref", record_ref),
        ("projection_ref", projection_ref),
    ):
        _ref(value, label)
    if type(projection) is not CognitiveGraphProjectionV2:
        raise TypeError("projection must be an exact CognitiveGraphProjectionV2")
    _assert_record_neighbor_bound(projection.record)
    _assert_record_provenance_bound(projection.record)
    encoded = projection.canonical_bytes()
    restored = CognitiveGraphProjectionV2.from_json(encoded)
    if (
        restored.canonical_bytes() != encoded
        or restored.projection_ref != projection_ref
        or restored.acquisition_ref != acquisition_ref
        or restored.record.record_ref != record_ref
        or restored.record.acquired_ordinal != ordinal
    ):
        raise ValueError("projection item does not match canonical projection bytes")
    return ordinal, acquisition_ref, record_ref, projection_ref, restored


def _head_parts(head: object) -> tuple[int, str | None, str | None]:
    try:
        next_ordinal = getattr(head, "next_ordinal")
        acquisition_ref = getattr(head, "acquisition_ref")
        record_ref = getattr(head, "record_ref")
    except AttributeError as exc:
        raise TypeError("head does not implement the acquisition-head contract") from exc
    next_ordinal = _ordinal(next_ordinal, "next_ordinal")
    if next_ordinal == 0:
        if acquisition_ref is not None or record_ref is not None:
            raise ValueError("an empty acquisition head cannot carry references")
    else:
        _ref(acquisition_ref, "head acquisition_ref")
        _ref(record_ref, "head record_ref")
    return next_ordinal, acquisition_ref, record_ref


class AcquisitionSituatedMemory:
    """Canonical-reference recall joined to one generic acquisition clock."""

    def __init__(
        self,
        *,
        source: CanonicalAcquisitionSource,
        backend: AcquisitionReferenceBackend,
        origin: MovingOriginIndex | None = None,
    ) -> None:
        if not isinstance(source, CanonicalAcquisitionSource):
            raise TypeError("source does not implement CanonicalAcquisitionSource")
        if not isinstance(backend, AcquisitionReferenceBackend):
            raise TypeError("backend does not implement AcquisitionReferenceBackend")
        self.source = source
        self.backend = backend
        self.origin = origin or MovingOriginIndex()
        self._bindings: dict[str, _ProjectedBinding] = {}
        self._tombstoned_record_refs: set[str] = set()
        self._projection_view_valid = True

    def _invalidate_projection_view(self) -> None:
        """Make a destructive/partial backend rebuild unusable for recall."""

        self.origin = MovingOriginIndex()
        self._bindings = {}
        self._tombstoned_record_refs = set()
        self._projection_view_valid = False

    @staticmethod
    def _synchronization_error(detail: str) -> ValueError:
        return ValueError(
            "acquisition memory is not synchronized with the canonical store; "
            f"rebuild is required ({detail})"
        )

    @staticmethod
    def _projection(acquisition: CognitiveAcquisition) -> CognitiveGraphProjectionV2:
        projection = CognitiveGraphProjectionV2.from_acquisition(acquisition)
        restored = CognitiveGraphProjectionV2.from_json(projection.canonical_bytes())
        restored.assert_acquisition(acquisition)
        return restored

    @staticmethod
    def _binding(
        ordinal: int,
        acquisition: CognitiveAcquisition,
        projection: CognitiveGraphProjectionV2,
    ) -> _ProjectedBinding:
        return _ProjectedBinding(
            ordinal=ordinal,
            acquisition_ref=acquisition.acquisition_ref,
            source_contract=acquisition.source_contract,
            source_ref=acquisition.source_ref,
            projection_ref=projection.projection_ref,
        )

    @staticmethod
    def _anchor_matches(
        anchor: object,
        ordinal: int,
        record: CognitiveMemoryRecord,
        projection_ref: str,
    ) -> bool:
        return (
            getattr(anchor, "event_ref", None) == record.record_ref
            and getattr(anchor, "ordinal", None) == ordinal
            and getattr(anchor, "world_valid_from", None) == record.world_valid_from
            and getattr(anchor, "world_valid_until", None) == record.world_valid_until
            and hmac.compare_digest(
                getattr(anchor, "projection_id", ""), projection_ref
            )
        )

    @staticmethod
    def _predecessor_matches(
        anchor: object,
        acquisition: CognitiveAcquisition,
        bindings: dict[str, _ProjectedBinding],
    ) -> bool:
        previous_record_ref = getattr(anchor, "previous_event_ref", None)
        if acquisition.ordinal == 0:
            return (
                previous_record_ref is None
                and acquisition.predecessor_acquisition_ref is None
            )
        if type(previous_record_ref) is not str:
            return False
        previous_binding = bindings.get(previous_record_ref)
        return (
            previous_binding is not None
            and acquisition.predecessor_acquisition_ref
            == previous_binding.acquisition_ref
        )

    def assert_synchronized(self, head: object | None = None) -> None:
        """Prove the disposable view matches one canonical head in O(1)."""

        if not self._projection_view_valid:
            raise self._synchronization_error(
                "the backend projection generation is invalid"
            )
        if head is None:
            head = self.source.acquisition_head()
        next_ordinal, acquisition_ref, record_ref = _head_parts(head)
        if next_ordinal == 0:
            if self.origin.size != 0 or self._bindings:
                raise self._synchronization_error(
                    "an empty head requires empty temporal and binding indexes"
                )
            return
        if (
            self.origin.size != next_ordinal
            or len(self._bindings) != next_ordinal
        ):
            raise self._synchronization_error(
                "temporal or binding cardinality differs from the head"
            )
        try:
            item = self.source.get_acquisition_item(record_ref)
            ordinal, item_acquisition_ref, item_record_ref, acquisition = _item_parts(
                item
            )
            projection = self._projection(acquisition)
            anchor = self.origin.anchor(item_record_ref)
        except (KeyError, TypeError, ValueError) as exc:
            raise self._synchronization_error(
                "canonical head cannot be rejoined to its projection"
            ) from exc
        expected_binding = self._binding(ordinal, acquisition, projection)
        if (
            ordinal != next_ordinal - 1
            or item_acquisition_ref != acquisition_ref
            or item_record_ref != record_ref
            or self._bindings.get(item_record_ref) != expected_binding
            or not self._anchor_matches(
                anchor, ordinal, acquisition.record, projection.projection_ref
            )
            or not self._predecessor_matches(
                anchor, acquisition, self._bindings
            )
        ):
            raise self._synchronization_error(
                "canonical head references or Moving Origin anchor differ"
            )

    async def _project(
        self,
        item: object,
        *,
        supplied_projection: CognitiveGraphProjectionV2 | None,
        origin: MovingOriginIndex,
        bindings: dict[str, _ProjectedBinding],
        tombstoned_record_refs: set[str],
        acknowledge: bool,
    ) -> AcquisitionProjectionReceipt:
        ordinal, acquisition_ref, record_ref, acquisition = _item_parts(item)
        projection = self._projection(acquisition)
        if supplied_projection is not None:
            supplied_projection.assert_acquisition(acquisition)
            if supplied_projection.canonical_bytes() != projection.canonical_bytes():
                raise ValueError("outbox projection differs from canonical acquisition")
        record = acquisition.record
        expected_binding = self._binding(ordinal, acquisition, projection)
        try:
            anchor = origin.anchor(record_ref)
        except KeyError:
            if origin.size != ordinal:
                raise ValueError(
                    "acquisition ordinal is not the next Moving Origin position"
                )
            anchor = origin.append(
                record_ref,
                projection.projection_ref,
                world_valid_from=record.world_valid_from,
                world_valid_until=record.world_valid_until,
            )
        else:
            if not self._anchor_matches(
                anchor, ordinal, record, projection.projection_ref
            ):
                raise ValueError(
                    "existing Moving Origin anchor does not match the acquisition"
                )
        if not self._predecessor_matches(anchor, acquisition, bindings):
            raise ValueError(
                "acquisition predecessor does not match the projected chain"
            )
        previous = bindings.get(record_ref)
        if previous is not None and previous != expected_binding:
            raise ValueError("record reference maps to a different acquisition")
        backend_ref = await self.backend.project(projection)
        if backend_ref is not None:
            backend_ref = _text(backend_ref, "backend_ref", 512)
        bindings[record_ref] = expected_binding
        tombstoned_record_refs.update(record.tombstone_refs)
        if acknowledge:
            self.source.ack_acquisition_projection(projection.projection_ref)
        return AcquisitionProjectionReceipt(
            ordinal=ordinal,
            acquisition_ref=acquisition_ref,
            source_ref=acquisition.source_ref,
            record_ref=record_ref,
            projection_ref=projection.projection_ref,
            backend_ref=backend_ref,
        )

    async def project_pending_item(
        self, outbox_item: object
    ) -> AcquisitionProjectionReceipt:
        (
            ordinal,
            acquisition_ref,
            record_ref,
            _projection_ref,
            supplied_projection,
        ) = _outbox_parts(outbox_item)
        item = self.source.get_acquisition_item(record_ref)
        item_ordinal, item_acquisition_ref, item_record_ref, acquisition = _item_parts(
            item
        )
        if (
            item_ordinal != ordinal
            or item_acquisition_ref != acquisition_ref
            or item_record_ref != record_ref
        ):
            raise ValueError("outbox item does not rejoin the canonical acquisition")
        supplied_projection.assert_acquisition(acquisition)
        return await self._project(
            item,
            supplied_projection=supplied_projection,
            origin=self.origin,
            bindings=self._bindings,
            tombstoned_record_refs=self._tombstoned_record_refs,
            acknowledge=True,
        )

    async def retry_pending_projections(self, limit: int = 64) -> tuple[str, ...]:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be an integer from 1 through 256")
        pending = tuple(self.source.pending_acquisition_projections(limit=limit))
        if len(pending) > limit:
            raise ValueError("canonical source returned more projections than requested")
        previous_ordinal = -1
        record_refs: list[str] = []
        for item in pending:
            ordinal = _ordinal(getattr(item, "ordinal", None), "projection ordinal")
            if ordinal <= previous_ordinal:
                raise ValueError("pending projections are not oldest-first")
            receipt = await self.project_pending_item(item)
            record_refs.append(receipt.record_ref)
            previous_ordinal = ordinal
        return tuple(record_refs)

    async def rebuild_from_store(self, *, page_size: int = 64) -> tuple[str, ...]:
        if type(page_size) is not int or not 1 <= page_size <= 256:
            raise ValueError("page_size must be an integer from 1 through 256")
        # Forgetting the backend is destructive.  Invalidate the published
        # local view before that await so any failure leaves ordinary recall
        # blocked instead of falsely blessing an old origin against an empty
        # or partially rebuilt backend namespace.
        self._invalidate_projection_view()
        try:
            await self.backend.forget_namespace()
            rebuilt_origin = MovingOriginIndex()
            rebuilt_bindings: dict[str, _ProjectedBinding] = {}
            rebuilt_tombstones: set[str] = set()
            record_refs: list[str] = []
            after_ordinal = -1
            while True:
                page = tuple(
                    self.source.acquisition_items(
                        after_ordinal=after_ordinal,
                        limit=page_size,
                    )
                )
                if len(page) > page_size:
                    raise ValueError(
                        "canonical source returned more items than requested"
                    )
                if not page:
                    break
                expected = after_ordinal + 1
                for item in page:
                    (
                        ordinal,
                        _acquisition_ref,
                        _record_ref,
                        _acquisition,
                    ) = _item_parts(item)
                    if ordinal != expected:
                        raise ValueError(
                            "canonical acquisition page is not contiguous and ordered"
                        )
                    receipt = await self._project(
                        item,
                        supplied_projection=None,
                        origin=rebuilt_origin,
                        bindings=rebuilt_bindings,
                        tombstoned_record_refs=rebuilt_tombstones,
                        acknowledge=False,
                    )
                    record_refs.append(receipt.record_ref)
                    expected += 1
                after_ordinal = expected - 1
                if len(page) < page_size:
                    break
            self.origin = rebuilt_origin
            self._bindings = rebuilt_bindings
            self._tombstoned_record_refs = rebuilt_tombstones
            self._projection_view_valid = True
            self.assert_synchronized()
            return tuple(record_refs)
        except BaseException:
            self._invalidate_projection_view()
            raise

    async def rebuild(self, limit: int = 64) -> tuple[str, ...]:
        return await self.rebuild_from_store(page_size=limit)

    def _resolve_record(
        self,
        record_ref: str,
        *,
        allowed_visibility: frozenset[str],
        admitted_epistemic_statuses: frozenset[EpistemicStatus],
        allow_missing_external_relation: bool = False,
    ) -> tuple[CognitiveMemoryRecord, CognitiveAcquisition] | str | None:
        try:
            item = self.source.get_acquisition_item(record_ref)
        except KeyError:
            if allow_missing_external_relation:
                return None
            return "REFERENCE_UNKNOWN_OR_INVALID"
        try:
            ordinal, _acquisition_ref, item_record_ref, acquisition = _item_parts(item)
            projection = self._projection(acquisition)
        except (TypeError, ValueError):
            return "REFERENCE_UNKNOWN_OR_INVALID"
        record = acquisition.record
        if not hmac.compare_digest(item_record_ref, record_ref):
            return "REFERENCE_IDENTITY_MISMATCH"
        if record.visibility not in allowed_visibility:
            return "REFERENCE_VISIBILITY_DENIED"
        if record.epistemic_status not in admitted_epistemic_statuses:
            return "REFERENCE_EPISTEMIC_STATUS_DENIED"
        if record.record_ref in self._tombstoned_record_refs:
            return "REFERENCE_TOMBSTONED"
        try:
            anchor = self.origin.anchor(record_ref)
        except KeyError:
            return "REFERENCE_TEMPORAL_UNKNOWN"
        if (
            self._bindings.get(record_ref)
            != self._binding(ordinal, acquisition, projection)
            or not self._anchor_matches(
                anchor, ordinal, record, projection.projection_ref
            )
            or not self._predecessor_matches(anchor, acquisition, self._bindings)
        ):
            return "REFERENCE_STALE_OR_TAMPERED"
        return record, acquisition

    @staticmethod
    def _canonical_neighbor_refs(record: CognitiveMemoryRecord) -> frozenset[str]:
        _assert_record_neighbor_bound(record)
        return frozenset(
            {
                *(relation.target_ref for relation in record.relations),
                *record.supersedes_refs,
                *record.tombstone_refs,
            }
        )

    @staticmethod
    def _world_valid(
        position: TemporalPosition, world_time: int | None
    ) -> bool | None:
        if world_time is None:
            return None
        if position.world_valid_from is not None and world_time < position.world_valid_from:
            return False
        if position.world_valid_until is not None and world_time > position.world_valid_until:
            return False
        return True

    def recall_from_hits(
        self,
        hits: Sequence[AcquisitionReferenceHit],
        *,
        limit: int = 12,
        allowed_visibility: Sequence[str] = ("LEARNER_VISIBLE",),
        world_time: int | None = None,
        frozen_origin: bool = False,
    ) -> AcquisitionRecallBatch:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be an integer from 1 through 256")
        if world_time is not None and type(world_time) is not int:
            raise ValueError("world_time must be an integer or None")
        if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
            raise TypeError("reference hits must be a sequence")
        if len(hits) > limit:
            raise ValueError("reference backend returned more hits than requested")
        self.assert_synchronized()
        allowed = frozenset(allowed_visibility)
        resolved: list[
            tuple[AcquisitionReferenceHit, CognitiveMemoryRecord, CognitiveAcquisition]
        ] = []
        rejected: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            if type(hit) is not AcquisitionReferenceHit:
                rejected.append("REFERENCE_RESULT_INVALID")
                continue
            result = self._resolve_record(
                hit.record_ref,
                allowed_visibility=allowed,
                admitted_epistemic_statuses=_ORDINARY_EPISTEMIC_STATUSES,
            )
            if isinstance(result, str):
                rejected.append(result)
                continue
            if result is None:
                raise RuntimeError(
                    "primary record resolution cannot yield an external relation"
                )
            record, acquisition = result
            if record.record_ref in seen:
                rejected.append("REFERENCE_DUPLICATE")
                continue
            seen.add(record.record_ref)
            resolved.append((hit, record, acquisition))

        items: list[AcquisitionRecall] = []
        for hit, record, acquisition in resolved:
            adjacent_records: list[CognitiveMemoryRecord] = []
            canonical_neighbors = self._canonical_neighbor_refs(record)
            generic_relation_refs = frozenset(
                relation.target_ref for relation in record.relations
            )
            dedicated_record_refs = frozenset(
                {*record.supersedes_refs, *record.tombstone_refs}
            )
            for adjacent_ref in hit.adjacent_record_refs:
                if adjacent_ref not in canonical_neighbors:
                    rejected.append("ADJACENT_REFERENCE_NOT_CANONICAL")
                    continue
                adjacent_result = self._resolve_record(
                    adjacent_ref,
                    allowed_visibility=allowed,
                    admitted_epistemic_statuses=_ORDINARY_EPISTEMIC_STATUSES,
                    allow_missing_external_relation=(
                        adjacent_ref in generic_relation_refs
                        and adjacent_ref not in dedicated_record_refs
                    ),
                )
                # Generic typed relations may point to evidence, commitments,
                # branches, or other canonical artifacts rather than acquired
                # memory records.  Their edge remains on the canonical record;
                # only an actual acquisition is exposed as adjacent memory.
                if adjacent_result is None:
                    continue
                if isinstance(adjacent_result, str):
                    rejected.append(
                        "ADJACENT_"
                        + adjacent_result.removeprefix("REFERENCE_")
                    )
                    continue
                adjacent_records.append(adjacent_result[0])
            position = (
                self.origin.frozen_position(record.record_ref)
                if frozen_origin
                else self.origin.position(record.record_ref)
            )
            items.append(
                AcquisitionRecall(
                    record=record,
                    acquisition_ref=acquisition.acquisition_ref,
                    source_contract=acquisition.source_contract,
                    source_ref=acquisition.source_ref,
                    position=position,
                    adjacent_records=tuple(adjacent_records),
                    backend_score=(
                        None if hit.score is None else float(hit.score)
                    ),
                    backend_ref=hit.backend_ref,
                    world_valid_at_query=self._world_valid(position, world_time),
                )
            )
        # A writer may advance the canonical clock while references are being
        # rejoined.  Recheck the O(1) head before exposing temporal ages.
        self.assert_synchronized()
        return AcquisitionRecallBatch(tuple(items), tuple(rejected))

    async def recall(
        self,
        query: str,
        *,
        limit: int = 12,
        allowed_visibility: Sequence[str] = ("LEARNER_VISIBLE",),
        world_time: int | None = None,
        frozen_origin: bool = False,
    ) -> AcquisitionRecallBatch:
        query = _text(query, "query", 16_384)
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("limit must be an integer from 1 through 256")
        self.assert_synchronized()
        hits = await self.backend.search(query, limit=limit)
        return self.recall_from_hits(
            hits,
            limit=limit,
            allowed_visibility=allowed_visibility,
            world_time=world_time,
            frozen_origin=frozen_origin,
        )


__all__ = [
    "AcquisitionProjectionReceipt",
    "AcquisitionRecall",
    "AcquisitionRecallBatch",
    "AcquisitionReferenceBackend",
    "AcquisitionReferenceHit",
    "AcquisitionSituatedMemory",
    "CanonicalAcquisitionSource",
    "MAX_ACQUISITION_NEIGHBORS",
    "MAX_ACQUISITION_PROVENANCE_REFS",
    "prospective_batch_record",
    "prospective_resolution_record",
]
