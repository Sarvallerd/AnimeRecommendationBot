"""Type-only contracts for the private acceptance scripts and their test loaders.

The scripts remain separately executable files. Their loaders validate ModuleSpec,
loader and consumed exports before narrowing a dynamic module to these contracts.
No data, transport, deployment, or scientific package is imported here.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict

if TYPE_CHECKING:
    from deploy.tests.live_verifier_v2 import (
        CaseInputs,
        CaseReceipt,
        LiveContext,
        LivePlan,
        OperationInputs,
        OperationReceipt,
        SummaryRecord,
    )

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class ProfileSeed(TypedDict):
    mal_id: int
    query: str
    neighbor_ids: list[int]


class EmptySeed(TypedDict):
    mal_id: int
    query: str


class ProfileExpectations(TypedDict):
    catalog_count: int
    neighbor_count_histogram: dict[str, int]
    smoke_seed: ProfileSeed
    empty_seed: EmptySeed | None
    vector_status_counts: NotRequired[dict[str, JsonValue]]
    zero_vector_ids: NotRequired[list[JsonValue]]


class ProfileReference(TypedDict):
    block_size: int
    build_report_sha256: str
    bundle_identity: str
    catalog_sha256: str
    neighbors_sha256: str
    producer: JsonValue


class NewProfileExpectations(TypedDict):
    catalog_count: int
    neighbor_count_histogram: dict[str, int]
    smoke_seed: ProfileSeed
    empty_seed: EmptySeed | None
    vector_status_counts: dict[str, JsonValue]
    zero_vector_ids: list[JsonValue]


class ProducerReference(TypedDict):
    code_files_sha256: dict[str, str]
    numpy_version: str
    package_version: str
    python_version: str
    source_commit: str


class ValidatedProfileReference(TypedDict):
    block_size: int
    build_report_sha256: str
    bundle_identity: str
    catalog_sha256: str
    neighbors_sha256: str
    producer: ProducerReference


class NamedSourceInput(TypedDict):
    identity: str
    snapshot_id: str


class CatalogReleaseInput(TypedDict):
    catalog_count: int
    catalog_sha256: str
    identity: str


class EmbeddingsInput(TypedDict):
    registry_sha256: str
    snapshot_id: str
    text_sha256: str
    text_size: int


class AlgorithmInput(TypedDict):
    max_neighbors: int
    name: str
    parameters: JsonObject
    similarity: str
    version: str


class ProfileInputs(TypedDict):
    algorithm: AlgorithmInput
    aliases: NamedSourceInput
    base: NamedSourceInput
    catalog_release: CatalogReleaseInput
    embeddings: EmbeddingsInput


class ProfileLock(TypedDict):
    schema_version: int
    kind: str
    release_id: str
    inputs: ProfileInputs
    reference: ValidatedProfileReference
    semantic_expectations: NewProfileExpectations


class Neighbor(TypedDict):
    mal_id: int
    similarity: int | float


class LegacyBundle(TypedDict):
    id: str
    count: int
    neighbors: dict[str, list[Neighbor]]


class ArtifactMount(TypedDict):
    source: str
    rw: bool


class ContainerRecord(TypedDict):
    id: str
    image_id: str
    project: str
    service: str
    user: str
    status: str
    running: bool
    exit_code: int
    started_at: str
    restart_count: int
    read_only_root: bool
    artifact_mount: ArtifactMount | None
    database_volume: str | None
    bundle_id: str | None
    bundle_count: int | None


class DeploymentRecord(TypedDict):
    project: str
    compose_sha256: str
    settings_sha256: str


class RuntimeRecord(TypedDict):
    bot: ContainerRecord | None
    prepare: ContainerRecord | None
    postgres: ContainerRecord | None


class SnapshotRecord(TypedDict):
    schema_version: int
    kind: str
    snapshot_sql_sha256: str
    source_commit: str
    captured_at: str
    deployment: DeploymentRecord
    runtime: RuntimeRecord
    database: JsonObject | None


class SettingsFileRecord(TypedDict):
    sha256: str
    uid: int
    gid: int
    mode: int


class SettingsGuardRecord(TypedDict):
    settings: SettingsFileRecord
    credential_sha256: dict[str, str]


class PollerInventoryRecord(TypedDict):
    matching: int
    scope_complete: bool
    scope: str
    matching_container_ids_sha256: list[str]


class PostgresIdentityRecord(TypedDict):
    id: str
    image_id: str
    database_volume: str | None
    started_at: str


class LiveGuardRecord(SettingsGuardRecord):
    daemon_id: str
    pollers: PollerInventoryRecord
    postgres_identity: PostgresIdentityRecord | None
    postgres_health: str
    stable: NotRequired[bool]


class LiveCaptureRecord(TypedDict):
    schema_version: int
    kind: str
    state: SnapshotRecord
    guards: LiveGuardRecord


class CatalogEntryView(TypedDict):
    title: NotRequired[str]
    aliases: NotRequired[list[str]]
    synopsis: NotRequired[str]


class BaselineBotView(TypedDict):
    id: str
    image_id: str
    bundle_id: str | None
    bundle_count: int | None
    artifact_mount: ArtifactMount | None


class BaselinePostgresView(TypedDict):
    id: str
    image_id: str
    running: bool
    database_volume: str | None


class BaselineRuntimeView(TypedDict):
    bot: BaselineBotView | None
    postgres: BaselinePostgresView | None


class BaselineStateView(TypedDict):
    kind: str
    source_commit: str
    runtime: BaselineRuntimeView


class BaselinePollerView(TypedDict):
    matching: int
    scope_complete: bool
    matching_container_ids_sha256: list[str]


class BaselineGuardView(TypedDict):
    daemon_id: str
    postgres_health: str
    pollers: BaselinePollerView


class BaselineCaptureView(TypedDict):
    kind: str
    state: BaselineStateView
    guards: BaselineGuardView


class AliasProbeView(TypedDict):
    mal_id: int
    query: str


class DescriptionProbeView(TypedDict):
    seed_mal_id: int
    query: str
    neighbor_mal_id: int


class UiProbesView(TypedDict):
    schema_version: int
    kind: str
    alias_search: AliasProbeView
    description_edit: DescriptionProbeView


class OfflineOriginalView(TypedDict):
    bundle_identity: str
    bundle_file_sha256: dict[str, str]


class OfflineBuildView(TypedDict):
    identity: str
    file_sha256: dict[str, str]


class OfflineRuntimeView(TypedDict):
    project: str
    database_volume: str
    bot_image_id: str


class OfflineReportView(TypedDict):
    original: OfflineOriginalView
    builds: dict[str, OfflineBuildView]
    runtime: OfflineRuntimeView


class SavedDescriptor(TypedDict):
    file: str
    sha256: str


class PathDescriptor(TypedDict):
    path: str
    sha256: str
    bytes: int


class ProfileError(ValueError):
    code: str


class EvidenceFailure(ValueError):
    code: str


class EvidenceModule(Protocol):
    ROOT: Path
    SQL: Path
    PROJECT: re.Pattern[str]
    HEX: re.Pattern[str]
    IDENTITY: re.Pattern[str]
    CASES: tuple[str, ...]
    NO_EFFECT: tuple[str, ...]
    EvidenceError: type[EvidenceFailure]
    compose: CaptureCompose
    docker: CaptureDocker

    def require(self, condition: bool, code: str) -> None: ...

    def sha(self, raw: bytes) -> str: ...

    def canonical(self, value: object) -> bytes: ...

    def utc(self, value: str) -> dt.datetime: ...

    def read_settings(self, path: Path) -> dict[str, str]: ...

    def container(self, project: str, settings: Path, service: str) -> ContainerRecord | None: ...

    def snapshot(
        self, project: str, settings: Path, runtime_only: bool = False
    ) -> SnapshotRecord: ...

    def validate_database(self, db: object, bundle: LegacyBundle | None = None) -> None: ...

    def private_parent(self, path: Path) -> None: ...

    def read_json(self, path: Path) -> JsonObject: ...

    def write_new(self, path: Path, value: object) -> str: ...


class AcceptanceProfile(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def release_id(self) -> str: ...

    @property
    def policy(self) -> str: ...

    @property
    def sha256(self) -> str: ...

    @property
    def lock(self) -> JsonObject | None: ...

    @property
    def lock_sha256(self) -> str | None: ...

    @property
    def expectations(self) -> ProfileExpectations: ...

    @property
    def reference(self) -> ProfileReference | None: ...

    @property
    def inputs(self) -> JsonObject | None: ...


class CurrentAcceptanceProfile(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def release_id(self) -> str: ...

    @property
    def policy(self) -> str: ...

    @property
    def sha256(self) -> str: ...

    @property
    def lock(self) -> ProfileLock: ...

    @property
    def lock_sha256(self) -> str: ...

    @property
    def expectations(self) -> NewProfileExpectations: ...

    @property
    def reference(self) -> ValidatedProfileReference: ...

    @property
    def inputs(self) -> ProfileInputs: ...


class BundleSummaryView(Protocol):
    identity: str
    count: int
    hashes: dict[str, str]
    neighbors: dict[str, list[dict[str, JsonValue]]]
    histogram: dict[str, int]


class BundleSummaryFactory(Protocol):
    def __call__(
        self,
        identity: str,
        count: int,
        hashes: dict[str, str],
        neighbors: dict[str, list[dict[str, JsonValue]]],
        histogram: dict[str, int],
    ) -> BundleSummaryView: ...


class ProfileModule(Protocol):
    NEW_RELEASE: str
    LEGACY_RELEASE: str
    PROFILE_DIR: Path
    PROFILE_FILES: dict[str, str]
    TRUSTED_NEW_PROFILE_SHA: str
    TRUSTED_LEGACY_PROFILE_SHA: str
    TRUSTED_LOCK_SHA: str
    LOCK_RESOURCE: str
    ProfileError: type[ProfileError]
    BundleSummary: BundleSummaryFactory

    def digest(self, raw: bytes) -> str: ...

    def canonical(self, raw: bytes) -> JsonObject: ...

    def regular_bytes(self, path: Path) -> bytes: ...

    def load_profile(
        self,
        release_id: str,
        *,
        _profile_bytes: bytes | None = None,
        _lock_bytes: bytes | None = None,
    ) -> AcceptanceProfile: ...

    def check_profile_bundle(
        self,
        bundle_dir: Path,
        profile: AcceptanceProfile | CurrentAcceptanceProfile,
        *,
        expected_identity: str,
    ) -> BundleSummaryView: ...


class Level1Runner(Protocol):
    started_pg: bool
    cleanup_ok: bool
    cases: dict[str, dict[str, str | int]]

    def command(
        self,
        name: str,
        argv: list[str],
        timeout: float,
        extra_env: dict[str, str] | None = None,
        check: bool = True,
    ) -> str: ...

    def compose(self, name: str, args: list[str], timeout: float = 120) -> str: ...

    def cleanup(self) -> None: ...


class Level1RunnerFactory(Protocol):
    def __call__(self, scope: str) -> Level1Runner: ...


class Level1Module(Protocol):
    COMPOSE_MARKERS: set[str]
    Runner: Level1RunnerFactory

    def require_cases(self, text: str, required: set[str]) -> None: ...

    def require_markers(self, text: str) -> None: ...


class BundleBytesContract(Protocol):
    def check_bundle_bytes(
        self, catalog_raw: bytes, neighbors_raw: bytes, manifest_raw: bytes
    ) -> tuple[str, int]: ...

    def parse_canonical(self, raw: bytes, location: str) -> JsonValue: ...


class LiveEvidenceMain(Protocol):
    def main(self, argv: list[str] | None = None) -> int: ...


class PathBundleContract(Protocol):
    def check_bundle(self, path: Path) -> tuple[str, int]: ...

    def read_canonical(self, path: Path) -> tuple[JsonValue, bytes]: ...


class CaptureCompose(Protocol):
    def __call__(
        self,
        project: str,
        settings: Path,
        *args: str,
        input_text: str | None = None,
    ) -> str: ...


class CaptureDocker(Protocol):
    def __call__(self, *args: str) -> str: ...


class EvidenceTransport(Protocol):
    compose: CaptureCompose
    docker: CaptureDocker


class ReleaseModule(Protocol):
    ReleaseError: type[ValueError]

    def verify_record(self, evidence: Path) -> OfflineReportView: ...

    def change_selection(self, original: bytes, bundle: Path) -> bytes: ...


class LiveVerifierModule(Protocol):
    e: EvidenceModule
    p: ProfileModule
    r: ReleaseModule
    OPERATIONS: tuple[str, ...]
    CASES: tuple[str, ...]

    def digest(self, value: object) -> str: ...

    def validate_plan(self, plan: object, *, verify_offline: bool = True) -> LivePlan: ...

    def _valid_capture(self, item: object, plan: LivePlan, *, full: bool = True) -> object: ...

    def verify_operation(
        self,
        plan: LivePlan,
        operation: str,
        before: LiveCaptureRecord,
        stopped: LiveCaptureRecord,
        prepared: LiveCaptureRecord,
        started: LiveCaptureRecord,
    ) -> OperationReceipt: ...

    def bind_context(
        self,
        plan: LivePlan,
        initial_operation: OperationReceipt,
        before: LiveCaptureRecord,
        after: LiveCaptureRecord,
    ) -> LiveContext: ...

    def verify_case(
        self,
        plan: LivePlan,
        context: LiveContext,
        case: str,
        before: LiveCaptureRecord,
        after: LiveCaptureRecord,
        *,
        during: LiveCaptureRecord | None = None,
        operation: OperationReceipt | None = None,
    ) -> CaseReceipt: ...

    def summarize_evidence(
        self,
        plan: LivePlan,
        operations: dict[str, OperationReceipt],
        results: dict[str, CaseReceipt],
        final: LiveCaptureRecord,
        context: LiveContext | None = None,
        observations: object | None = None,
        operation_inputs: dict[str, OperationInputs] | None = None,
        case_inputs: dict[str, CaseInputs] | None = None,
    ) -> SummaryRecord: ...
