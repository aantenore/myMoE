from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import secrets
import stat
from typing import Callable, Mapping, Sequence

from .assistant_bridge import AssistantTaskEnvelope, BridgeRunResult
from .paired_execution_bridge import (
    PairedArmExecutor,
    PairedArmPlan,
    paired_arm_operation_sha256,
)
from .assistant_bridge_workspace import WorkspaceSnapshot
from .paired_execution_contracts import (
    PairedOutcomeBinding,
    PairedRunRoot,
    PairedRunSlot,
)
from .paired_evidence import VerifiedPairedEvidence
from .paired_execution_pricing import (
    IncompleteCostEvidenceError,
    PairedCostEvidence,
    PricingContract,
    build_cost_evidence,
)
from .paired_execution_store import (
    PairedExecutionStore,
    PairedRunIndeterminateError,
    PairedRunStatus,
)
from .route_outcomes import (
    OutcomeStore,
    VerifiedOutcomeRecord,
    build_verified_outcome,
    runtime_plan_sha256,
)
from .route_promotion import PromotionCase, VerifiedRoutingEvidencePlan
from .route_signals import (
    MetadataTaskSignalProvider,
    TaskSignals,
)
from .verified_routing_contracts import (
    VerifiedRoutingError,
    require_sha256,
    sha256_json,
)


_OS_NAME = os.name
_MAX_RUNNER_SOURCE_FILES = 512
_MAX_RUNNER_SOURCE_BYTES = 32 * 1024 * 1024
_MAX_RUNNER_SOURCE_FILE_BYTES = 4 * 1024 * 1024


def _new_run_instance_nonce() -> str:
    return secrets.token_hex(32)


@dataclass(frozen=True)
class PairedCaseResult:
    """Metadata-only read model for a completed or resumed paired case."""

    state: str
    root: PairedRunRoot
    records: tuple[VerifiedOutcomeRecord, ...]
    cost_evidence_payloads: tuple[Mapping[str, object] | None, ...]

    def __post_init__(self) -> None:
        if self.state != "complete":
            raise VerifiedRoutingError(
                "Paired case results are returned only after complete execution."
            )
        records = tuple(self.records)
        costs = tuple(self.cost_evidence_payloads)
        if len(records) != 2 or len(costs) != 2:
            raise VerifiedRoutingError(
                "A completed paired case requires exactly two outcomes and costs."
            )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "cost_evidence_payloads", costs)

    @property
    def cost_complete(self) -> bool:
        return all(item is not None for item in self.cost_evidence_payloads)

    def metadata_payload(self) -> dict[str, object]:
        return {
            "state": self.state,
            "root": self.root.payload(),
            "outcome_record_ids": [record.record_id for record in self.records],
            "cost_complete": self.cost_complete,
            "cost_evidence": [
                None if item is None else dict(item)
                for item in self.cost_evidence_payloads
            ],
            "privacy": "metadata_only",
        }


def paired_runner_sha256(
    *,
    executor_config_sha256: str,
    lifecycle_config_sha256: str,
    signal_provider_config_sha256: str,
    runner_source_sha256: str | None = None,
) -> str:
    """Digest semantic source plus every injected execution configuration."""

    source_sha256 = (
        paired_runner_source_sha256()
        if runner_source_sha256 is None
        else require_sha256(runner_source_sha256, "runner source")
    )
    return sha256_json(
        {
            "contract": "verified-paired-runner/v2",
            "runner_source_sha256": source_sha256,
            "executor_config_sha256": require_sha256(
                executor_config_sha256,
                "paired executor config",
            ),
            "lifecycle_config_sha256": require_sha256(
                lifecycle_config_sha256,
                "paired lifecycle config",
            ),
            "signal_provider_config_sha256": require_sha256(
                signal_provider_config_sha256,
                "paired signal provider config",
            ),
        }
    )


def paired_runner_source_sha256() -> str:
    """Digest executable package source independently from run-specific state."""

    source_manifest, source_bytes = _runner_source_manifest()
    return sha256_json(
        {
            "contract": "verified-paired-runner-source/v1",
            "source_manifest": source_manifest,
            "source_file_count": len(source_manifest),
            "source_total_bytes": source_bytes,
        }
    )


def paired_execution_harness_sha256(
    *,
    executor_harness_sha256: str,
    signal_provider_config_sha256: str,
) -> str:
    """Bind the inspected executor harness to the plan's signal semantics."""

    return sha256_json(
        {
            "contract": "mymoe-paired-plan-execution-harness/v1",
            "executor_harness_sha256": require_sha256(
                executor_harness_sha256,
                "executor harness",
            ),
            "signal_provider_config_sha256": require_sha256(
                signal_provider_config_sha256,
                "signal provider configuration",
            ),
        }
    )


def _runner_source_manifest() -> tuple[tuple[dict[str, object], ...], int]:
    """Return a bounded, ordered manifest of every executable package module."""

    package = Path(__file__).resolve(strict=True).parent
    sources: list[Path] = []
    try:
        for directory, directory_names, file_names in os.walk(
            package,
            topdown=True,
            followlinks=False,
        ):
            current = Path(directory)
            for name in directory_names:
                metadata = (current / name).lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise VerifiedRoutingError(
                        "Paired runner source tree cannot contain directory links."
                    )
                if not stat.S_ISDIR(metadata.st_mode):
                    raise VerifiedRoutingError(
                        "Paired runner source tree contains an invalid directory."
                    )
            for name in file_names:
                if not name.endswith(".py"):
                    continue
                sources.append(current / name)
                if len(sources) > _MAX_RUNNER_SOURCE_FILES:
                    raise VerifiedRoutingError(
                        "Paired runner source manifest exceeds its file limit."
                    )
    except VerifiedRoutingError:
        raise
    except OSError as exc:
        raise VerifiedRoutingError(
            "Paired runner source tree cannot be enumerated safely."
        ) from exc

    manifest: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(sources, key=lambda item: item.relative_to(package).as_posix()):
        content = _read_runner_source_file(path)
        total_bytes += len(content)
        if total_bytes > _MAX_RUNNER_SOURCE_BYTES:
            raise VerifiedRoutingError(
                "Paired runner source manifest exceeds its byte limit."
            )
        manifest.append(
            {
                "path": path.relative_to(package).as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    if not manifest:
        raise VerifiedRoutingError("Paired runner source manifest is empty.")
    return tuple(manifest), total_bytes


def _read_runner_source_file(path: Path) -> bytes:
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_RUNNER_SOURCE_FILE_BYTES
        ):
            raise VerifiedRoutingError(
                "Paired runner source must be a bounded regular non-link file."
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (before.st_dev, before.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise VerifiedRoutingError(
                    "Paired runner source changed while it was opened."
                )
            chunks: list[bytes] = []
            remaining = _MAX_RUNNER_SOURCE_FILE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = path.lstat()
    except VerifiedRoutingError:
        raise
    except OSError as exc:
        raise VerifiedRoutingError(
            "Paired runner source cannot be read safely."
        ) from exc
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, opened, after, current)
    }
    if len(identities) != 1 or len(content) != before.st_size:
        raise VerifiedRoutingError(
            "Paired runner source changed while it was read."
        )
    return content


def run_paired_case(
    *,
    task: AssistantTaskEnvelope,
    plan: VerifiedRoutingEvidencePlan,
    case: PromotionCase,
    source_workspace: str | Path,
    pricing: PricingContract,
    run_store: PairedExecutionStore | str | Path,
    outcome_store: OutcomeStore | str | Path,
    executor: PairedArmExecutor,
    signal_provider: MetadataTaskSignalProvider | None = None,
    context_tokens: int | None = None,
    created_at: str | None = None,
) -> PairedCaseResult:
    """Run one frozen AB/BA case with durable pre-invocation claims.

    Each claim is durable before ticket planning or provider execution.  Any later
    failure is deliberately indeterminate and is never retried.  Only metadata-only
    outcome records and exact pricing evidence are persisted.
    """

    if not isinstance(task, AssistantTaskEnvelope):
        raise TypeError("task must be an AssistantTaskEnvelope.")
    if not isinstance(plan, VerifiedRoutingEvidencePlan):
        raise TypeError("plan must be a VerifiedRoutingEvidencePlan.")
    if not isinstance(case, PromotionCase):
        raise TypeError("case must be a PromotionCase.")
    if not isinstance(pricing, PricingContract):
        raise TypeError("pricing must be a PricingContract.")
    executable_signal_provider = MetadataTaskSignalProvider()
    signals_provider = (
        executable_signal_provider
        if signal_provider is None
        else signal_provider
    )
    if type(signals_provider) is not MetadataTaskSignalProvider:
        raise TypeError(
            "Schema 1.0 requires the concrete MetadataTaskSignalProvider."
        )
    if (
        signals_provider.config_sha256
        != executable_signal_provider.config_sha256
    ):
        raise VerifiedRoutingError(
            "Schema 1.0 requires the default MetadataTaskSignalProvider config."
        )
    signals = MetadataTaskSignalProvider.signals_from_metadata(
        signals_provider,
        task.metadata_payload(),
        context_tokens=context_tokens,
    )
    if not isinstance(signals, TaskSignals):
        raise TypeError("signal_provider must return TaskSignals.")
    if signals.provider_config_sha256 != executable_signal_provider.config_sha256:
        raise VerifiedRoutingError(
            "Task signals do not match the executable provider configuration."
        )
    preflight = getattr(executor, "preflight", None)
    if not callable(preflight):
        raise TypeError("paired executor must implement callable preflight.")
    verify_outcome = getattr(executor, "verify_outcome", None)
    if not callable(verify_outcome):
        raise TypeError("paired executor must implement callable verify_outcome.")
    preflight(task)
    state_paths = _executor_state_paths(executor)
    _validate_case_inputs(task, plan, case, pricing, signals, executor)
    executor_config_sha256 = require_sha256(
        executor.configuration_sha256,
        "paired executor configuration",
    )
    execution_harness_sha256 = paired_execution_harness_sha256(
        executor_harness_sha256=executor.execution_harness_sha256,
        signal_provider_config_sha256=signals.provider_config_sha256,
    )
    runner_source_sha256 = paired_runner_source_sha256()
    if (
        execution_harness_sha256 != plan.execution_harness_sha256
        or runner_source_sha256 != plan.runner_source_sha256
    ):
        raise VerifiedRoutingError(
            "Paired executor does not match the preregistered execution harness."
        )
    lifecycle_config_sha256 = sha256_json(
        {
            "contract": "mymoe-paired-lifecycle/v1",
            "plan_sha256": plan.plan_sha256,
            "bridge_config_sha256": case.config_sha256,
            "executor_config_sha256": executor_config_sha256,
        }
    )

    store = (
        run_store
        if isinstance(run_store, PairedExecutionStore)
        else PairedExecutionStore(run_store)
    )
    outcomes = (
        outcome_store
        if isinstance(outcome_store, OutcomeStore)
        else OutcomeStore(outcome_store)
    )
    source_workspace = _preflight_store_isolation(
        source_workspace=source_workspace,
        run_store=store,
        outcome_store=outcomes,
        executor_state_paths=state_paths,
    )
    existing_status = store.status()
    if existing_status.state == "indeterminate":
        raise PairedRunIndeterminateError(
            "Paired execution has an uncheckpointed claim; retry is forbidden."
        )
    _require_executor_configuration(executor, executor_config_sha256)
    source_snapshot = executor.snapshot_source(source_workspace)
    if not isinstance(source_snapshot, WorkspaceSnapshot):
        raise VerifiedRoutingError(
            "Paired arm executor returned an invalid source snapshot."
        )
    runner_sha256 = paired_runner_sha256(
        executor_config_sha256=executor_config_sha256,
        lifecycle_config_sha256=lifecycle_config_sha256,
        signal_provider_config_sha256=signals.provider_config_sha256,
        runner_source_sha256=runner_source_sha256,
    )
    run_instance_nonce = (
        _new_run_instance_nonce()
        if existing_status.root is None
        else existing_status.root.run_instance_nonce
    )
    root = PairedRunRoot.build(
        plan_sha256=plan.plan_sha256,
        case_sha256=sha256_json(case.payload()),
        task_fingerprint=case.task_fingerprint,
        normalized_item_sha256=case.normalized_item_sha256,
        source_snapshot_sha256=source_snapshot.fingerprint,
        bridge_config_sha256=case.config_sha256,
        executor_config_sha256=executor_config_sha256,
        execution_harness_sha256=execution_harness_sha256,
        lifecycle_config_sha256=lifecycle_config_sha256,
        signals_sha256=signals.signals_sha256,
        runner_sha256=runner_sha256,
        runner_source_sha256=runner_source_sha256,
        pricing_sha256=pricing.pricing_sha256,
        run_instance_nonce=run_instance_nonce,
        order=case.order,
        baseline_route=case.baseline_route,
        candidate_route=case.candidate_route,
    )
    store.prepare(root)
    status = store.status()
    _validate_resumable_status(status, root)
    if status.state == "complete":
        _require_runner_configuration(
            runner_sha256,
            executor_config_sha256=executor_config_sha256,
            lifecycle_config_sha256=lifecycle_config_sha256,
            signal_provider_config_sha256=signals.provider_config_sha256,
        )
        records = _load_checkpointed_records(
            status,
            outcomes,
            case,
            pricing,
            verify_outcome=verify_outcome,
        )
        executor.assert_source_unchanged(source_snapshot)
        return _result(root, records)

    while status.state in {"ready", "partial"}:
        _require_executor_configuration(executor, executor_config_sha256)
        _require_runner_configuration(
            runner_sha256,
            executor_config_sha256=executor_config_sha256,
            lifecycle_config_sha256=lifecycle_config_sha256,
            signal_provider_config_sha256=signals.provider_config_sha256,
        )
        if status.checkpoints:
            # A resumed arm must not run unless every earlier checkpoint still
            # has its durable signed evidence.  This also rechecks the first
            # arm immediately before claiming the second arm in one process.
            _load_checkpointed_records(
                status,
                outcomes,
                case,
                pricing,
                verify_outcome=verify_outcome,
            )
        slot = status.next_slot
        if slot is None:
            raise VerifiedRoutingError("Paired run has no next execution slot.")
        claim = store.claim(slot)
        try:
            binding = store.binding_for(claim)
            # The claim precedes both ticket planning and provider execution.  The
            # bridge operation digest is derived only from this durable permit.
            arm_plan = executor.plan_arm(
                task,
                source_workspace=source_workspace,
                source_snapshot=source_snapshot,
                signals=signals,
                lifecycle_config_sha256=lifecycle_config_sha256,
                baseline_route=case.baseline_route,
                slot=slot,
                permit=binding,
            )
            receipt = _validate_arm_plan(
                arm_plan,
                binding=binding,
                root=root,
                case=case,
                task=task,
                signals=signals,
                executor_config_sha256=executor_config_sha256,
                lifecycle_config_sha256=lifecycle_config_sha256,
            )
            _require_runner_configuration(
                runner_sha256,
                executor_config_sha256=executor_config_sha256,
                lifecycle_config_sha256=lifecycle_config_sha256,
                signal_provider_config_sha256=signals.provider_config_sha256,
            )
            result = executor.run_arm(
                task,
                source_workspace=source_workspace,
                source_snapshot=source_snapshot,
                signals=signals,
                lifecycle_config_sha256=lifecycle_config_sha256,
                baseline_route=case.baseline_route,
                plan=arm_plan,
            )
            _require_runner_configuration(
                runner_sha256,
                executor_config_sha256=executor_config_sha256,
                lifecycle_config_sha256=lifecycle_config_sha256,
                signal_provider_config_sha256=signals.provider_config_sha256,
            )
            executor.assert_source_unchanged(source_snapshot)
            _validate_arm_result(result, receipt, root, case, signals)
            cost = _cost_evidence(result, pricing)
            record = build_verified_outcome(
                result.metadata_payload(),
                signals,
                estimated_cost_usd=(
                    None if cost is None else float(cost.total_cost_usd)
                ),
                # Caller time is never authority.  The only chronology copied
                # into the row comes from the signed CandidateBinding.
                created_at=_signed_evidence_time(result),
                paired_run=binding.payload(),
                paired_cost=None if cost is None else cost.payload(),
                paired_evidence=result.paired_evidence,
            )
            _validate_record(record, binding, case, cost, pricing)
            reconstructed = verify_outcome(record, pricing)
            if (
                not isinstance(reconstructed, VerifiedPairedEvidence)
                or reconstructed.record != record
            ):
                raise VerifiedRoutingError(
                    "Paired outcome failed immediate signed-evidence reconstruction."
                )
            _require_runner_configuration(
                runner_sha256,
                executor_config_sha256=executor_config_sha256,
                lifecycle_config_sha256=lifecycle_config_sha256,
                signal_provider_config_sha256=signals.provider_config_sha256,
            )
            outcomes.append(record)
            _require_runner_configuration(
                runner_sha256,
                executor_config_sha256=executor_config_sha256,
                lifecycle_config_sha256=lifecycle_config_sha256,
                signal_provider_config_sha256=signals.provider_config_sha256,
            )
            store.complete(
                binding,
                outcome_record_id=record.record_id,
                route_receipt_id=record.route_receipt_id,
                route_receipt_sha256=record.route_receipt_sha256,
                evidence_sha256=record.evidence_sha256,
            )
        except BaseException as exc:
            store.abandon(claim)
            try:
                executor.assert_source_unchanged(source_snapshot)
            except Exception as integrity_error:
                raise integrity_error from exc
            raise
        status = store.status()
        _validate_resumable_status(status, root)

    if status.state != "complete":
        raise VerifiedRoutingError("Paired execution did not reach completion.")
    _require_runner_configuration(
        runner_sha256,
        executor_config_sha256=executor_config_sha256,
        lifecycle_config_sha256=lifecycle_config_sha256,
        signal_provider_config_sha256=signals.provider_config_sha256,
    )
    records = _load_checkpointed_records(
        status,
        outcomes,
        case,
        pricing,
        verify_outcome=verify_outcome,
    )
    executor.assert_source_unchanged(source_snapshot)
    return _result(root, records)


def _preflight_store_isolation(
    *,
    source_workspace: str | Path,
    run_store: PairedExecutionStore,
    outcome_store: OutcomeStore,
    executor_state_paths: Sequence[Path] = (),
) -> Path:
    """Resolve storage paths before snapshotting and reject physical overlap."""

    source_root = _resolved_path(source_workspace, "source workspace")
    run_root = _resolved_path(run_store.run_dir, "paired run store")
    outcome_targets = (
        _resolved_path(outcome_store.path, "outcome store"),
        _resolved_path(outcome_store.lock_path, "outcome store lock"),
    )
    state_paths = tuple(
        _resolved_path(path, "paired executor state")
        for path in executor_state_paths
    )
    for path in state_paths:
        _validate_sensitive_state_path(path)
    if _paths_overlap(source_root, run_root) or any(
        _paths_overlap(source_root, target)
        for target in (*outcome_targets, *state_paths)
    ):
        raise VerifiedRoutingError(
            "Paired execution stores must be physically isolated from the "
            "source workspace."
        )
    if any(_paths_overlap(run_root, target) for target in outcome_targets):
        raise VerifiedRoutingError(
            "Paired run and outcome stores must not alias or overlap."
        )
    if any(_paths_overlap(run_root, target) for target in state_paths) or any(
        _paths_overlap(outcome, target)
        for outcome in outcome_targets
        for target in state_paths
    ):
        raise VerifiedRoutingError(
            "Paired evidence state must not overlap run or outcome stores."
        )
    for index, path in enumerate(state_paths):
        if any(_paths_overlap(path, other) for other in state_paths[index + 1 :]):
            raise VerifiedRoutingError(
                "Paired executor state paths must not alias or overlap."
            )
    return source_root


def _executor_state_paths(executor: PairedArmExecutor) -> tuple[Path, ...]:
    declared = getattr(executor, "state_paths", None)
    if callable(declared):
        declared = declared()
    if (
        isinstance(declared, (str, bytes, bytearray))
        or not isinstance(declared, (list, tuple))
        or any(not isinstance(item, Path) for item in declared)
    ):
        raise TypeError("paired executor must expose Path state_paths.")
    return tuple(declared)


def _validate_sensitive_state_path(path: Path) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        _validate_nearest_private_parent(path)
        return
    except OSError as exc:
        raise VerifiedRoutingError(
            "Paired executor state path is unavailable."
        ) from exc
    if stat.S_ISLNK(details.st_mode):
        raise VerifiedRoutingError(
            "Paired executor state path must be a regular file or directory."
        )
    if stat.S_ISDIR(details.st_mode):
        _validate_private_state_directory(path, details)
        return
    if not stat.S_ISREG(details.st_mode):
        raise VerifiedRoutingError(
            "Paired executor state path must be a regular file or directory."
        )
    if details.st_nlink != 1:
        raise VerifiedRoutingError(
            "Paired executor state path cannot be hard-linked."
        )
    if _OS_NAME == "posix" and (
        stat.S_IMODE(details.st_mode) != 0o600
        or details.st_uid != os.getuid()
    ):
        raise VerifiedRoutingError(
            "Paired executor state file permissions are unsafe."
        )
    _validate_private_state_directory(path.parent)


def _validate_nearest_private_parent(path: Path) -> None:
    parent = path.parent
    while True:
        try:
            details = parent.lstat()
        except FileNotFoundError:
            if parent == parent.parent:
                raise VerifiedRoutingError(
                    "Paired executor state path has no existing private parent."
                )
            parent = parent.parent
            continue
        except OSError as exc:
            raise VerifiedRoutingError(
                "Paired executor state parent is unavailable."
            ) from exc
        _validate_private_state_directory(parent, details)
        return


def _validate_private_state_directory(
    path: Path,
    details: os.stat_result | None = None,
) -> None:
    if details is None:
        try:
            details = path.lstat()
        except OSError as exc:
            raise VerifiedRoutingError(
                "Paired executor state parent is unavailable."
            ) from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise VerifiedRoutingError(
            "Paired executor state parent must be a non-link directory."
        )
    if _OS_NAME == "posix" and (
        stat.S_IMODE(details.st_mode) != 0o700
        or details.st_uid != os.getuid()
    ):
        raise VerifiedRoutingError(
            "Paired executor state directory permissions are unsafe."
        )


def _resolved_path(value: str | Path, label: str) -> Path:
    try:
        expanded = Path(value).expanduser()
    except (OSError, RuntimeError) as exc:
        raise VerifiedRoutingError(f"{label} cannot be resolved safely.") from exc
    if _OS_NAME == "nt":
        try:
            details = expanded.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise VerifiedRoutingError(
                f"{label} cannot be resolved safely."
            ) from exc
        else:
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            attributes = int(getattr(details, "st_file_attributes", 0))
            if stat.S_ISLNK(details.st_mode) or attributes & reparse_flag:
                raise VerifiedRoutingError(
                    f"{label} must be physically isolated and cannot be a "
                    "symlink or Windows reparse point."
                )
    try:
        return expanded.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise VerifiedRoutingError(f"{label} cannot be resolved safely.") from exc


def _paths_overlap(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        if left.exists() and right.exists() and left.samefile(right):
            return True
    except OSError as exc:
        raise VerifiedRoutingError(
            "Paired execution storage aliases cannot be verified safely."
        ) from exc
    return _path_within(left, right) or _path_within(right, left)


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_case_inputs(
    task: AssistantTaskEnvelope,
    plan: VerifiedRoutingEvidencePlan,
    case: PromotionCase,
    pricing: PricingContract,
    signals: TaskSignals,
    executor: PairedArmExecutor,
) -> None:
    matches = tuple(item for item in plan.cases if item.task_fingerprint == case.task_fingerprint)
    if len(matches) != 1 or matches[0] != case:
        raise VerifiedRoutingError("Promotion case is not an exact member of the plan.")
    if plan.pricing_contract is None:
        raise VerifiedRoutingError(
            "Paired execution requires an embedded pricing contract."
        )
    if (
        plan.pricing_sha256 != pricing.pricing_sha256
        or plan.pricing_contract.payload() != pricing.payload()
    ):
        raise VerifiedRoutingError(
            "Pricing contract does not match the paired evidence plan."
        )
    if task.task_fingerprint != case.task_fingerprint:
        raise VerifiedRoutingError("Task does not match the paired promotion case.")
    if task.profile != case.profile:
        raise VerifiedRoutingError("Task profile does not match the promotion case.")
    if tuple(sorted(task.capability_demand.required)) != case.capabilities:
        raise VerifiedRoutingError(
            "Task capabilities do not match the promotion case."
        )
    if signals.request_fingerprint != case.task_fingerprint:
        raise VerifiedRoutingError("Task signals belong to another promotion case.")
    if signals.provider_config_sha256 != case.signal_provider_config_sha256:
        raise VerifiedRoutingError(
            "Signal provider does not match the promotion case."
        )
    if signals.difficulty != case.difficulty:
        raise VerifiedRoutingError("Task difficulty does not match the promotion case.")
    if executor.bridge_config_sha256 != case.config_sha256:
        raise VerifiedRoutingError(
            "Assistant Bridge configuration does not match the promotion case."
        )
    require_sha256(executor.configuration_sha256, "paired executor configuration")


def _require_executor_configuration(
    executor: PairedArmExecutor,
    expected_sha256: str,
) -> None:
    if executor.configuration_sha256 != expected_sha256:
        raise VerifiedRoutingError(
            "Paired executor configuration changed before slot claim."
        )


def _require_runner_configuration(
    expected_sha256: str,
    *,
    executor_config_sha256: str,
    lifecycle_config_sha256: str,
    signal_provider_config_sha256: str,
) -> None:
    current = paired_runner_sha256(
        executor_config_sha256=executor_config_sha256,
        lifecycle_config_sha256=lifecycle_config_sha256,
        signal_provider_config_sha256=signal_provider_config_sha256,
    )
    if current != expected_sha256:
        raise VerifiedRoutingError(
            "Paired runner implementation changed during execution."
        )


def _validate_resumable_status(status: PairedRunStatus, root: PairedRunRoot) -> None:
    if status.root != root:
        raise VerifiedRoutingError("Paired execution store root changed.")
    if status.state == "running":
        raise VerifiedRoutingError("Paired execution is already running.")
    if status.state == "indeterminate":
        raise PairedRunIndeterminateError(
            "Paired execution has an uncheckpointed claim; retry is forbidden."
        )
    if status.state not in {"ready", "partial", "complete"}:
        raise VerifiedRoutingError("Paired execution store is not resumable.")


def _validate_arm_plan(
    plan: PairedArmPlan,
    *,
    binding: PairedOutcomeBinding,
    root: PairedRunRoot,
    case: PromotionCase,
    task: AssistantTaskEnvelope,
    signals: TaskSignals,
    executor_config_sha256: str,
    lifecycle_config_sha256: str,
) -> dict[str, object]:
    if not isinstance(plan, PairedArmPlan):
        raise VerifiedRoutingError("Paired executor returned an invalid arm plan.")
    if (
        plan.slot not in root.slots
        or plan.permit != binding
        or plan.signals != signals
        or binding.run_id != root.run_id
        or plan.operation_sha256 != paired_arm_operation_sha256(binding)
    ):
        raise VerifiedRoutingError("Paired arm plan is bound to another run slot.")
    raw = _mapping(plan.bridge_plan, "paired bridge plan")
    required = {
        "mode",
        "execute",
        "guarded_baseline_route",
        "evaluation_route",
        "source_snapshot_sha256",
        "route_receipt",
        "generator_config_sha256",
        "paired_executor_config_sha256",
        "lifecycle_config_sha256",
        "operation_sha256",
        "authority",
        "privacy",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise VerifiedRoutingError(
            "Paired bridge plan is missing fields: " + ", ".join(missing) + "."
        )
    if (
        raw["mode"] != "assistant_bridge_paired_evidence_plan"
        or raw["execute"] is not False
        or raw["privacy"] != "metadata_only"
        or raw["guarded_baseline_route"] != case.baseline_route
        or raw["evaluation_route"] != plan.slot.route
        or raw["source_snapshot_sha256"] != root.source_snapshot_sha256
        or raw["paired_executor_config_sha256"] != executor_config_sha256
        or raw["lifecycle_config_sha256"] != lifecycle_config_sha256
        or raw["operation_sha256"] != plan.operation_sha256
        or binding.executor_config_sha256 != executor_config_sha256
        or binding.lifecycle_config_sha256 != lifecycle_config_sha256
        or binding.signals_sha256 != signals.signals_sha256
    ):
        raise VerifiedRoutingError("Paired bridge plan does not match the run root.")
    require_sha256(
        raw["generator_config_sha256"],  # type: ignore[arg-type]
        "paired generator configuration",
    )
    authority = _mapping(raw["authority"], "paired bridge authority")
    if authority.get("source_apply") != "forbidden":
        raise VerifiedRoutingError("Paired execution must forbid source apply.")
    receipt = _mapping(raw["route_receipt"], "paired route receipt")
    _validate_receipt(receipt, root, case, task, signals, plan.slot.route)
    return receipt


def _validate_arm_result(
    result: BridgeRunResult,
    planned_receipt: Mapping[str, object],
    root: PairedRunRoot,
    case: PromotionCase,
    signals: TaskSignals,
) -> None:
    if not isinstance(result, BridgeRunResult):
        raise VerifiedRoutingError("Paired executor returned an invalid bridge result.")
    receipt = result.receipt.payload()
    if receipt != dict(planned_receipt):
        raise VerifiedRoutingError(
            "Executed route receipt differs from the confirmed arm plan."
        )
    task_raw = _mapping(receipt.get("task"), "executed route task")
    if task_raw.get("task_fingerprint") != case.task_fingerprint:
        raise VerifiedRoutingError("Executed result belongs to another task.")
    if runtime_plan_sha256(receipt) != case.runtime_plan_sha256:
        raise VerifiedRoutingError("Executed runtime plan changed after planning.")
    # RouteDecisionReceipt.workspace uses the bridge attestation digest domain;
    # the raw WorkspaceSnapshot fingerprint is instead bound by the arm plan,
    # root, generator precondition, and post-run unchanged-source assertion.
    if signals.request_fingerprint != case.task_fingerprint:
        raise VerifiedRoutingError("Executed arm signals changed.")


def _validate_receipt(
    receipt: Mapping[str, object],
    root: PairedRunRoot,
    case: PromotionCase,
    task: AssistantTaskEnvelope,
    signals: TaskSignals,
    route: str,
) -> None:
    route_task = _mapping(receipt.get("task"), "paired route task")
    _mapping(receipt.get("workspace"), "paired route workspace")
    demand = _mapping(route_task.get("capability_demand"), "paired route demand")
    if receipt.get("route") != route:
        raise VerifiedRoutingError(
            "Paired route receipt does not match the declared arm route."
        )
    if receipt.get("config_sha256") != case.config_sha256:
        raise VerifiedRoutingError(
            "Paired route receipt changed the bridge configuration."
        )
    if route_task != task.metadata_payload():
        raise VerifiedRoutingError(
            "Paired route receipt changed the task metadata."
        )
    # Do not compare receipt.workspace.fingerprint with the root fingerprint:
    # they are intentionally different digest domains.  _validate_arm_plan binds
    # bridge_plan.source_snapshot_sha256 to the root before reaching this helper.
    if tuple(sorted(_string_sequence(demand.get("required")))) != case.capabilities:
        raise VerifiedRoutingError(
            "Paired route receipt changed the required capabilities."
        )
    if runtime_plan_sha256(receipt) != case.runtime_plan_sha256:
        raise VerifiedRoutingError("Paired route receipt changed the runtime plan.")
    if (
        signals.difficulty != case.difficulty
        or signals.provider_config_sha256 != case.signal_provider_config_sha256
    ):
        raise VerifiedRoutingError(
            "Paired route receipt does not match the frozen task signals."
        )
    if "route_canary" in receipt:
        raise VerifiedRoutingError(
            "Paired evidence cannot consume canary routing authority."
        )


def _cost_evidence(
    result: BridgeRunResult,
    pricing: PricingContract,
) -> PairedCostEvidence | None:
    try:
        commands = _command_cost_metadata(result)
        return build_cost_evidence(pricing, commands)
    except IncompleteCostEvidenceError:
        return None


def _signed_evidence_time(result: BridgeRunResult) -> str:
    value = result.paired_evidence_created_at
    if value is None or result.paired_evidence is None:
        raise VerifiedRoutingError(
            "Paired result is missing its signed attestation receipt or time."
        )
    return datetime.fromtimestamp(value, tz=timezone.utc).replace(
        microsecond=0
    ).isoformat()


def _command_cost_metadata(result: BridgeRunResult) -> tuple[dict[str, object], ...]:
    receipt = result.receipt.payload()
    local_provider = receipt.get("local_provider")
    premium_provider = receipt.get("premium_provider")
    runtimes: dict[str, Mapping[str, object]] = {}
    if isinstance(local_provider, str):
        runtimes[local_provider] = _mapping(
            receipt.get("local_runtime"),
            "local runtime",
        )
    if isinstance(premium_provider, str):
        runtimes[premium_provider] = _mapping(
            receipt.get("premium_runtime"),
            "premium runtime",
        )
    metadata: list[dict[str, object]] = []
    for command in result.commands:
        runtime = runtimes.get(command.provider_id)
        if runtime is None:
            raise VerifiedRoutingError(
                "Command provider is not attested by the route receipt."
            )
        if command.prompt_tokens is None or command.completion_tokens is None:
            raise IncompleteCostEvidenceError(
                "Cost evidence is incomplete: command token usage is unavailable."
            )
        model = runtime.get("model")
        runtime_digest = runtime.get("runtime_sha256")
        if not isinstance(model, str) or not model:
            raise IncompleteCostEvidenceError(
                "Cost evidence is incomplete: runtime model is unavailable."
            )
        if not isinstance(runtime_digest, str) or not runtime_digest:
            raise IncompleteCostEvidenceError(
                "Cost evidence is incomplete: runtime digest is unavailable."
            )
        metadata.append(
            {
                "provider_id": command.provider_id,
                "model": model,
                "provider_runtime_sha256": runtime_digest,
                "prompt_tokens": command.prompt_tokens,
                "completion_tokens": command.completion_tokens,
            }
        )
    if not metadata:
        raise IncompleteCostEvidenceError(
            "Cost evidence is incomplete: the arm has no provider command."
        )
    return tuple(metadata)


def _load_checkpointed_records(
    status: PairedRunStatus,
    outcomes: OutcomeStore,
    case: PromotionCase,
    pricing: PricingContract,
    *,
    verify_outcome: Callable[
        [VerifiedOutcomeRecord, PricingContract], VerifiedPairedEvidence
    ],
) -> tuple[VerifiedOutcomeRecord, ...]:
    by_id = {record.record_id: record for record in outcomes.list_records()}
    records: list[VerifiedOutcomeRecord] = []
    for checkpoint in status.checkpoints:
        record = by_id.get(checkpoint.outcome_record_id)
        if record is None:
            raise VerifiedRoutingError(
                "Paired checkpoint outcome is missing from the outcome store."
            )
        if (
            record.route_receipt_id != checkpoint.route_receipt_id
            or record.route_receipt_sha256 != checkpoint.route_receipt_sha256
            or record.evidence_sha256 != checkpoint.evidence_sha256
        ):
            raise VerifiedRoutingError(
                "Paired outcome does not match its execution checkpoint."
            )
        payload = record.payload()
        paired_cost_raw = payload.get("paired_cost")
        cost = (
            None
            if paired_cost_raw is None
            else PairedCostEvidence.from_payload(
                _mapping(paired_cost_raw, "paired cost evidence"),
                pricing=pricing,
            )
        )
        _validate_record(record, checkpoint.binding, case, cost, pricing)
        reconstructed = verify_outcome(record, pricing)
        if (
            not isinstance(reconstructed, VerifiedPairedEvidence)
            or reconstructed.record != record
        ):
            raise VerifiedRoutingError(
                "Checkpointed outcome failed signed-evidence reconstruction."
            )
        records.append(record)
    return tuple(records)


def _validate_record(
    record: VerifiedOutcomeRecord,
    binding: PairedOutcomeBinding,
    case: PromotionCase,
    cost: PairedCostEvidence | None,
    pricing: PricingContract,
) -> None:
    payload = record.payload()
    if payload.get("paired_run") != binding.payload():
        raise VerifiedRoutingError("Outcome has invalid paired-run lineage.")
    if payload.get("paired_evidence") is None:
        raise VerifiedRoutingError(
            "Outcome has no reconstructible paired attestation receipt."
        )
    if (
        record.task_fingerprint != case.task_fingerprint
        or record.profile != case.profile
        or record.capabilities != case.capabilities
        or record.difficulty != case.difficulty
        or record.config_sha256 != case.config_sha256
        or record.signal_provider_config_sha256
        != case.signal_provider_config_sha256
        or record.runtime_plan_sha256 != case.runtime_plan_sha256
        or record.planned_route != binding.route
    ):
        raise VerifiedRoutingError("Outcome does not match its promotion case.")
    paired_cost = payload.get("paired_cost")
    if cost is None:
        if paired_cost is not None or record.estimated_cost_usd is not None:
            raise VerifiedRoutingError(
                "Incomplete cost evidence cannot carry an estimated cost."
            )
        return
    parsed = PairedCostEvidence.from_payload(
        _mapping(paired_cost, "paired cost evidence"),
        pricing=pricing,
    )
    if parsed != cost:
        raise VerifiedRoutingError("Outcome paired cost evidence changed.")


def _result(
    root: PairedRunRoot,
    records: Sequence[VerifiedOutcomeRecord],
) -> PairedCaseResult:
    costs: list[Mapping[str, object] | None] = []
    for record in records:
        raw = record.payload().get("paired_cost")
        costs.append(
            None if raw is None else _mapping(raw, "paired cost evidence")
        )
    return PairedCaseResult(
        state="complete",
        root=root,
        records=tuple(records),
        cost_evidence_payloads=tuple(costs),
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise VerifiedRoutingError(f"{label} must be an object.")
    return dict(value)


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise VerifiedRoutingError("Paired route capabilities must be a list.")
    return tuple(value)
