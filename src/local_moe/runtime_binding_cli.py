"""Read-only CLI for inspecting one exact local cell runtime binding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

from .adaptive_advisor_cli import (
    AdvisorCliError,
    ProtectedRootIdentity,
    _write_output,
)
from .runtime_binding_inspector import (
    CellBindingInspectionBundle,
    RuntimeBindingInspectionError,
    inspect_cell_binding,
    load_cell_binding_inspect_request,
)


EXIT_VERIFIED = 0
EXIT_ABSTAINED = 1
EXIT_INVALID = 2

_PUBLIC_ERROR_CODE = re.compile(r"\A[a-z][a-z0-9_]{0,63}\Z")


class RuntimeBindingCliError(ValueError):
    """Stable, request-safe command failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _RuntimeBindingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise RuntimeBindingCliError(
            "invocation_invalid",
            "Invalid cell binding inspection invocation.",
        )


def build_parser() -> argparse.ArgumentParser:
    boundary = (
        "Inspection only: this command does not start or download models, "
        "access the network, grant authorization, or reserve resources."
    )
    parser = _RuntimeBindingArgumentParser(
        prog="mymoe cell-bind",
        description=(
            "Fingerprint one declared local cell binding without starting it."
        ),
        epilog=boundary,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect",
        help="Inspect and hash one declared local cell binding.",
        description=(
            "Inspect and hash one declared local cell binding without changing "
            "runtime state."
        ),
        epilog=boundary,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inspect.add_argument(
        "--request",
        required=True,
        metavar="PATH",
        help="Bounded local CellBindingInspectRequest JSON file.",
    )
    inspect.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit the complete manifest and inspection receipt as stable JSON.",
    )
    inspect.add_argument(
        "--out",
        metavar="PATH",
        help=(
            "Atomically publish the complete JSON bundle to one new owner-only "
            "file outside inspected roots; existing files are never replaced."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command != "inspect":
            raise RuntimeBindingCliError(
                "invocation_invalid",
                "A cell binding inspection command is required.",
            )
        request_path = _absolute_cli_path(args.request)
        output_path = _absolute_cli_path(args.out) if args.out is not None else None
        bundle = inspect_cell_binding(request_path, publication_path=output_path)
        payload = bundle.payload()
        _validate_payload_shape(payload)
        _validate_status(bundle.receipt.status)
        if args.out is not None:
            protected_inputs = _publication_inputs(
                request_path=request_path,
                expected_request_sha256=bundle.request_sha256,
                protected_root_identities=bundle.publication_protected_roots,
            )
            _publish_bundle(
                output_path=output_path,
                payload=payload,
                protected_inputs=protected_inputs,
                protected_root_identities=bundle.publication_protected_roots,
            )
        rendered = (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
            if args.json_output
            else render_human(bundle)
        )
    except RuntimeBindingCliError as exc:
        _render_error(exc.code, str(exc))
        return EXIT_INVALID
    except RuntimeBindingInspectionError as exc:
        _render_error(
            _safe_error_code(getattr(exc, "code", "inspection_failed")),
            "Cell binding inspection could not be completed safely.",
        )
        return EXIT_INVALID
    except AdvisorCliError as exc:
        _render_error(
            _safe_error_code(getattr(exc, "code", "output_publish_failed")),
            "Cell binding output could not be published safely.",
        )
        return EXIT_INVALID
    except Exception:
        _render_error(
            "inspection_failed",
            "Cell binding inspection could not be completed safely.",
        )
        return EXIT_INVALID

    print(rendered)
    return EXIT_VERIFIED if bundle.receipt.status == "verified" else EXIT_ABSTAINED


def render_human(bundle: CellBindingInspectionBundle) -> str:
    receipt = bundle.receipt
    manifest = bundle.manifest
    if receipt.status == "verified":
        heading = f"Cell binding verified: {manifest.cell_id}"
        reasons = "Every declared local component matched."
    else:
        heading = f"Cell binding inspection abstained: {manifest.cell_id}"
        reasons = "; ".join(receipt.reason_codes)
    return "\n".join(
        (
            heading,
            f"Why: {reasons}",
            (
                "Components observed: "
                f"{receipt.observed_component_count}/{receipt.component_count}"
            ),
            f"Manifest SHA-256: {manifest.digest}",
            f"Receipt SHA-256: {receipt.digest}",
            (
                "Boundary: inspection only. No model was started or downloaded; "
                "no network, authorization, resource reservation, or runtime "
                "mutation was used."
            ),
        )
    )


def _publish_bundle(
    *,
    output_path: Path,
    payload: Mapping[str, object],
    protected_inputs: Sequence[Path],
    protected_root_identities: Sequence[ProtectedRootIdentity],
) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_output(
        output_path,
        encoded,
        protected_inputs=protected_inputs,
        protected_root_identities=protected_root_identities,
    )


def _publication_inputs(
    *,
    request_path: Path,
    expected_request_sha256: str,
    protected_root_identities: Sequence[ProtectedRootIdentity],
) -> tuple[Path, ...]:
    try:
        request_file = Path(os.path.abspath(os.fspath(request_path)))
        request = load_cell_binding_inspect_request(request_file)
    except (
        OSError,
        OverflowError,
        RuntimeBindingInspectionError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeBindingCliError(
            "request_changed_during_inspection",
            "The inspection request changed before output publication.",
        ) from exc
    if request.digest != expected_request_sha256:
        raise RuntimeBindingCliError(
            "request_changed_during_inspection",
            "The inspection request changed before output publication.",
        )
    root = request_file.parent
    expected_roots = (
        root / request.runtime_root,
        root / request.model_artifact_root,
    )
    observed_roots = tuple(item.path for item in protected_root_identities)
    if observed_roots != expected_roots:
        raise RuntimeBindingCliError(
            "request_changed_during_inspection",
            "The inspection publication boundary does not match its request.",
        )
    return (
        request_file,
        root / request.catalog_path,
        root / request.runtime_config_path,
    )


def _absolute_cli_path(value: object) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(value)))
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        raise RuntimeBindingCliError(
            "invocation_invalid",
            "Cell binding paths could not be resolved.",
        ) from exc


def _validate_payload_shape(payload: object) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeBindingCliError(
            "inspection_result_invalid",
            "Cell binding inspection returned an invalid result.",
        )
    if not isinstance(payload.get("binding_manifest"), Mapping) or not isinstance(
        payload.get("inspection_receipt"), Mapping
    ):
        raise RuntimeBindingCliError(
            "inspection_result_invalid",
            "Cell binding inspection returned an invalid result.",
        )


def _validate_status(status: object) -> None:
    if status not in {"verified", "abstained"}:
        raise RuntimeBindingCliError(
            "inspection_result_invalid",
            "Cell binding inspection returned an invalid result.",
        )


def _safe_error_code(value: object) -> str:
    rendered = str(value or "")
    return rendered if _PUBLIC_ERROR_CODE.fullmatch(rendered) else "inspection_failed"


def _render_error(code: str, message: str) -> None:
    print(
        json.dumps(
            {
                "error": "cell_binding_error",
                "code": code,
                "message": message,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_ABSTAINED",
    "EXIT_INVALID",
    "EXIT_VERIFIED",
    "RuntimeBindingCliError",
    "build_parser",
    "main",
    "render_human",
]
