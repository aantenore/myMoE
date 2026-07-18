from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREDICATE_URI = (
    "https://github.com/aantenore/myMoE/tree/main/docs/spec/"
    "independent-candidate-attestation/v1"
)
SPEC_ROOT = ROOT / "docs" / "spec" / "independent-candidate-attestation" / "v1"


def _runtime_contract_available() -> bool:
    try:
        return find_spec("local_moe.assistant_bridge_two_phase_contracts") is not None
    except ModuleNotFoundError:
        return False


class IndependentCandidatePredicateSpecTests(unittest.TestCase):
    def test_published_path_and_example_expose_the_exact_predicate_identity(
        self,
    ) -> None:
        repository_relative = PREDICATE_URI.split("/tree/main/", maxsplit=1)[1]
        readme = SPEC_ROOT / "README.md"
        example = SPEC_ROOT / "example.statement.json"

        self.assertEqual((ROOT / repository_relative).resolve(), SPEC_ROOT.resolve())
        self.assertTrue(readme.is_file())
        self.assertTrue(example.is_file())
        self.assertIn(PREDICATE_URI, readme.read_text(encoding="utf-8"))

        statement = json.loads(example.read_text(encoding="utf-8"))
        self.assertEqual(
            set(statement),
            {"_type", "subject", "predicateType", "predicate"},
        )
        self.assertEqual(statement["_type"], "https://in-toto.io/Statement/v1")
        self.assertEqual(statement["predicateType"], PREDICATE_URI)
        self.assertEqual(
            set(statement["predicate"]),
            {
                "schemaVersion",
                "binding",
                "bindingSha256",
                "attestation",
                "outcome",
            },
        )

    @unittest.skipUnless(
        _runtime_contract_available(),
        "two-phase runtime contract is not present on this pre-foundation branch",
    )
    def test_documented_example_is_exactly_rebuilt_by_the_runtime_contract(
        self,
    ) -> None:
        contracts = import_module("local_moe.assistant_bridge_two_phase_contracts")
        requirement = contracts.VerifierRequirement(
            "verifier-a",
            "dsse-ed25519-v1",
            "key-a",
            "8" * 64,
            "9" * 64,
        )
        policy = contracts.VerificationPolicy(
            "policy-v1",
            1,
            (requirement,),
        )
        binding = contracts.CandidateBinding(
            "workflow-example-001",
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "5" * 64,
            contracts.ArtifactDescriptor(
                "application/vnd.mymoe.workspace-manifest+json",
                "6" * 64,
                512,
            ),
            contracts.ArtifactDescriptor(
                "application/vnd.mymoe.changeset+json",
                "7" * 64,
                256,
            ),
            policy,
            1735689600.0,
            1735693200.0,
        )
        statement = contracts.build_attestation_statement(
            binding,
            requirement,
            attestation_id="attestation-example-001",
            issued_at=1735689660.0,
            expires_at=1735690200.0,
            checks=(
                contracts.AttestationCheck(
                    "contract-tests",
                    True,
                    "a" * 64,
                ),
                contracts.AttestationCheck(
                    "workspace-integrity",
                    True,
                    "b" * 64,
                ),
            ),
        )
        documented = json.loads(
            (SPEC_ROOT / "example.statement.json").read_text(encoding="utf-8")
        )

        self.assertEqual(documented, statement)
        self.assertEqual(
            statement["predicateType"],
            contracts.INDEPENDENT_PREDICATE_V1,
        )


if __name__ == "__main__":
    unittest.main()
