from __future__ import annotations

from pathlib import Path
import unittest

from scripts import run_packaging_smoke


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SDIST_ASSETS = (
    ".agents/plugins/marketplace.json",
    "configs/local-cascade.example.json",
    "configs/local-cascade-moe.example.json",
    "docs/local-cascade.md",
    "experiments/benchmark_local_cascade.py",
    "outputs/local-cascade-contract-benchmark.json",
    "plugins/mymoe-local-cascade/.codex-plugin/plugin.json",
    "plugins/mymoe-local-cascade/.mcp.json",
    "plugins/mymoe-local-cascade/scripts/launch_mcp.py",
    "plugins/mymoe-local-cascade/skills/mymoe-local-cascade/SKILL.md",
    "plugins/mymoe-local-cascade/skills/mymoe-local-cascade/agents/openai.yaml",
)


class LocalCascadePackagingContractTests(unittest.TestCase):
    def test_release_metadata_exposes_bounded_mcp_extra_and_entrypoint(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

        self.assertIn('version = "0.14.0a1"', pyproject)
        self.assertIn('local-cascade = [\n  "mcp>=1.27.2,<2"\n]', pyproject)
        self.assertIn(
            'mymoe-local-cascade-mcp = "local_moe.local_cascade_mcp:main"',
            pyproject,
        )
        self.assertIn('name = "local-moe-orchestrator"\nversion = "0.14.0a1"', lock)
        self.assertIn(
            '{ name = "mcp", marker = "extra == \'local-cascade\'", '
            'specifier = ">=1.27.2,<2" }',
            lock,
        )

    def test_sdist_manifest_uses_explicit_tracked_assets(self) -> None:
        declarations = {
            line.strip()
            for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        for relative in REQUIRED_SDIST_ASSETS:
            with self.subTest(asset=relative):
                self.assertTrue((ROOT / relative).is_file())
                self.assertIn(f"include {relative}", declarations)
        self.assertIn("include scripts/run_packaging_smoke.py", declarations)

        local_cascade_declarations = {
            declaration
            for declaration in declarations
            if "local-cascade" in declaration or ".agents/plugins" in declaration
        }
        self.assertTrue(local_cascade_declarations)
        self.assertFalse(
            any(
                token in declaration
                for declaration in local_cascade_declarations
                for token in ("*", "?", "recursive-include")
            )
        )

    def test_packaging_smoke_covers_local_cascade_release_contract(self) -> None:
        self.assertTrue(
            set(REQUIRED_SDIST_ASSETS).issubset(
                run_packaging_smoke.REQUIRED_SDIST_ARTIFACTS
            )
        )
        self.assertEqual(
            run_packaging_smoke.LOCAL_CASCADE_RUNTIME_REQUIREMENTS,
            ("mcp==1.27.2",),
        )
        self.assertEqual(
            run_packaging_smoke.EXPECTED_LOCAL_CASCADE_TOOLS,
            (
                "delegate_plan",
                "delegate_run",
                "machine_inspect",
                "receipt_inspect",
            ),
        )
        client_smoke = run_packaging_smoke.LOCAL_CASCADE_MCP_CLIENT_SMOKE
        self.assertIn("async with stdio_client(server)", client_smoke)
        self.assertIn("await session.initialize()", client_smoke)
        self.assertIn("await session.list_tools()", client_smoke)


if __name__ == "__main__":
    unittest.main()
