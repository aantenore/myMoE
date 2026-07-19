from __future__ import annotations

import unittest

from local_moe.verified_routing_contracts import (
    VerifiedRoutingError,
    canonical_json,
    require_finite_number,
    require_identifier_tuple,
    require_non_negative_int,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


class VerifiedRoutingContractTests(unittest.TestCase):
    def test_canonical_digest_is_order_independent(self) -> None:
        left = {"profile": "balanced", "routes": ["local", "premium"]}
        right = {"routes": ["local", "premium"], "profile": "balanced"}

        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(sha256_json(left), sha256_json(right))

    def test_rejects_non_finite_metrics(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(VerifiedRoutingError):
                require_finite_number(value, "metric")

    def test_rejects_boolean_integer_and_duplicate_identifiers(self) -> None:
        with self.assertRaises(VerifiedRoutingError):
            require_non_negative_int(True, "count")
        with self.assertRaises(VerifiedRoutingError):
            require_identifier_tuple(["code", "code"], "capabilities")

    def test_requires_lowercase_sha256_and_utc_timestamp(self) -> None:
        digest = "a" * 64

        self.assertEqual(require_sha256(digest, "digest"), digest)
        self.assertEqual(
            require_utc_timestamp("2026-07-19T08:30:00Z", "created_at"),
            "2026-07-19T08:30:00+00:00",
        )
        with self.assertRaises(VerifiedRoutingError):
            require_sha256("A" * 64, "digest")
        with self.assertRaises(VerifiedRoutingError):
            require_utc_timestamp("2026-07-19T08:30:00+02:00", "created_at")


if __name__ == "__main__":
    unittest.main()
