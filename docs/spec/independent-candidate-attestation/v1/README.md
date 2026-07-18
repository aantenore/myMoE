# Independent Candidate Attestation Predicate v1

Status: stable

Predicate type:

<https://github.com/aantenore/myMoE/tree/main/docs/spec/independent-candidate-attestation/v1>

This document defines the version 1 predicate used to bind independently
produced verification evidence to one immutable myMoE candidate workflow. It
describes provenance and integrity data only. It does not carry an assistant
transcript, hidden reasoning, credentials, or signing material.

Version 1 is immutable. An incompatible field, validation, or derivation change
requires a new predicate-type URI.

## Conformance

The key words MUST, MUST NOT, SHOULD, and MAY describe normative requirements.
A consumer MUST reject unknown or missing fields wherever this specification
defines an exact field set.

The statement MUST use:

- in-toto Statement v1 as its statement type;
- the predicate-type URI above;
- one subject named <code>urn:mymoe:candidate:&lt;workflowId&gt;</code>;
- a subject SHA-256 digest equal to <code>candidateFingerprint</code>;
- RFC 8785 JSON Canonicalization Scheme bytes for the signed statement payload.

The current transport profile is DSSE with payload type
<code>application/vnd.in-toto+json</code>. The
<code>dsse-ed25519-v1</code> adapter accepts exactly one signature whose key ID
matches the workflow requirement. Verifier signing material is managed outside
myMoE; this repository contains no private signing key.

## Statement shape

The statement has exactly four fields:

| Field | Required value |
| --- | --- |
| <code>_type</code> | <code>https://in-toto.io/Statement/v1</code> |
| <code>subject</code> | The single candidate subject described above |
| <code>predicateType</code> | This specification's predicate-type URI |
| <code>predicate</code> | The exact predicate object below |

The predicate has exactly these fields:

| Field | Meaning |
| --- | --- |
| <code>schemaVersion</code> | <code>1.0</code> |
| <code>binding</code> | Complete candidate workflow binding |
| <code>bindingSha256</code> | SHA-256 of the RFC 8785 canonical binding |
| <code>attestation</code> | Verifier identity, trust, and lifetime metadata |
| <code>outcome</code> | Ordered verification checks and their aggregate result |

## Candidate binding

The binding contains exactly:

- <code>schemaVersion</code>
- <code>workflowId</code>
- <code>stageIdempotencySha256</code>
- <code>taskFingerprint</code>
- <code>configSha256</code>
- <code>sourceFingerprint</code>
- <code>candidateContentSha256</code>
- <code>candidateFingerprint</code>
- <code>challengeSha256</code>
- <code>manifest</code>
- <code>changeset</code>
- <code>verificationPolicy</code>
- <code>verificationPolicySha256</code>
- <code>createdAt</code>
- <code>expiresAt</code>

Each artifact descriptor contains exactly <code>mediaType</code>,
<code>digest.sha256</code>, and <code>sizeBytes</code>.

The candidate-content digest is the SHA-256 of the RFC 8785 canonical form of:

<pre>
{
  "derivation": "mymoe-candidate-content/v1",
  "manifest": &lt;manifest descriptor&gt;,
  "changeset": &lt;changeset descriptor&gt;
}
</pre>

It intentionally excludes workflow freshness so identical immutable candidate
content has a stable content identity.

The candidate fingerprint is the SHA-256 of the RFC 8785 canonical form of:

<pre>
{
  "derivation": "mymoe-candidate-binding/v1",
  "taskFingerprint": &lt;task digest&gt;,
  "configSha256": &lt;configuration digest&gt;,
  "sourceFingerprint": &lt;source digest&gt;,
  "candidateContentSha256": &lt;candidate-content digest&gt;,
  "verificationPolicy": &lt;complete policy&gt;,
  "verificationPolicySha256": &lt;policy digest&gt;
}
</pre>

The complete binding digest additionally covers workflow identity, idempotency,
fresh challenge, artifact descriptors, policy, and lifetime.

## Verification policy

A verification policy contains exactly <code>policyId</code>,
<code>quorum</code>, and <code>verifiers</code>. Every verifier requirement
contains exactly:

- <code>verifierId</code>
- <code>adapterId</code>
- <code>keyId</code>
- <code>publicKeySha256</code>
- <code>specSha256</code>

Verifier IDs, adapter-scoped key IDs, and physical public-key digests MUST be
unique within a policy. Quorum MUST be between one and the number of configured
verifiers. <code>specSha256</code> identifies the independent verifier's own
declared verification contract.

## Attestation metadata and outcome

The attestation object contains exactly:

- <code>attestationId</code>
- <code>verifierId</code>
- <code>adapterId</code>
- <code>keyId</code>
- <code>publicKeySha256</code>
- <code>specSha256</code>
- <code>trustPolicySha256</code>
- <code>issuedAt</code>
- <code>expiresAt</code>

Verifier and trust fields MUST exactly match a requirement in the candidate
binding. The attestation lifetime MUST be non-empty, start no earlier than the
workflow, end no later than the workflow, and include the verification time.

The outcome contains exactly <code>passed</code> and <code>checks</code>.
Checks MUST be non-empty, sorted by ID, and unique. Each check contains exactly
<code>id</code>, <code>passed</code>, and <code>evidenceSha256</code>. A
conforming accepted statement has <code>passed=true</code> for the outcome and
for every check. Evidence digests bind verifier-owned result artifacts; they do
not embed those artifacts in the predicate.

## Deterministic example

[example.statement.json](example.statement.json) is a complete deterministic
statement payload. Its digests are illustrative fixtures. The contract test
rebuilds it through the production constructor and requires exact structural
and derivation equality.
