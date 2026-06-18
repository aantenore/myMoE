# Distillation Plan

## Objective

Distill the expensive part of the system into local artifacts:

1. a router classifier that chooses experts better than keyword rules,
2. optionally a compact student model for frequent tasks.

## Stage 1: Route Label Distillation

Use a stronger teacher to label prompts with:

- primary expert,
- fallback expert,
- confidence,
- rationale category,
- expected output risk.

Example label:

```json
{
  "prompt_id": "arch_001",
  "primary": "architect",
  "fallback": "general",
  "confidence": 0.82,
  "reason": "architecture_design",
  "risk": "needs_specificity"
}
```

Train a local lightweight classifier over prompt features. Start with deterministic features:

- keyword counts,
- code block presence,
- language hints,
- prompt length buckets,
- task verbs.

Only move to embeddings if deterministic features plateau.

## Stage 2: Expert Answer Distillation

For repeated task families, collect:

- prompt,
- routed expert outputs,
- teacher critique,
- corrected target answer.

Fine-tune a small local student only if:

- system-level MoE latency is too high,
- tasks are repetitive enough,
- evals show the student preserves quality.

## Stage 3: Quality Gates

Every distilled artifact must beat the baseline on:

- routing accuracy,
- answer completeness,
- latency,
- local memory footprint,
- failure transparency.

Do not accept a distilled model that is faster but hides uncertainty.

