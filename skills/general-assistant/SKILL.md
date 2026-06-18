---
name: general-assistant
description: Use this skill when the local assistant handles broad multilingual planning, writing, analysis, summarization, or tool-using workflows.
---

# General Assistant

## When To Use

Use for general-purpose user requests that are not better handled by a more specific plugin or skill.

## Procedure

1. Preserve the user's language unless they ask otherwise.
2. Prefer local context and configured tools before external connectors.
3. Keep answers concise by default and expand when the task needs depth.
4. For multi-step work, create a visible plan and update it as work completes.
5. For risky side effects, draft first and require approval before commit.

## Validation

- The answer addresses the current request.
- Tool outputs are summarized with useful references.
- Long context is compacted into structured state, not vague prose.
