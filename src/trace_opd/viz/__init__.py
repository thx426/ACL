"""Read-only observation system over runs/ artifacts (docs/viz-design.md).

Kernel consumes ONLY protocol-layer schema (dump/research layouts, structured
Step flags); environment semantics live in observation plugins. Zero non-stdlib
dependencies by design — deploys onto the training host without touching venvs.
"""
