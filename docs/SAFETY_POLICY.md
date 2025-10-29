# Safety Policy

## Model Safety
- Enforce dataset lineage: manifests must include source provenance before promotion.
- Maintain evaluation rubrics for bias, toxicity, and hallucination checks.
- Require human-in-the-loop sign-off for any model destined for production surfaces.

## Operational Safety
- Access to infrastructure scripts gated via IAM & audited.
- Secrets managed through environment variables and not committed to source control.
- Distributed launches validate node health before joining the training job to avoid partial failure.

## Incident Response
- On evaluation failure or key metric regression, automatically halt deployment pipelines.
- Archive logs/checkpoints for post-mortem; run `nexa_feedback` workflow to capture remediation steps.
