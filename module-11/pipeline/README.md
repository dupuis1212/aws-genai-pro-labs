# `pipeline/` — the CI/CD buildspecs for Relay (Module 11)

The CodePipeline (`cdk/relay_cdk/pipeline_stack.py`) runs these on every commit:

| Stage | File | What it does |
|---|---|---|
| **Build** | `buildspec.yml` | `uv sync`, a security scan (`pip-audit` over the locked deps), and the **offline** `tests/smoke_test.py` (moto/Stubber — no AWS calls). A red gate STOPS the pipeline before any deploy. |
| **Deploy** | (inline in the stack) | `cdk deploy RelayApiStack` — CloudFormation rolls back automatically on a failed deploy. |
| **Smoke** | `smoke_buildspec.yml` → `smoke_test_live.py` | A real `POST /tickets` round-trip + a poll of `GET /tickets/{id}` against the **deployed** API. A failed smoke stage stops the pipeline before promotion (the rollback gate). |

The **eval-gate** stage (golden-set regression: block on `aggregate.grounding < 0.8` or a
>5-pt drop, the same `config.GROUNDING_THRESHOLD` constant) is added in **Module 13** — it
is left commented in `pipeline_stack.py`. Module 11 builds the pipeline + the smoke tests;
Module 13 branches the eval-gate onto it.

## Run the smoke check by hand

```bash
uv run python pipeline/smoke_test_live.py \
  https://<api-id>.execute-api.us-east-1.amazonaws.com/prod
# POSTs data/tickets/sample.json, polls GET until a terminal status, exits non-zero on failure.
```

`smoke_test_live.py` uses only the standard library (urllib + json) — the Smoke CodeBuild
image needs nothing beyond Python.

## Cost

A running pipeline is the **one** Module 11 resource with a real idle cost (~**$1 / active
pipeline / month**, as of June 2026). `teardown.py` deletes `RelayPipelineStack` so nothing
is idle-billed (course rule **B5**).
