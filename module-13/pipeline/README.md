# `pipeline/` — the CI/CD buildspecs for Relay (Module 11 + Module 13 eval-gate)

The CodePipeline (`cdk/relay_cdk/pipeline_stack.py`) runs these on every commit:

| Stage | File | What it does |
|---|---|---|
| **Build** | `buildspec.yml` | `uv sync`, a security scan (`pip-audit` over the locked deps), and the **offline** `tests/smoke_test.py` (moto/Stubber — no AWS calls). A red gate STOPS the pipeline before any deploy. |
| **Deploy** | (inline in the stack) | `cdk deploy RelayApiStack` — CloudFormation rolls back automatically on a failed deploy. |
| **Smoke** | `smoke_buildspec.yml` → `smoke_test_live.py` | A real `POST /tickets` round-trip + a poll of `GET /tickets/{id}` against the **deployed** API. A failed smoke stage stops the pipeline before promotion (the rollback gate). |
| **EvalGate** *(Module 13)* | `eval_buildspec.yml` → `evals/run_evals.py --gate` | The golden-set evals + the **regression gate** against the deployed build. Fails (and BLOCKS promotion) on `aggregate.grounding < 0.8` (`config.EVAL_GROUNDING_FLOOR`, the same 0.8 the M9 escalation + M14 alarm use) or a **>5-pt drop** vs the committed baseline `evals/results/run-baseline.json`. |

The **EvalGate** stage is **wired in Module 13** (it was named-but-commented in Module 11 —
no forward dependency). Module 11 built the pipeline + the smoke tests; Module 13 branches
the eval-gate onto it, after Smoke and before promotion: the smoke stage proves the API
**works**, the eval-gate proves Relay is still **good**.

## Run the smoke check by hand

```bash
uv run python pipeline/smoke_test_live.py \
  https://<api-id>.execute-api.us-east-1.amazonaws.com/prod
# POSTs data/tickets/sample.json, polls GET until a terminal status, exits non-zero on failure.
```

`smoke_test_live.py` uses only the standard library (urllib + json) — the Smoke CodeBuild
image needs nothing beyond Python.

## Run the eval-gate by hand (Module 13)

```bash
uv run python evals/run_evals.py \
  --fixture data/eval_fixtures/baseline_fixture.json \
  --out evals/results/run-latest.json \
  --gate --baseline evals/results/run-baseline.json
# prints the per-ticket table + the aggregate, then PASS/FAIL on the grounding gate;
# exits non-zero on a regression (so the CodeBuild stage fails and the pipeline blocks).
```

Swap `--fixture …` for `--live` to score the **real** deployed answers (spends tokens,
needs Bedrock access). The offline fixture path is the default so the gate is fast + free.

## Cost

A running pipeline is the **one** Module 11 resource with a real idle cost (~**$1 / active
pipeline / month**, as of June 2026). `teardown.py` deletes `RelayPipelineStack` so nothing
is idle-billed (course rule **B5**).
