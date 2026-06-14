# AWS GenAI Pro Mastery — labs

Companion code for **AWS GenAI Pro Mastery**, a free 16-module course that takes
you from your first Amazon Bedrock call to AWS-certified (AIP-C01). Across the
build you incrementally ship **Relay**, a production-grade GenAI support agent
for **CloudCart**, a hosted e-commerce platform.

Each `module-NN/` is a self-contained, runnable snapshot. From Module 2 onward
each module ships the complete cumulative `relay/` package (inherited code is
never rewritten). Module 1 bootstraps this repo and makes the first Converse
call — there is no `relay/` package yet.

## Course map

| Module | Topic | Exam domain |
|---|---|---|
| 01 | Bedrock, models, and your first Converse call | D1 |
| 02 | Prompt engineering: templates, structured output, Prompt Management | D1 |
| 03 | The FM integration layer: model selection, routing, streaming, resilience | D1 + D2 |
| 04 | RAG foundations: chunking, embeddings, vector stores | D1 |
| 05 | Managed RAG: Bedrock Knowledge Bases, hybrid search, rerankers | D1 |
| 06 | Data pipelines for FMs: validation, multimodal input, document processing | D1 |
| 07 | Agentic AI: Strands Agents, tool calling, MCP | D2 |
| 08 | Multi-agent systems and Bedrock AgentCore: runtime, memory, HITL | D2 |
| 09 | Safety engineering: guardrails, prompt injection, hallucination control | D3 |
| 10 | Security, privacy, governance: IAM, PII, responsible AI | D3 |
| 11 | Shipping Relay: serverless deployment, integration, CI/CD | D2 |
| 12 | The token economy: cost and performance optimization | D4 |
| 13 | Evaluating GenAI apps: Bedrock Evaluations, LLM-as-a-judge, RAG metrics | D5 + D3 |
| 14 | Operating GenAI in production: observability, monitoring, troubleshooting | D4 + D5 |
| 15 | Capstone: ship Relay end-to-end | All |
| 16 | Exam strategy and mock exam | — |

Exam domains: **D1** Foundation Model Integration, Data Management, and
Compliance (31%) · **D2** GenAI Application Development and Integration (26%) ·
**D3** Responsible AI, Security, and Compliance (20%) · **D4** Cost, Performance,
and Operational Optimization (12%) · **D5** GenAI Solution Monitoring and
Evaluation (11%).

## Conventions (apply to every module)

- **Region:** `us-east-1` everywhere. **Profile:** `AWS_PROFILE=aws-genai-pro`.
- **No AWS keys in code, no `.env` for AWS.** `boto3` uses the default session,
  which respects `AWS_PROFILE`. Least-privilege IAM, one policy per module/role.
- **Inference profiles only.** Every model ID is a `us.`/`global.` inference
  profile — never a bare regional ID (those fail on-demand). Every generation
  call goes through **Converse**/ConverseStream.
- **Idempotent, verbose setup/teardown.** `setup.py` is safe to run twice and
  prints what it creates and the expected cost. `teardown.py` always runs and
  verifies nothing is left billing while idle.
- **Tests pass offline by default.** `uv run pytest` in any module needs no AWS
  credentials (botocore Stubber / moto). Tests that make real calls are marked
  `live` and gated behind `RELAY_LIVE_TESTS=1`.
- **Tooling:** Python 3.12, `uv` with a committed `uv.lock`, `boto3~=1.43`,
  `moto~=5` + `pytest` for offline tests.

## Cost — cheap, not free

The whole course costs roughly **$20–25** of AWS usage if you follow the
teardown steps. It is **not free**: Amazon Bedrock is pay-per-token, and a few
modules stand up real (cheap, on-demand) infrastructure. Every module:

- states its expected cost up front (all are well under a few dollars),
- tears down everything that would bill while idle,
- runs under the **$5/month budget alarm** that Module 1's `setup.py` creates
  and every later module keeps in place.

Set that alarm before your first token. Run each module's `teardown.py` when you
finish it.

## Quick start

```bash
cd module-01
uv sync
export RELAY_BUDGET_EMAIL="you@example.com"
uv run python setup.py
uv run python hello_bedrock.py "What does CloudCart sell?"
```
