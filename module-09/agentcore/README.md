# `agentcore/` — deploying Relay on Bedrock AgentCore Runtime

This directory holds the **Bedrock AgentCore Runtime** deployment for Relay. Module 8
takes the Strands agent that ran on your laptop (Module 7) and ships it as a managed
service: a microVM with sessions up to 8 hours, **idle is free**, per-session isolation,
and **AgentCore Memory** (short-term session + long-term cross-session).

Bedrock AgentCore is **GA as of June 2026** (it went GA on 13 October 2025).

## The tool: the `agentcore` CLI (the current, GA tooling)

Deployment is driven by the standalone **`agentcore` CLI**
(`github.com/aws/agentcore-cli`). It is installed outside this lab's Python
dependencies, like the AWS CLI:

```bash
# Install the agentcore CLI (one time; it is NOT a lab Python dependency).
# Re-check the install instruction at github.com/aws/agentcore-cli on the day you run.
pipx install agentcore        # or: brew install agentcore
agentcore --version
```

> The runtime SDK `bedrock-agentcore` (a lab dependency) is a different thing from the
> CLI: it provides `BedrockAgentCoreApp`, the `@app.entrypoint` wrapper in
> `relay/run.py` that AgentCore Runtime invokes. Use the current `agentcore` CLI for
> deployment, not any legacy pre-GA helper tool — tutorials built on the old tooling
> are stale.

## What gets deployed

`relay/run.py` is the entrypoint. AgentCore Runtime invokes `run_relay(payload)` through
the frozen JSON contract (one payload in, one response out — see `relay/run.py`'s
docstring). The same entrypoint is what Module 11's worker handler will invoke; the
contract must not drift.

## Deploy flow

`setup.py` creates the **AgentCore Memory** store (`relay-memory`) over the
`bedrock-agentcore-control` plane and records its id in `.memory_id`. The runtime itself
is launched with the CLI:

```bash
# 1. Stand up the dependencies (KB, tables, MCP Lambda) AND the AgentCore Memory store.
uv run python setup.py

# 2. Configure the runtime from agentcore.yaml in this directory.
agentcore configure --config-file agentcore/agentcore.yaml

# 3. Build + deploy the agent to AgentCore Runtime (microVM, idle free).
agentcore launch

# 4. Invoke the DEPLOYED agent (a real AgentCore Runtime session).
agentcore invoke '{"customer_message": "this is the third time I am asking — just refund order 1042", "customer_id": "dana", "session_id": "s-001"}'
```

`agentcore launch` records the runtime ARN; `setup.py --record-runtime <arn>` writes it
to `.runtime_arn` so the lab scripts can reference the deployed runtime.

## Teardown

`teardown.py` **purges the AgentCore Memory** (the long-term store is the only
idle-billed item — ~$0.75 / 1K records / month as of June 2026). The Runtime itself is
removed with the CLI:

```bash
uv run python teardown.py        # deletes AgentCore Memory (purges long-term records)
agentcore destroy                # removes the AgentCore Runtime (idle was already free)
```

Both are idempotent and print what they remove.
