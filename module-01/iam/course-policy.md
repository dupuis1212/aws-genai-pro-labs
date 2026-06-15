# `course-policy.json` — what each statement grants and why

Least-privilege policy for the AWS GenAI Pro Mastery course, Module 1. Attach it to a
dedicated IAM user/role used **only** for this course (`AWS_PROFILE=aws-genai-pro`, region
`us-east-1`). This is the **Module 1 footprint only**; later modules ship their own
per-component policies under `relay/iam/policies/` (introduced in Module 10).

> The policy JSON carries no comments on purpose: IAM's policy grammar is strict and
> `aws iam create-policy` rejects any unrecognized element (e.g. a `Comment` key) with
> `MalformedPolicyDocument`. The rationale lives here instead. Intent is carried in the
> JSON by descriptive `Sid` values.

| `Sid` | Why it's here |
|---|---|
| **BedrockConverseAndInvoke** | The first token. `Converse`/`ConverseStream` is the course's call path. `InvokeModel*` is included because on-demand inference-profile calls are authorized as `InvokeModelWithResponseStream` under the hood, and later modules use `invoke_model` exactly once (Titan embeddings, Module 4). The `foundation-model/*` ARNs are the underlying model resources an inference-profile call **also** authorizes — a Converse call on an inference profile evaluates both the `inference-profile/*` ARN and the `foundation-model/*` ARN. They are **resource scopes**, never `modelId`s passed to `converse()` (those are always `us.*` inference profiles). This is not a bare-regional-ID violation. |
| **BedrockReadCatalogAndAccess** | List the model catalog (`setup.py` sanity check) and read/submit the one-time Anthropic use-case form. No `InvokeModel` here — read/admin only. |
| **BudgetsManageCourseBudget** | `setup.py` creates/reconciles the $5 monthly budget and its 80% notification; `teardown.py` reads it and (at course end, `--delete-budget`) deletes it. Budgets is global, anchored to `us-east-1`; its IAM actions (`ViewBudget`/`ModifyBudget`) collapse create/update/describe/delete and cannot be resource-scoped per-budget reliably, so they are granted on the account budget namespace. `ModifyBudget` covers create, update, and delete. |
| **SnsForBudgetNotifications** | AWS Budgets uses email subscribers directly (no SNS topic needed for the 80% alert), but SNS is granted for the budget-notification plumbing and for later modules' alarms. Scoped to course-prefixed topics (`aws-genai-pro-*`). |
| **StsWhoAmI** | `setup.py`/`teardown.py` resolve the account ID (Budgets calls are account-scoped). Read-only. |
