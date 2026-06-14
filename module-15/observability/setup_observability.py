"""observability/setup_observability.py — turn ON Relay's ops layer (Module 14).

One idempotent, verbose script (the brief's observable result):

    uv run python observability/setup_observability.py

does four things and prints the dashboard URL:

  1. MODEL INVOCATION LOGGING (Bedrock -> CloudWatch Logs). put_model_invocation_logging_
     configuration points Bedrock at the `/relay/bedrock/model-invocations` log group, so
     every Converse request + response is searchable in Logs Insights (tokens, latency,
     model). FREE on the Bedrock side — you pay only CloudWatch Logs storage (cents). NOT
     CloudTrail (M10 management-event audit) — the exam's distinction. The logs carry
     prompts/responses (sensitive — link M10 PII), so the log group gets a SHORT 14-day
     retention.
  2. The `relay-ops` DASHBOARD — 8 widgets answering operational + business questions
     (tokens in/out by tier, $/ticket, p95 API latency, errors/throttling, escalation rate,
     guardrail block rate, eval grounding, agent tool latency). Built from a PURE definition
     function the smoke test asserts offline.
  3. The four ALARMS — p95 latency, throttling > 0 / 5 min, cost ANOMALY DETECTION (a band,
     not a static line), and grounding < 0.8 (the ONE M9/M13 constant — gate <-> alarm <->
     escalation coherence). Each alarm notifies one SNS topic; a subscribed email is the
     on-call signal (no PagerDuty — the brief's "alarme -> email SNS suffit").
  4. The SNS topic + (if RELAY_BUDGET_EMAIL is set) the email subscription.

All names/thresholds come from relay.config. No model ID, no generation call — observability
WATCHES the existing calls. CloudWatch generative AI observability (GA re:Invent 2025) is the
native LLM/agent tracing path; the dashboard/alarms here are the operational + business layer
on top. Idempotent: re-running updates in place (PutDashboard / PutMetricAlarm / a stable
topic name) — never duplicates. teardown.py undoes all of it (B5).

Offline: every BUILDER (dashboard body, alarm specs, the logging-role policy) is a pure
function; the AWS-touching ensure_* functions take an injected client so the smoke test runs
them on moto / no AWS. Live wiring needs CloudWatch + Logs + SNS + IAM permissions.
"""

from __future__ import annotations

import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from relay import config

REGION = config.REGION

# Bedrock's put_model_invocation_logging_configuration SYNCHRONOUSLY validates that the
# logging-delivery role can write to the log group. On a clean account the role + inline
# policy we just created have not yet propagated to that validation path, so the very first
# call fails with a ValidationException ("Failed to validate permissions for log group ...
# with role ...") even though the role is correct. This is pure IAM eventual consistency.
# We retry the enable call a few times with a short backoff; a re-run (role already
# propagated) succeeds on the first try, so this only ever costs time on the cold first run.
_LOGGING_VALIDATION_RETRIES = 6
_LOGGING_VALIDATION_BACKOFF_SECONDS = 5
_NOT_FOUND = ("ResourceNotFoundException", "NotFoundException", "NoSuchEntity", "404")


def _cloudwatch():
    return boto3.client("cloudwatch", region_name=REGION)


def _logs():
    return boto3.client("logs", region_name=REGION)


def _sns():
    return boto3.client("sns", region_name=REGION)


def _iam():
    return boto3.client("iam", region_name=REGION)


def _bedrock_control():
    """The bedrock CONTROL-plane client (put_model_invocation_logging_configuration lives
    on `bedrock`, not `bedrock-runtime`)."""
    return boto3.client("bedrock", region_name=REGION)


def _sts():
    return boto3.client("sts", region_name=REGION)


# =============================================================================
# 1. Model invocation logging (Bedrock -> CloudWatch Logs).
# =============================================================================
def invocation_logging_trust_policy() -> str:
    """Trust policy: the Bedrock service may assume the logging-delivery role. PURE."""
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })


def invocation_logging_role_policy(account: str) -> str:
    """Least-privilege policy: Bedrock may WRITE invocation logs to the two log groups. PURE.

    Explicit log-group ARNs, zero wildcards beyond the log-stream suffix (the M10 least-
    privilege pattern). Bedrock needs CreateLogStream + PutLogEvents on the invocation log
    group AND its delivery-status companion group.
    """
    region = config.REGION
    groups = (config.RELAY_INVOCATION_LOG_GROUP, config.RELAY_INVOCATION_DELIVERY_LOG_GROUP)
    resources = [f"arn:aws:logs:{region}:{account}:log-group:{g}:log-stream:*"
                 for g in groups]
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "BedrockToInvocationLogGroup",
            "Effect": "Allow",
            "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
            "Resource": resources,
        }],
    })


def ensure_log_group(logs, name: str, *, retention_days: int) -> None:
    """Create a CloudWatch log group with a retention, idempotently.

    A short retention is the brief's data-cost + sensitivity control (prompts/responses are
    sensitive AND voluminous). Re-running just re-asserts the retention — never duplicates.
    """
    try:
        logs.create_log_group(logGroupName=name)
        print(f"  log group '{name}': CREATED.")
    except ClientError as err:
        if err.response["Error"]["Code"] in ("ResourceAlreadyExistsException",):
            print(f"  log group '{name}': already exists. Fine.")
        else:
            raise
    logs.put_retention_policy(logGroupName=name, retentionInDays=retention_days)
    print(f"    retention set to {retention_days} days (sensitive prompts — short).")


def ensure_invocation_logging_role(iam, account: str) -> str:
    """Create/refresh the IAM role Bedrock assumes to deliver invocation logs. Returns ARN.

    Same least-privilege pattern as the M10 component roles + the M12/M13 service roles:
    explicit ARNs, zero wildcards. IAM is FREE; teardown deletes it (B5). Idempotent.
    """
    role_name = config.RELAY_INVOCATION_LOG_ROLE_NAME
    trust = invocation_logging_trust_policy()
    try:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=trust,
            Description="Lets Amazon Bedrock deliver model-invocation logs to CloudWatch "
                        "Logs (Module 14 observability).",
        )
        arn = resp["Role"]["Arn"]
        print(f"  IAM role '{role_name}': CREATED.")
    except ClientError as err:
        if err.response["Error"]["Code"] == "EntityAlreadyExists":
            arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
            iam.update_assume_role_policy(RoleName=role_name, PolicyDocument=trust)
            print(f"  IAM role '{role_name}': already exists (trust refreshed). Fine.")
        else:
            raise
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=config.IAM_COMPONENT_POLICY_NAME,
        PolicyDocument=invocation_logging_role_policy(account),
    )
    print("    least-privilege write policy on the two log groups attached.")
    return arn


def enable_model_invocation_logging(bedrock, *, role_arn: str, sleep=time.sleep) -> None:
    """Turn ON Bedrock model-invocation logging to CloudWatch Logs. Idempotent.

    put_model_invocation_logging_configuration is a PUT — re-running overwrites the same
    config (no duplicate). We log the full text+image data the troubleshooting workflow needs
    (large/image/embedding data toggles default on); a real fleet trims these for cost +
    sensitivity (the article's retention/sampling note). This is the brief's step 2.

    Bedrock validates the delivery role's log-group permissions inline; on a clean account the
    role we just created has not propagated yet, so the first call raises a ValidationException
    ("Failed to validate permissions for log group ... with role ..."). That is IAM eventual
    consistency, not a misconfiguration — we retry with a short backoff so the cold first run
    succeeds without the operator having to re-run setup. `sleep` is injectable so the smoke
    test exercises the retry path offline without real delay.
    """
    config_kwargs = {
        "loggingConfig": {
            "cloudWatchConfig": {
                "logGroupName": config.RELAY_INVOCATION_LOG_GROUP,
                "roleArn": role_arn,
            },
            "textDataDeliveryEnabled": True,
            "imageDataDeliveryEnabled": True,
            "embeddingDataDeliveryEnabled": True,
        }
    }
    for attempt in range(1, _LOGGING_VALIDATION_RETRIES + 1):
        try:
            bedrock.put_model_invocation_logging_configuration(**config_kwargs)
            break
        except ClientError as err:
            code = err.response["Error"]["Code"]
            message = err.response["Error"].get("Message", "")
            is_propagation = (
                code == "ValidationException"
                and "validate permissions" in message
                and attempt < _LOGGING_VALIDATION_RETRIES
            )
            if not is_propagation:
                raise
            wait = _LOGGING_VALIDATION_BACKOFF_SECONDS * attempt
            print(f"    logging-role permissions not propagated yet (attempt {attempt}/"
                  f"{_LOGGING_VALIDATION_RETRIES}); waiting {wait}s for IAM consistency...")
            sleep(wait)
    print(f"  model invocation logging: ENABLED -> {config.RELAY_INVOCATION_LOG_GROUP} "
          "(FREE on the Bedrock side; you pay only CloudWatch Logs storage).")


# =============================================================================
# 2. The `relay-ops` dashboard (8 widgets) — PURE definition + the put.
# =============================================================================
def _metric_widget(title: str, metrics: list[list], *, stat: str = "Average",
                   period: int = 300, x: int = 0, y: int = 0,
                   width: int = 12, height: int = 6) -> dict:
    """One CloudWatch metric widget (the dashboard-body JSON shape). PURE."""
    return {
        "type": "metric", "x": x, "y": y, "width": width, "height": height,
        "properties": {
            "title": title,
            "region": config.REGION,
            "stat": stat,
            "period": period,
            "metrics": metrics,
            "view": "timeSeries",
        },
    }


def build_dashboard_body() -> str:
    """Build the `relay-ops` dashboard body JSON — 8 widgets. PURE (the smoke test asserts it).

    Each widget answers a QUESTION (brief §6 step 4), not decoration. Eight widgets, in the
    frozen order the article lists:
      1. tokens in/out (the FM signal classic infra omits)
      2. $/ticket (the M12 cost_cents, now a metric — the business line)
      3. p95 API latency (the "40-second answer" symptom — Lambda Duration p95)
      4. errors / throttling (FM API-integration errors — skill 5.2.2)
      5. escalation rate (Escalated averaged = the deflection KPI)
      6. guardrail block rate (M9 safety signal)
      7. eval grounding (the prod-canary quality signal the grounding alarm watches)
      8. agent tool latency (tool-calling observability — skill 4.3.4)
    The custom metrics reference the relay.config namespace + the Service=Relay dimension;
    the infra widgets (p95, throttling) read the AWS namespaces the M11 Lambdas already emit.
    """
    ns = config.RELAY_METRIC_NAMESPACE
    dim = config.METRIC_DIMENSION_SERVICE
    val = config.METRIC_SERVICE_VALUE
    worker = "RelayWorker"  # the worker Lambda's logical name (infra widgets read its metrics)

    widgets = [
        # 1. tokens in/out
        _metric_widget(
            "FM tokens in / out (per ticket)",
            [[ns, config.METRIC_INPUT_TOKENS, dim, val, {"stat": "Sum", "label": "input"}],
             [ns, config.METRIC_OUTPUT_TOKENS, dim, val, {"stat": "Sum", "label": "output"}]],
            stat="Sum", x=0, y=0,
        ),
        # 2. $/ticket
        _metric_widget(
            "$/ticket (cost_cents — M12)",
            [[ns, config.METRIC_COST_CENTS, dim, val, {"stat": "Average"}]],
            x=12, y=0,
        ),
        # 3. p95 API latency (the worker Lambda's Duration p95 — the 40s-answer symptom)
        _metric_widget(
            "API p95 latency (ms)",
            [["AWS/Lambda", "Duration", "FunctionName", worker, {"stat": "p95"}]],
            stat="p95", x=0, y=6,
        ),
        # 4. errors / throttling
        _metric_widget(
            "Errors / throttling",
            [["AWS/Lambda", "Errors", "FunctionName", worker, {"stat": "Sum",
                                                               "label": "errors"}],
             ["AWS/Lambda", "Throttles", "FunctionName", worker, {"stat": "Sum",
                                                                  "label": "throttles"}]],
            stat="Sum", x=12, y=6,
        ),
        # 5. escalation rate (Escalated averaged over the window = the rate)
        _metric_widget(
            "Escalation rate",
            [[ns, config.METRIC_ESCALATED, dim, val, {"stat": "Average"}]],
            x=0, y=12,
        ),
        # 6. guardrail block rate (M9)
        _metric_widget(
            "Guardrail block rate (M9)",
            [[ns, config.METRIC_GUARDRAIL_BLOCKED, dim, val, {"stat": "Average"}]],
            x=12, y=12,
        ),
        # 7. eval grounding (the prod canary — the grounding<0.8 alarm watches this)
        _metric_widget(
            "Eval grounding (golden-set canary)",
            [[ns, config.METRIC_EVAL_GROUNDING, dim, val, {"stat": "Average"}]],
            x=0, y=18,
        ),
        # 8. agent tool latency (skill 4.3.4)
        _metric_widget(
            "Agent tool latency (ms)",
            [[ns, config.METRIC_TOOL_LATENCY_MS, dim, val, {"stat": "Average"}]],
            x=12, y=18,
        ),
    ]
    return json.dumps({"widgets": widgets})


def ensure_dashboard(cloudwatch) -> None:
    """PUT the `relay-ops` dashboard. Idempotent (PutDashboard overwrites in place)."""
    cloudwatch.put_dashboard(
        DashboardName=config.RELAY_DASHBOARD_NAME,
        DashboardBody=build_dashboard_body(),
    )
    print(f"  dashboard '{config.RELAY_DASHBOARD_NAME}': PUT "
          f"({config.RELAY_DASHBOARD_WIDGET_COUNT} widgets).")


def dashboard_url() -> str:
    """The console URL of the `relay-ops` dashboard (printed at the end of setup)."""
    return (f"https://{config.REGION}.console.aws.amazon.com/cloudwatch/home"
            f"?region={config.REGION}#dashboards:name={config.RELAY_DASHBOARD_NAME}")


# =============================================================================
# 3. The four alarms (PURE specs + the puts) + the SNS topic.
# =============================================================================
def ensure_alarm_topic(sns) -> str:
    """Create the SNS topic alarms notify (idempotent — CreateTopic is by name). Returns ARN.

    If RELAY_BUDGET_EMAIL is set (the SAME env var the M1 budget alarm uses), subscribe the
    address — a confirmation email arrives and the operator clicks to confirm. With no
    address the topic is still created (alarms still fire + show on the dashboard); only the
    email subscription is skipped (printed, not silent).
    """
    arn = sns.create_topic(Name=config.RELAY_ALARM_TOPIC_NAME)["TopicArn"]
    print(f"  SNS topic '{config.RELAY_ALARM_TOPIC_NAME}': {arn}")
    email = os.environ.get(config.RELAY_ALARM_EMAIL_ENV, "").strip()
    if email:
        sns.subscribe(TopicArn=arn, Protocol="email", Endpoint=email)
        print(f"    email subscription requested for {email} "
              "(confirm the link in your inbox).")
    else:
        print(f"    no {config.RELAY_ALARM_EMAIL_ENV} set — topic created without an email "
              "subscription (alarms still fire; set the env var to get notified).")
    return arn


def p95_latency_alarm_spec(topic_arn: str) -> dict:
    """PutMetricAlarm kwargs for the p95-latency alarm. PURE.

    Watches the worker Lambda's Duration p95; trips when an answer takes longer than the
    threshold (the article's 40-second-answer symptom -> runbook entry "slow answers").
    """
    return {
        "AlarmName": config.ALARM_P95_LATENCY,
        "AlarmDescription": "Relay API p95 latency above threshold — see docs/runbook.md "
                            "'Slow answers'.",
        "Namespace": "AWS/Lambda",
        "MetricName": "Duration",
        "Dimensions": [{"Name": "FunctionName", "Value": "RelayWorker"}],
        "ExtendedStatistic": "p95",
        "Period": config.ALARM_PERIOD_SECONDS,
        "EvaluationPeriods": 1,
        "Threshold": float(config.ALARM_P95_LATENCY_THRESHOLD_MS),
        "ComparisonOperator": "GreaterThanThreshold",
        "TreatMissingData": "notBreaching",
        "AlarmActions": [topic_arn],
    }


def throttling_alarm_spec(topic_arn: str) -> dict:
    """PutMetricAlarm kwargs for the throttling alarm (ThrottlingException > 0 / 5 min). PURE.

    Trips on a single throttle in the period — a quota/backoff signal (skill 5.2.2) -> runbook
    entry "Throttling bursts". Reads the worker Lambda's Throttles metric.
    """
    return {
        "AlarmName": config.ALARM_THROTTLING,
        "AlarmDescription": "Relay throttling detected — see docs/runbook.md "
                            "'Throttling bursts'.",
        "Namespace": "AWS/Lambda",
        "MetricName": "Throttles",
        "Dimensions": [{"Name": "FunctionName", "Value": "RelayWorker"}],
        "Statistic": "Sum",
        "Period": config.ALARM_PERIOD_SECONDS,
        "EvaluationPeriods": 1,
        "Threshold": float(config.ALARM_THROTTLING_THRESHOLD),
        "ComparisonOperator": "GreaterThanThreshold",
        "TreatMissingData": "notBreaching",
        "AlarmActions": [topic_arn],
    }


def grounding_alarm_spec(topic_arn: str) -> dict:
    """PutMetricAlarm kwargs for the grounding<0.8 alarm. PURE.

    The PROD-canary quality alarm: the EvalGrounding metric (the golden set re-run in prod)
    below the ONE M9/M13 0.8 floor (config.ALARM_GROUNDING_THRESHOLD) -> runbook entry "Vague
    answers / grounding drop". The SAME 0.8 the deploy gate (M13) and the per-answer
    escalation (M9) use — gate <-> alarm <-> escalation coherence.
    """
    return {
        "AlarmName": config.ALARM_GROUNDING,
        "AlarmDescription": "Relay eval grounding below the 0.8 floor (the M9/M13 constant) "
                            "— see docs/runbook.md 'Vague answers / grounding drop'.",
        "Namespace": config.RELAY_METRIC_NAMESPACE,
        "MetricName": config.METRIC_EVAL_GROUNDING,
        "Dimensions": [{"Name": config.METRIC_DIMENSION_SERVICE,
                        "Value": config.METRIC_SERVICE_VALUE}],
        "Statistic": "Minimum",
        "Period": 86400,  # one day — the golden-set canary runs (at most) daily
        "EvaluationPeriods": 1,
        "Threshold": float(config.ALARM_GROUNDING_THRESHOLD),
        "ComparisonOperator": "LessThanThreshold",
        "TreatMissingData": "notBreaching",
        "AlarmActions": [topic_arn],
    }


def cost_anomaly_alarm_spec(topic_arn: str) -> dict:
    """PutMetricAlarm kwargs for the cost ANOMALY-DETECTION alarm. PURE.

    NOT a static dollar line (a growing app would trip it on every busy day) — a CloudWatch
    ANOMALY-DETECTION band: it learns the daily $/ticket's normal range and trips when a value
    falls OUTSIDE a band config.ALARM_COST_ANOMALY_BAND_STDDEV std-devs wide (skill 4.3.2,
    "cost anomaly detection / token burst") -> runbook entry "Cost doubled, no extra traffic".
    Anomaly-detection alarms use Metrics[] with an ANOMALY_DETECTION_BAND expression + a
    ThresholdMetricId, not a scalar Threshold.
    """
    band = config.ALARM_COST_ANOMALY_BAND_STDDEV
    return {
        "AlarmName": config.ALARM_COST_ANOMALY,
        "AlarmDescription": "Relay daily cost outside its learned band (anomaly detection) — "
                            "see docs/runbook.md 'Cost doubled, no extra traffic'.",
        "Metrics": [
            {
                "Id": "m1",
                "MetricStat": {
                    "Metric": {
                        "Namespace": config.RELAY_METRIC_NAMESPACE,
                        "MetricName": config.METRIC_COST_CENTS,
                        "Dimensions": [{"Name": config.METRIC_DIMENSION_SERVICE,
                                        "Value": config.METRIC_SERVICE_VALUE}],
                    },
                    "Period": 86400,
                    "Stat": "Sum",
                },
                "ReturnData": True,
            },
            {
                "Id": "ad1",
                "Expression": f"ANOMALY_DETECTION_BAND(m1, {band})",
                "Label": "CostCents (expected band)",
                "ReturnData": True,
            },
        ],
        "ThresholdMetricId": "ad1",
        "ComparisonOperator": "GreaterThanUpperThreshold",
        "EvaluationPeriods": 1,
        "TreatMissingData": "notBreaching",
        "AlarmActions": [topic_arn],
    }


def alarm_specs(topic_arn: str) -> list[dict]:
    """The four alarm specs, in the order config.RELAY_ALARM_NAMES lists them. PURE."""
    return [
        p95_latency_alarm_spec(topic_arn),
        throttling_alarm_spec(topic_arn),
        cost_anomaly_alarm_spec(topic_arn),
        grounding_alarm_spec(topic_arn),
    ]


def ensure_alarms(cloudwatch, topic_arn: str) -> int:
    """PUT the four alarms. Idempotent (PutMetricAlarm overwrites by name). Returns the count.

    The cost-anomaly alarm is metric-math (a band) so it goes through put_metric_alarm with a
    Metrics[]/ThresholdMetricId shape; the other three are scalar-threshold alarms.
    """
    specs = alarm_specs(topic_arn)
    for spec in specs:
        cloudwatch.put_metric_alarm(**spec)
        print(f"  alarm '{spec['AlarmName']}': PUT (-> runbook entry).")
    return len(specs)


# =============================================================================
# The orchestration (called by setup.py module_14_setup + runnable directly).
# =============================================================================
def setup_observability(*, account: str, cloudwatch=None, logs=None, sns=None, iam=None,
                        bedrock=None) -> str:
    """Wire the whole ops layer; return the dashboard URL. Clients injectable for tests.

    Order: log groups + logging role -> enable invocation logging -> SNS topic -> dashboard
    -> alarms. Idempotent throughout. Returns the dashboard URL the caller prints.
    """
    cloudwatch = cloudwatch or _cloudwatch()
    logs = logs or _logs()
    sns = sns or _sns()
    iam = iam or _iam()
    bedrock = bedrock or _bedrock_control()

    print("\nModule 14 — Relay's ops layer (observability):")
    print("\nModel invocation logging (Bedrock -> CloudWatch Logs; NOT CloudTrail):")
    ensure_log_group(logs, config.RELAY_INVOCATION_LOG_GROUP,
                     retention_days=config.RELAY_INVOCATION_LOG_RETENTION_DAYS)
    ensure_log_group(logs, config.RELAY_INVOCATION_DELIVERY_LOG_GROUP,
                     retention_days=config.RELAY_INVOCATION_LOG_RETENTION_DAYS)
    role_arn = ensure_invocation_logging_role(iam, account)
    enable_model_invocation_logging(bedrock, role_arn=role_arn)

    print("\nSNS topic alarms notify (email on-call signal):")
    topic_arn = ensure_alarm_topic(sns)

    print(f"\nDashboard '{config.RELAY_DASHBOARD_NAME}' "
          f"({config.RELAY_DASHBOARD_WIDGET_COUNT} widgets):")
    ensure_dashboard(cloudwatch)

    print("\nAlarms (each -> a docs/runbook.md entry):")
    ensure_alarms(cloudwatch, topic_arn)

    return dashboard_url()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        print("Usage: uv run python observability/setup_observability.py", file=sys.stderr)
        return 1

    print("Setting up Module 14 — Relay's observability layer:")
    print("Adds: model invocation logging (-> CloudWatch Logs, FREE on the Bedrock side; you "
          "pay\nonly Logs storage, cents), the 'relay-ops' dashboard (8 widgets), and four "
          "alarms\n(p95 latency, throttling, cost ANOMALY DETECTION, grounding<0.8 — the one "
          "M9/M13\nconstant) wired to an SNS email topic. Custom metrics flow from the worker "
          "(by\naddition) + run_evals.py. NO new model call: observability WATCHES Relay's "
          "existing\ncalls. CloudWatch generative AI observability (GA re:Invent 2025) is the "
          "native\nLLM/agent tracing path; X-Ray is the AWS tool for cross-service tracing on "
          "the\nAPI->SQS->agent path (renvoi M11 boundary tracing). Expected cost: < $1 (Logs "
          "storage cents + "
          "a handful of\ncustom metrics + the dashboard/alarms). teardown.py removes all of "
          "it (B5).\n")

    try:
        account = config.account_id(_sts())
    except NoCredentialsError:
        print("  [FAIL] no AWS credentials — set AWS_PROFILE=aws-genai-pro.", file=sys.stderr)
        return 1

    try:
        url = setup_observability(account=account)
    except ClientError as err:
        code = err.response["Error"]["Code"]
        message = err.response["Error"]["Message"]
        print(f"\nAWS call failed ({code}):\n  {message}\n\n"
              "If this is AccessDenied, your course IAM role needs: bedrock:PutModelInvocation"
              "LoggingConfiguration;\nlogs:CreateLogGroup/PutRetentionPolicy on /relay/bedrock"
              "/*; cloudwatch:PutDashboard/\nPutMetricAlarm; sns:CreateTopic/Subscribe; "
              "iam:CreateRole/PutRolePolicy/PassRole on\n'" + config.RELAY_INVOCATION_LOG_ROLE_NAME
              + "'. See lab.md.", file=sys.stderr)
        return 1

    print("\nDone. Relay is OBSERVED. Open the dashboard:")
    print(f"  {url}")
    print("\nExplore 3 invocations in CloudWatch Logs Insights (queries provided):")
    print(f"  log group: {config.RELAY_INVOCATION_LOG_GROUP}")
    print("  query    : observability/queries/invocations_tokens_latency.logsinsights")
    print("\nThen inject a fault and follow the runbook (docs/runbook.md):")
    print("  uv run python observability/inject_fault.py --list")
    print("  uv run python observability/inject_fault.py --fault context-overflow")
    print("  # diagnose with the dashboard + Logs Insights, remedy, then:")
    print("  uv run python observability/inject_fault.py --restore")
    print("  uv run python evals/run_evals.py --fixture "
          "data/eval_fixtures/baseline_fixture.json \\")
    print("    --out evals/results/run-postfix-context-overflow.json  # back to baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
