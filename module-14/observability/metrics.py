"""observability/metrics.py — emit Relay's custom CloudWatch metrics (Module 14).

The dashboard and the alarms are only as good as the metrics behind them. Classic infra
dashboards show Lambda duration and API 5xx; they have NO concept of $/ticket, escalation
rate, guardrail block rate, or eval grounding. This module is the ONE place Relay PUBLISHES
those FM/business metrics, two ways:

  - EMF (Embedded Metric Format): print a specially-shaped JSON line to stdout. Inside a
    Lambda (the worker), the CloudWatch agent parses that line and turns it into metrics
    for FREE — no extra API call, no extra latency on the customer's ticket. This is the
    PREFERRED path for the worker (skill 4.3.2: emit metrics from the log line you already
    write). `emit_emf()` returns the dict it logs so a test can assert the shape offline.
  - PutMetricData: a direct CloudWatch API call. Used by the eval harness (run_evals.py),
    which runs OUTSIDE Lambda, to push the last run's aggregate grounding as a metric the
    grounding<0.8 alarm watches. INJECTABLE (pass a stubbed client) so tests stay offline;
    with no client and no live flag it is a documented no-op (never a silent AWS call in a
    unit test).

Every metric name + the namespace + the one bounded dimension (Service=Relay) come from
relay.config — so the dashboard/alarm builder and this emitter never drift, and metric
CARDINALITY (what CloudWatch bills, brief §7) stays low (we deliberately do NOT key by
ticket_id). No model ID, no generation call here — pure metric plumbing.
"""

from __future__ import annotations

import json
from typing import Any

from relay import config


def _dimensions() -> list[dict[str, str]]:
    """The single bounded dimension every Relay metric carries: Service=Relay.

    One small dimension keeps cardinality — and the CloudWatch bill (brief §7) — low. We do
    NOT add ticket_id (that would mint a metric per ticket); a per-ticket value belongs in
    the invocation logs / the record, not in a dimension.
    """
    return [{"Name": config.METRIC_DIMENSION_SERVICE, "Value": config.METRIC_SERVICE_VALUE}]


# --- The metric set a single ticket produces ----------------------------------
def ticket_metrics(
    *,
    cost_cents: float,
    input_tokens: int,
    output_tokens: int,
    escalated: bool,
    guardrail_blocked: bool,
    tool_latency_ms: float | None = None,
) -> list[dict[str, Any]]:
    """Build the list of {name, value, unit} metrics ONE processed ticket emits.

    Mirrors exactly what the worker already KNOWS at the end of a ticket (the metered cost,
    the token usage, the final status, whether the guardrail intervened) — observability is
    emitting facts the worker computed, never re-deriving them. tool_latency_ms is optional
    (only present when the agent called a tool). The names are the config METRIC_* constants.
    """
    metrics: list[dict[str, Any]] = [
        {"name": config.METRIC_COST_CENTS, "value": round(float(cost_cents), 6),
         "unit": "Count"},
        {"name": config.METRIC_INPUT_TOKENS, "value": int(input_tokens), "unit": "Count"},
        {"name": config.METRIC_OUTPUT_TOKENS, "value": int(output_tokens), "unit": "Count"},
        # Escalated / GuardrailBlocked are 0/1 gauges: averaged over a window they ARE the
        # escalation rate / guardrail block rate the dashboard widget shows.
        {"name": config.METRIC_ESCALATED, "value": 1 if escalated else 0, "unit": "Count"},
        {"name": config.METRIC_GUARDRAIL_BLOCKED,
         "value": 1 if guardrail_blocked else 0, "unit": "Count"},
    ]
    if tool_latency_ms is not None:
        metrics.append({"name": config.METRIC_TOOL_LATENCY_MS,
                        "value": round(float(tool_latency_ms), 3), "unit": "Milliseconds"})
    return metrics


# --- EMF (the FREE, in-log emission path — used by the worker) -----------------
def build_emf(metrics: list[dict[str, Any]], *, extra: dict[str, Any] | None = None) -> dict:
    """Build a CloudWatch EMF log object from a list of {name, value, unit} metrics.

    EMF embeds a `_aws` block telling the CloudWatch agent which top-level fields ARE metrics
    (and their units), so logging this JSON line publishes the metrics with no extra API
    call. The namespace + the one dimension come from relay.config. `extra` adds non-metric
    context fields (e.g. ticket_id, status) for log search WITHOUT minting a metric — they
    are NOT listed in the metric directives, so they cost nothing as dimensions. Returns the
    dict (the caller logs json.dumps of it); a test asserts the shape without touching AWS.
    """
    directives = {
        "Namespace": config.RELAY_METRIC_NAMESPACE,
        "Dimensions": [[config.METRIC_DIMENSION_SERVICE]],
        "Metrics": [{"Name": m["name"], "Unit": m["unit"]} for m in metrics],
    }
    body: dict[str, Any] = {
        "_aws": {
            # CloudWatch wants milliseconds since epoch; the agent stamps it on ingest if we
            # omit it, but emitting it makes the line self-describing for offline reads/tests.
            "CloudWatchMetrics": [directives],
        },
        config.METRIC_DIMENSION_SERVICE: config.METRIC_SERVICE_VALUE,
    }
    for m in metrics:
        body[m["name"]] = m["value"]
    if extra:
        # Context fields for Logs Insights search — NOT metrics (not in the directives).
        for key, value in extra.items():
            if key not in body:
                body[key] = value
    return body


def emit_emf(metrics: list[dict[str, Any]], *, extra: dict[str, Any] | None = None,
             printer=print) -> dict:
    """Log one EMF line so CloudWatch turns it into metrics for free. Returns the EMF dict.

    `printer` is injectable (defaults to print) so a test captures the line without a real
    stdout dependency, and the worker can route it through the Lambda logger. A no-metric
    list is a clean no-op (nothing to publish) — never an empty malformed line.
    """
    if not metrics:
        return {}
    body = build_emf(metrics, extra=extra)
    printer(json.dumps(body))
    return body


# --- PutMetricData (the direct path — used by run_evals.py, OUTSIDE Lambda) ----
def put_metrics(metrics: list[dict[str, Any]], *, cloudwatch=None) -> int:
    """Publish metrics via CloudWatch PutMetricData. Returns the count published.

    INJECTABLE: pass a (stubbed) cloudwatch client; with no client this is a DOCUMENTED
    no-op (returns 0) so importing/using the emitter in a unit test never makes a silent AWS
    call. The eval harness passes a real client behind its --live flag. Names/namespace/
    dimension all come from relay.config; a value's unit defaults to Count.
    """
    if not metrics:
        return 0
    if cloudwatch is None:
        # No client supplied: the caller is offline (a unit test or a dry run). Do nothing —
        # never reach for a default boto3 client implicitly (that would hit AWS in a test).
        return 0
    metric_data = [
        {
            "MetricName": m["name"],
            "Dimensions": _dimensions(),
            "Value": float(m["value"]),
            "Unit": m.get("unit", "Count"),
        }
        for m in metrics
    ]
    # PutMetricData accepts up to 1000 metrics per call; Relay emits a handful, so one call.
    cloudwatch.put_metric_data(Namespace=config.RELAY_METRIC_NAMESPACE, MetricData=metric_data)
    return len(metric_data)


def emit_eval_grounding(grounding: float, *, cloudwatch=None) -> dict[str, Any]:
    """Emit the last eval run's AGGREGATE grounding as the EvalGrounding metric.

    This is the metric the `relay-ops` grounding<0.8 alarm watches — the golden set used as a
    PRODUCTION CANARY (skill 4.3.6 / brief §6 step 3): a scheduled re-run of evals pushes its
    grounding here, and a drop below the ONE 0.8 floor (config.ALARM_GROUNDING_THRESHOLD,
    which IS the M9/M13 constant) trips the same alarm the deploy gate enforces. Returns the
    single-metric dict; publishes it when a cloudwatch client is supplied, else a no-op.
    """
    metric = {"name": config.METRIC_EVAL_GROUNDING,
              "value": round(float(grounding), 4), "unit": "None"}
    put_metrics([metric], cloudwatch=cloudwatch)
    return metric
