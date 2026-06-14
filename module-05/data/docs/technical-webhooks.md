---
title: Setting up webhooks and the Orders API
category: technical
---

# Setting up webhooks and the Orders API

CloudCart can push events to your systems as they happen (webhooks) and let you
read and write store data on demand (the REST API). Most integrations use both: a
webhook to know something changed, then an API call to act on it.

## Creating a webhook endpoint

1. Go to **Settings -> Developer -> Webhooks**.
2. Click **Add endpoint** and paste a publicly reachable HTTPS URL. Plain HTTP is
   rejected; the endpoint must present a valid TLS certificate.
3. Select the events you want, for example `order.created`, `order.fulfilled`, or
   `refund.created`.
4. Save. CloudCart sends a test ping immediately so you can confirm your endpoint
   returns a 2xx.

## Verifying webhook signatures

Every webhook carries an `X-CloudCart-Signature` header: an HMAC-SHA256 of the raw
request body using your endpoint's signing secret. Recompute it on your side and
compare before trusting the payload. A mismatch means the request did not come
from CloudCart and must be rejected. Verify against the RAW body, before any JSON
parsing reserializes it.

## Retries and idempotency

If your endpoint does not return a 2xx within 10 seconds, CloudCart retries with
exponential backoff for up to 24 hours. Because retries happen, your handler must
be idempotent: use the `event_id` to ignore an event you have already processed,
so a redelivered `order.created` does not create a second record.

## Reading orders from the API

Authenticate API calls with a key from **Settings -> Developer -> API keys**.
`GET /v1/orders` lists orders with cursor pagination; `GET /v1/orders/{id}`
returns one. Rate limits are 4 requests per second per key, returned in the
`X-RateLimit-Remaining` header; a 429 means slow down and retry after the
`Retry-After` window.

## Sandbox vs. live keys

Test against a sandbox key first — it talks to a separate test store and never
moves real money. Live keys start with `ck_live_` and sandbox keys with
`ck_test_`; mixing them is the most common integration mistake.
