---
title: Checkout error codes and what they mean
category: technical
---

# Checkout error codes and what they mean

When a checkout fails, CloudCart returns a short error code so you can act without
guessing. Each code maps to one cause and one fix. The codes below are the ones
support sees most often.

## ERR-401 — Payment authentication required

The customer's bank asked for extra verification (3-D Secure) and it was not
completed. The customer should retry and complete the bank's verification prompt.
Nothing is wrong with the store configuration.

## ERR-402 — Payment declined by the issuer

The card issuer declined the charge. This is `ERR-402`, and it is the single most
common checkout error. It is a decision by the customer's bank, not by CloudCart:
insufficient funds, a fraud hold, an expired card, or a regional block. The store
cannot override an issuer decline. The customer should use a different card or
contact their bank; retrying the same card usually fails the same way. Do NOT
treat ERR-402 as a CloudCart outage — the gateway is working; the bank said no.

## ERR-403 — Currency not supported

The store does not accept the currency the customer's card settles in. Enable the
currency under **Settings -> Payments -> Currencies**, or the customer pays in a
supported one.

## ERR-409 — Duplicate order detected

CloudCart blocked what looked like a double submission (the customer clicked
**Pay** twice). No charge was made on the blocked attempt. The first order, if it
succeeded, is the real one.

## ERR-500 — Gateway timeout

The payment gateway did not respond in time. This one IS transient and on the
provider side. The customer should retry in a few minutes. If ERR-500 persists
across many customers, check the CloudCart status page for a gateway incident.

## Reading the code in a failed order

A failed checkout records its error code under **Orders -> (the failed order) ->
Payment**. Quoting the exact code in a support ticket gets it solved fastest.
