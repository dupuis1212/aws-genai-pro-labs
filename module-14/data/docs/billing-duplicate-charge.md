---
title: Duplicate and double charges
category: billing
---

# Duplicate and double charges

If you or one of your customers sees the same charge twice, it is almost always
one of three things: a retried payment that settled twice, a pending
authorization sitting next to the real capture, or two genuinely separate orders
that look alike. This article explains how to tell them apart and what to do.

## Pending authorizations vs. real charges

When a card is used, the bank first places an **authorization** hold, then the
charge is **captured** a moment later. For a short window both can appear in a
statement, which looks like a double charge but is not. The authorization drops
off on its own, usually within 3-5 business days. No action is needed; refunding
it does nothing because no money has actually moved.

## A retried payment that settled twice

If a payment failed, was retried, and both attempts ultimately settled, that is a
true duplicate. Open **Billing -> Transactions**, find the two charges with the
same amount and order number, and refund the later one. The refund reaches the
customer in 5-10 business days depending on their bank.

## Two separate orders

Two charges for the same amount on the same day are sometimes just two orders. The
order numbers differ. This is not a duplicate; do not refund unless the customer
confirms one order was a mistake.

## Issuing the refund

1. Open **Billing -> Transactions**.
2. Use the order number to locate the exact charge. Confirm the amount and date.
3. Click **Refund** and choose full or partial. Add a short internal note for the
   audit trail.
4. The customer is emailed automatically. Refunds cannot be reversed once
   submitted, so confirm the right transaction first.

## When the duplicate is on a subscription

A subscription billed twice in one cycle is a known edge case when a card update
overlaps a renewal. Refund the extra cycle from **Billing -> Subscriptions** and
the next renewal date is unaffected.
