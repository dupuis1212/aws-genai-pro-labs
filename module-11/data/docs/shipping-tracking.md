---
title: Shipping, tracking, and delivery estimates
category: shipping
---

# Shipping, tracking, and delivery estimates

Shipping questions split cleanly into two groups: "where is my order right now?"
and "why does the estimate say what it says?". CloudCart shows both a live
tracking status and a delivery estimate, and they are computed differently.

## Finding a tracking number

A tracking number appears on an order the moment the carrier accepts the package,
not when the label is printed. Open the order under **Orders**, and the tracking
number and carrier are listed under **Fulfillment**. Clicking it opens the
carrier's own tracking page. If an order shows **Fulfilled** but has no tracking
number, the label was printed but the package has not yet been scanned by the
carrier.

## Reading the delivery estimate

The delivery estimate is the carrier's quoted transit time plus your store's
handling time. Handling time is the gap between an order being paid and it being
handed to the carrier; you set it per shipping zone under **Settings ->
Shipping**. A long estimate is more often a long handling time than a slow
carrier.

## Stuck or delayed packages

A package that has not moved in several days is usually waiting at a carrier
sorting facility, not lost. The tracking page shows the last scan. If there has
been no scan for more than the carrier's stated window, open a trace with the
carrier using the tracking number; CloudCart cannot move a package the carrier
already has.

## Changing an address after an order ships

Once a package has a tracking number, the address is locked with the carrier.
CloudCart cannot reroute it. The customer must contact the carrier directly, or
wait for the package to return and reship to the corrected address.

## International shipping and customs

For international orders, the estimate excludes customs clearance, which is
unpredictable. Duties and taxes are the recipient's responsibility unless you have
enabled prepaid duties under **Settings -> Shipping -> International**.
