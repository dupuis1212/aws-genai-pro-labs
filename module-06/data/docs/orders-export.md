---
title: Exporting your order history
category: orders
---

# Exporting your order history

CloudCart keeps a full record of every order placed through your store. You can
export that history at any time, whether you are reconciling your books, migrating
to another platform, or simply keeping an offline backup before you downgrade your
plan.

## Where the export lives

The export tool is under **Settings -> Data & Privacy -> Export data**. It is not
on the Orders page itself, which is the single most common reason customers cannot
find it. The Orders page shows and filters orders; it does not produce a file.

## Exporting to CSV

1. Open **Settings -> Data & Privacy -> Export data**.
2. Under **What to export**, tick **Orders**. You can also tick Customers and
   Products to export them in the same job.
3. Choose a date range, or leave it as **All time** for the complete history.
4. Click **Start export**. CloudCart prepares the file in the background.
5. When the job finishes you receive an email with a secure download link. Large
   stores can take a few minutes; the link stays valid for 24 hours.

The orders export is a single CSV with one row per order line, including the order
number, date, customer, SKU, quantity, unit price, tax, and fulfillment status.

## Before you downgrade or close your store

If you are downgrading to a plan with a shorter data-retention window, or closing
your store, export your full order history first. Once a downgrade takes effect,
orders older than the new plan's retention window are no longer visible in the
dashboard, though they are never deleted on the back end within the legal
retention period.

## Automating the export

Stores on the Growth plan and above can schedule a recurring export (daily,
weekly, or monthly) from the same screen, or pull orders programmatically from the
Orders API. See the developer docs for the API; the scheduled export needs no
code.
