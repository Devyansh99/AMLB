# Phase 1: Data Cleaning and Merge

## Purpose
Load, clean, and merge all 9 Olist tables into a single order-level dataset ready for feature engineering.

## Input
- 9 CSVs in `datasets/raw/` (orders, items, payments, reviews, products, sellers, customers, geolocation, category translation)
- Total: 99,441 orders, 112,650 items, 103,886 payments, 32,951 products, 3,095 sellers, 99,441 customers, 1M geolocation rows

## Key Decisions
- **Target variable**: `is_late = 1` if delivered after estimated date. Added 11:59:59 buffer to estimated date to avoid false-late labels on same-day deliveries.
- **Geolocation**: Used centroid (mean lat/lng per zip) instead of `drop_duplicates` to average out GPS noise across 1M rows down to 19,015 unique zips.
- **Aggregation**: Order-level (not item-level). Multi-item orders aggregated to 1 row per `order_id` — sum for price/freight, max for weight/dimensions, first for seller/category.
- **Payments**: Summed payment_value, took max installments, mode for payment_type. Dropped `not_defined` type.
- **Missing values**: Category median for product dimensions, state centroid for missing geolocation, "unknown" for missing category.
- **Date filtering**: Excluded Sep-Dec 2016 (platform launch, few orders) and Sep-Oct 2018 (truncated data).

## Records Removed
- Non-delivered orders: 2,963 (kept only `order_status == 'delivered'`)
- Null delivery date: 8
- Invalid timestamps (approval before purchase): 14
- Incomplete months: 267

## Output
- `datasets/processed/olist_merged_clean.csv`
- 96,189 rows x 34 columns
- Date range: Jan 2017 - Aug 2018
- Late rate: 7.94% (7,637 late orders)
- Mean distance: 601 km, mean actual delivery: 12.5 days, mean estimated: 24.1 days
