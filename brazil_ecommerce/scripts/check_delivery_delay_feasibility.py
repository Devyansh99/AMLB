import pandas as pd
import numpy as np

print('=== DELIVERY DELAY FEASIBILITY CHECK ===\n')

# Load datasets
orders = pd.read_csv('olist_orders_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
items = pd.read_csv('olist_order_items_dataset.csv')
geolocation = pd.read_csv('olist_geolocation_dataset.csv')

print('Total orders:', len(orders))

# Check delivery timestamps
print('\n=== KEY TIMESTAMP COLUMNS ===')
print(f"order_purchase_timestamp: {orders['order_purchase_timestamp'].notna().sum():,} non-null ({orders['order_purchase_timestamp'].notna().mean()*100:.1f}%)")
print(f"order_delivered_customer_date: {orders['order_delivered_customer_date'].notna().sum():,} non-null ({orders['order_delivered_customer_date'].notna().mean()*100:.1f}%)")
print(f"order_estimated_delivery_date: {orders['order_estimated_delivery_date'].notna().sum():,} non-null ({orders['order_estimated_delivery_date'].notna().mean()*100:.1f}%)")

print('\n=== ORDER STATUS DISTRIBUTION ===')
print(orders['order_status'].value_counts())

# Calculate delivery delay for delivered orders
delivered = orders[orders['order_delivered_customer_date'].notna()].copy()
delivered['actual_delivery'] = pd.to_datetime(delivered['order_delivered_customer_date'])
delivered['estimated_delivery'] = pd.to_datetime(delivered['order_estimated_delivery_date'])
delivered['delivery_delay_days'] = (delivered['actual_delivery'] - delivered['estimated_delivery']).dt.total_seconds() / 86400
delivered['is_late'] = delivered['delivery_delay_days'] > 0

print('\n=== TARGET VARIABLE: IS_LATE ===')
print(f"Usable orders (with delivery dates): {len(delivered):,}")
print(f"Late orders: {delivered['is_late'].sum():,} ({delivered['is_late'].mean()*100:.1f}%)")
print(f"On-time orders: {(~delivered['is_late']).sum():,} ({(~delivered['is_late']).mean()*100:.1f}%)")

print('\n=== DELAY STATISTICS (days) ===')
print(delivered['delivery_delay_days'].describe())

print('\n=== GEOGRAPHIC DATA AVAILABILITY ===')
print(f"Customers with zip_code_prefix: {customers['customer_zip_code_prefix'].notna().sum():,} / {len(customers):,} ({customers['customer_zip_code_prefix'].notna().mean()*100:.1f}%)")
print(f"Sellers with zip_code_prefix: {sellers['seller_zip_code_prefix'].notna().sum():,} / {len(sellers):,} ({sellers['seller_zip_code_prefix'].notna().mean()*100:.1f}%)")
print(f"Geolocation entries: {len(geolocation):,}")
print(f"Unique zip codes in geolocation: {geolocation['geolocation_zip_code_prefix'].nunique():,}")

# Check if we can link orders to customer and seller locations
merged = orders.merge(customers, on='customer_id', how='left')
merged = merged.merge(items[['order_id', 'seller_id']], on='order_id', how='left')
merged = merged.merge(sellers, on='seller_id', how='left')

print('\n=== MERGED DATA FOR ML ===')
print(f"Orders after merging: {len(merged):,}")
print(f"Orders with customer zip: {merged['customer_zip_code_prefix'].notna().sum():,} ({merged['customer_zip_code_prefix'].notna().mean()*100:.1f}%)")
print(f"Orders with seller zip: {merged['seller_zip_code_prefix'].notna().sum():,} ({merged['seller_zip_code_prefix'].notna().mean()*100:.1f}%)")

# Check for geographic distance calculation feasibility
print('\n=== DISTANCE CALCULATION FEASIBILITY ===')
customer_zips_in_geoloc = merged['customer_zip_code_prefix'].isin(geolocation['geolocation_zip_code_prefix']).sum()
seller_zips_in_geoloc = merged['seller_zip_code_prefix'].isin(geolocation['geolocation_zip_code_prefix']).sum()
print(f"Orders with customer zip in geolocation table: {customer_zips_in_geoloc:,} ({customer_zips_in_geoloc/len(merged)*100:.1f}%)")
print(f"Orders with seller zip in geolocation table: {seller_zips_in_geoloc:,} ({seller_zips_in_geoloc/len(merged)*100:.1f}%)")

# Final usable dataset
final = merged[
    (merged['order_delivered_customer_date'].notna()) &
    (merged['customer_zip_code_prefix'].notna()) &
    (merged['seller_zip_code_prefix'].notna())
]
print(f'\n=== FINAL USABLE DATASET ===')
print(f"Orders with ALL required fields: {len(final):,}")
print(f"Percentage of original data: {len(final)/len(orders)*100:.1f}%")
