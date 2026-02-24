import pandas as pd
import numpy as np

print('=== COMPLETE FEATURE INVENTORY FOR DELIVERY DELAY PREDICTION ===\n')

# Load all datasets
orders = pd.read_csv('olist_orders_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
items = pd.read_csv('olist_order_items_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
geolocation = pd.read_csv('olist_geolocation_dataset.csv')

# Merge into full dataset
merged = orders.merge(customers, on='customer_id', how='left')
merged = merged.merge(items, on='order_id', how='left')
merged = merged.merge(sellers, on='seller_id', how='left')
merged = merged.merge(products, on='product_id', how='left')

# Add payment info (aggregate by order)
payment_agg = payments.groupby('order_id').agg({
    'payment_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',
    'payment_installments': 'max',
    'payment_value': 'sum'
}).reset_index()
merged = merged.merge(payment_agg, on='order_id', how='left')

# Filter to delivered orders only
delivered = merged[merged['order_delivered_customer_date'].notna()].copy()

print(f"TOTAL USABLE ORDERS: {len(delivered):,}\n")
print("="*80)

# Feature categories
features = {}

# 1. TEMPORAL FEATURES
print("\n1. TEMPORAL FEATURES (from order_purchase_timestamp)")
print("-" * 80)
delivered['order_purchase_timestamp'] = pd.to_datetime(delivered['order_purchase_timestamp'])
temp_features = {
    'month': len(delivered),
    'quarter': len(delivered),
    'day_of_week': len(delivered),
    'day_of_month': len(delivered),
    'is_weekend': len(delivered),
    'hour_of_day': len(delivered),
    'year': len(delivered)
}
for feat, count in temp_features.items():
    print(f"  {feat:30s}: {count:>10,} ({count/len(delivered)*100:>5.1f}%)")
print(f"  SUBTOTAL: {len(temp_features)} features")

# 2. GEOGRAPHIC FEATURES
print("\n2. GEOGRAPHIC FEATURES")
print("-" * 80)
geo_features = {
    'customer_zip_code_prefix': delivered['customer_zip_code_prefix'].notna().sum(),
    'customer_city': delivered['customer_city'].notna().sum(),
    'customer_state': delivered['customer_state'].notna().sum(),
    'seller_zip_code_prefix': delivered['seller_zip_code_prefix'].notna().sum(),
    'seller_city': delivered['seller_city'].notna().sum(),
    'seller_state': delivered['seller_state'].notna().sum(),
    'same_state (derived)': delivered[delivered['customer_state'].notna() & delivered['seller_state'].notna()].shape[0],
    'same_city (derived)': delivered[delivered['customer_city'].notna() & delivered['seller_city'].notna()].shape[0],
    'distance_km (calculable)': delivered[(delivered['customer_zip_code_prefix'].isin(geolocation['geolocation_zip_code_prefix'])) & 
                                           (delivered['seller_zip_code_prefix'].isin(geolocation['geolocation_zip_code_prefix']))].shape[0]
}
for feat, count in geo_features.items():
    print(f"  {feat:30s}: {count:>10,} ({count/len(delivered)*100:>5.1f}%)")
print(f"  SUBTOTAL: {len(geo_features)} features")

# 3. ORDER CHARACTERISTICS
print("\n3. ORDER CHARACTERISTICS")
print("-" * 80)
order_agg = delivered.groupby('order_id').agg({
    'order_item_id': 'count',
    'price': 'sum',
    'freight_value': 'sum'
}).reset_index()
order_features = {
    'order_id': len(delivered),
    'customer_id': delivered['customer_id'].notna().sum(),
    'number_of_items (derived)': len(delivered),
    'total_order_value (derived)': delivered['price'].notna().sum(),
    'total_freight_cost (derived)': delivered['freight_value'].notna().sum(),
    'avg_item_price (derived)': delivered['price'].notna().sum(),
    'freight_to_value_ratio (derived)': delivered[(delivered['price'].notna()) & (delivered['freight_value'].notna())].shape[0]
}
for feat, count in order_features.items():
    print(f"  {feat:30s}: {count:>10,} ({count/len(delivered)*100:>5.1f}%)")
print(f"  SUBTOTAL: {len(order_features)} features")

# 4. PRODUCT FEATURES
print("\n4. PRODUCT FEATURES")
print("-" * 80)
prod_features = {
    'product_id': delivered['product_id'].notna().sum(),
    'product_category_name': delivered['product_category_name'].notna().sum(),
    'product_weight_g': delivered['product_weight_g'].notna().sum(),
    'product_length_cm': delivered['product_length_cm'].notna().sum(),
    'product_height_cm': delivered['product_height_cm'].notna().sum(),
    'product_width_cm': delivered['product_width_cm'].notna().sum(),
    'product_volume (derived)': delivered[(delivered['product_length_cm'].notna()) & 
                                           (delivered['product_height_cm'].notna()) & 
                                           (delivered['product_width_cm'].notna())].shape[0],
    'product_photos_qty': delivered['product_photos_qty'].notna().sum(),
    'is_bulky_item (derived)': delivered['product_weight_g'].notna().sum()
}
for feat, count in prod_features.items():
    print(f"  {feat:30s}: {count:>10,} ({count/len(delivered)*100:>5.1f}%)")
print(f"  SUBTOTAL: {len(prod_features)} features")

# 5. PAYMENT FEATURES
print("\n5. PAYMENT FEATURES")
print("-" * 80)
pay_features = {
    'payment_type': delivered['payment_type'].notna().sum(),
    'payment_installments': delivered['payment_installments'].notna().sum(),
    'payment_value': delivered['payment_value'].notna().sum(),
    'payment_per_installment (derived)': delivered[(delivered['payment_value'].notna()) & 
                                                     (delivered['payment_installments'].notna())].shape[0]
}
for feat, count in pay_features.items():
    print(f"  {feat:30s}: {count:>10,} ({count/len(delivered)*100:>5.1f}%)")
print(f"  SUBTOTAL: {len(pay_features)} features")

# 6. SELLER FEATURES
print("\n6. SELLER FEATURES")
print("-" * 80)
seller_orders = delivered.groupby('seller_id').size().reset_index(name='seller_order_count')
delivered_with_seller = delivered.merge(seller_orders, on='seller_id', how='left')
seller_features = {
    'seller_id': delivered['seller_id'].notna().sum(),
    'seller_order_count (derived)': len(delivered_with_seller)
}
for feat, count in seller_features.items():
    print(f"  {feat:30s}: {count:>10,} ({count/len(delivered)*100:>5.1f}%)")
print(f"  SUBTOTAL: {len(seller_features)} features")

# 7. TIME PRESSURE FEATURES (can be calculated at order time)
print("\n7. TIME PRESSURE FEATURES (available at purchase time)")
print("-" * 80)
delivered['estimated_delivery'] = pd.to_datetime(delivered['order_estimated_delivery_date'])
delivered['purchase_time'] = pd.to_datetime(delivered['order_purchase_timestamp'])
delivered['estimated_delivery_days'] = (delivered['estimated_delivery'] - delivered['purchase_time']).dt.total_seconds() / 86400
time_features = {
    'order_estimated_delivery_date': delivered['order_estimated_delivery_date'].notna().sum(),
    'estimated_delivery_days (derived)': delivered['estimated_delivery_days'].notna().sum(),
    'estimated_delivery_urgency (<7 days)': (delivered['estimated_delivery_days'] < 7).sum()
}
for feat, count in time_features.items():
    print(f"  {feat:30s}: {count:>10,} ({count/len(delivered)*100:>5.1f}%)")
print(f"  SUBTOTAL: {len(time_features)} features")

# SUMMARY
print("\n" + "="*80)
print("FEATURE SUMMARY")
print("="*80)
total_features = len(temp_features) + len(geo_features) + len(order_features) + len(prod_features) + len(pay_features) + len(seller_features) + len(time_features)
print(f"  Temporal Features:        {len(temp_features):>3}")
print(f"  Geographic Features:      {len(geo_features):>3}")
print(f"  Order Characteristics:    {len(order_features):>3}")
print(f"  Product Features:         {len(prod_features):>3}")
print(f"  Payment Features:         {len(pay_features):>3}")
print(f"  Seller Features:          {len(seller_features):>3}")
print(f"  Time Pressure Features:   {len(time_features):>3}")
print(f"  {'─'*35}")
print(f"  TOTAL FEATURES:          {total_features:>3}")
print(f"\n  Usable data points:      {len(delivered):>6,}")
print("="*80)

# Check class balance
delivered['actual_delivery'] = pd.to_datetime(delivered['order_delivered_customer_date'])
delivered['delivery_delay_days'] = (delivered['actual_delivery'] - delivered['estimated_delivery']).dt.total_seconds() / 86400
delivered['is_late'] = delivered['delivery_delay_days'] > 0

print("\n" + "="*80)
print("TARGET VARIABLE DISTRIBUTION")
print("="*80)
print(f"  Late deliveries:      {delivered['is_late'].sum():>6,} ({delivered['is_late'].mean()*100:>5.1f}%)")
print(f"  On-time deliveries:   {(~delivered['is_late']).sum():>6,} ({(~delivered['is_late']).mean()*100:>5.1f}%)")
print(f"  Class imbalance ratio: 1:{(~delivered['is_late']).sum() / delivered['is_late'].sum():.1f}")
print("="*80)

# Data quality check
print("\n" + "="*80)
print("DATA QUALITY ASSESSMENT")
print("="*80)
complete_records = delivered[
    (delivered['customer_zip_code_prefix'].notna()) &
    (delivered['seller_zip_code_prefix'].notna()) &
    (delivered['price'].notna()) &
    (delivered['freight_value'].notna()) &
    (delivered['payment_value'].notna()) &
    (delivered['product_weight_g'].notna())
].shape[0]
print(f"  Complete records (all key features): {complete_records:>6,} ({complete_records/len(delivered)*100:.1f}%)")
print("="*80)
