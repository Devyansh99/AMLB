import pandas as pd
import os
import numpy as np

# Paths
raw_path = 'Project/brazil_ecommerce/datasets/raw/'
processed_path = 'Project/brazil_ecommerce/datasets/processed/'

print("Loading datasets...")
orders = pd.read_csv(os.path.join(raw_path, 'olist_orders_dataset.csv'))
reviews = pd.read_csv(os.path.join(raw_path, 'olist_order_reviews_dataset.csv'))
order_items = pd.read_csv(os.path.join(raw_path, 'olist_order_items_dataset.csv'))
products = pd.read_csv(os.path.join(raw_path, 'olist_products_dataset.csv'))
sellers = pd.read_csv(os.path.join(raw_path, 'olist_sellers_dataset.csv'))
customers = pd.read_csv(os.path.join(raw_path, 'olist_customers_dataset.csv'))
geo = pd.read_csv(os.path.join(raw_path, 'olist_geolocation_dataset.csv'))

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371 
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# 1. Geolocation
geo_clean = geo.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'median',
    'geolocation_lng': 'median'
}).reset_index()

# 2. Reviews
reviews_clean = reviews.groupby('order_id')['review_score'].mean().reset_index()

# 3. Order Items
items_agg = order_items.groupby('order_id').agg({
    'seller_id': 'first',
    'product_id': 'first',
    'price': 'sum',
    'freight_value': 'sum',
    'order_item_id': 'count'
}).rename(columns={'order_item_id': 'n_items'}).reset_index()

# 4. Master Merge
df = orders.merge(reviews_clean, on='order_id', how='inner')
df = df.merge(items_agg, on='order_id', how='left')
df = df.merge(products, on='product_id', how='left')
df = df.merge(sellers, on='seller_id', how='left')
df = df.merge(customers, on='customer_id', how='left')

# 5. Geolocation
df = df.merge(geo_clean, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
df = df.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}).drop(columns='geolocation_zip_code_prefix')
df = df.merge(geo_clean, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
df = df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}).drop(columns='geolocation_zip_code_prefix')

df['distance_km'] = haversine(df['customer_lat'], df['customer_lng'], df['seller_lat'], df['seller_lng'])

# 6. Time Features (The IEEE Prep Time)
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
df['order_delivered_carrier_date'] = pd.to_datetime(df['order_delivered_carrier_date'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

# preparation_time = how long the seller took to give it to the carrier
df['preparation_days'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days

# actual_delivery_days = total time
df['actual_delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
df['estimated_delivery_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
df['delivery_delay_days'] = df['actual_delivery_days'] - df['estimated_delivery_days']
df['is_late'] = (df['delivery_delay_days'] > 0).astype(int)

# 7. Seller Experience
seller_counts = df.groupby('seller_id').size().to_dict()
df['seller_total_orders'] = df['seller_id'].map(seller_counts)

# Drop missing targets
df = df.dropna(subset=['review_score', 'actual_delivery_days', 'distance_km'])

# Save
output_file = os.path.join(processed_path, 'olist_reviews_rich.csv')
df.to_csv(output_file, index=False)
print(f"Rich Dataset Final: {df.shape[0]} rows, {df.shape[1]} columns.")
