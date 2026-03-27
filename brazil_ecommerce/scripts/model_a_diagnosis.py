import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Project/brazil_ecommerce/datasets/processed/olist_reviews_rich.csv')

# 1. Feature Engineering (Environment Only)
df['log_price'] = np.log1p(df['price'])
df['log_distance'] = np.log1p(df['distance_km'])
df['freight_ratio'] = df['freight_value'] / (df['price'] + 1)
df['delivery_delta'] = df['actual_delivery_days'] - df['estimated_delivery_days']

features = [
    'actual_delivery_days', 'delivery_delta', 'is_late', 'log_distance', 
    'log_price', 'freight_ratio', 'product_weight_g', 'product_photos_qty'
]

X = df[features].copy()
y = df['review_score'].astype(int) # LDA needs discrete classes

# Handle NaNs
X = X.fillna(X.median())

# 2. LDA Diagnosis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lda = LDA(n_components=None) # We have 5 classes, so max 4 components
X_lda = lda.fit_transform(X_scaled, y)

# 3. Explained Variance
print(f"LDA Explained Variance Ratio: {lda.explained_variance_ratio_}")

# 4. Feature conflict check: Group by features and see variance of reviews
# If same logistics lead to both 1 and 5 reviews, the "noise" is in the data
rounded_X = X.round(1)
conflict_check = df.assign(**rounded_X[features]).groupby(features)['review_score'].agg(['mean', 'std', 'count'])
high_conflict = conflict_check[conflict_check['std'] > 1.5].sort_values('count', ascending=False)

print("\nTop 5 High-Conflict Logistic Situations (Same features, very different reviews):")
print(high_conflict.head())

# 5. Visualizing the Overlap (First 2 LDA components)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette='RdYlGn', alpha=0.3)
plt.title('LDA Projection of Review Scores (Environment Features)')
plt.xlabel('LDA1 (Logistics/Delay focus)')
plt.ylabel('LDA2')
plt.savefig('Project/brazil_ecommerce/plots/eda/lda_diagnosis.png')
print("\nLDA plot saved to Project/brazil_ecommerce/plots/eda/lda_diagnosis.png")
