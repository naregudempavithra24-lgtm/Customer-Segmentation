"""
Step 2: Preprocess data, find optimal k, fit KMeans, save cluster assignments.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('/home/claude/segmentation/customer_data.csv')

# ---- Feature selection for clustering ----
# Use behavioral + key demographic signals. Scale everything.
features = ['Age', 'AnnualIncome', 'TenureYears', 'Recency', 'Frequency',
            'Monetary', 'AvgOrderValue', 'DiscountAffinity', 'OnlineRatio']
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Determine optimal k: elbow + silhouette ----
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(K_range), inertias, marker='o', color='#4C72B0')
axes[0].set_title('Elbow Method (Inertia)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')

axes[1].plot(list(K_range), sil_scores, marker='o', color='#DD8452')
axes[1].set_title('Silhouette Score by k')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
plt.tight_layout()
plt.savefig('/home/claude/segmentation/chart_k_selection.png', dpi=150)
plt.close()

best_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"Silhouette scores: {dict(zip(K_range, np.round(sil_scores,3)))}")
print(f"Chosen k (best silhouette): {best_k}")

# Note: silhouette peaks at k=2 because behavioral features were generated as
# overlapping continuous distributions (realistic - real customers don't fall
# into hard clusters). We select k=4 for business interpretability & actionability,
# since it separates customers into distinct, targetable marketing segments
# without a large silhouette penalty (score stays within ~0.06 of the k=2 peak).
k_final = 4
print(f"Final k used: {k_final} (chosen for business interpretability)")

km_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X_scaled)

# ---- PCA for 2D visualization ----
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = pcs[:, 0], pcs[:, 1]
print(f"Explained variance by 2 PCs: {pca.explained_variance_ratio_.sum():.2%}")

df.to_csv('/home/claude/segmentation/customer_data_clustered.csv', index=False)
print(df['Cluster'].value_counts())
