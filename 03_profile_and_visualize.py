"""
Step 3: Profile each cluster and build visualizations for the report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
df = pd.read_csv('/home/claude/segmentation/customer_data_clustered.csv')

behavior_cols = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue',
                  'DiscountAffinity', 'OnlineRatio']
demo_cols = ['Age', 'AnnualIncome', 'TenureYears']

profile = df.groupby('Cluster')[behavior_cols + demo_cols].mean().round(2)
sizes = df['Cluster'].value_counts().sort_index()
profile['Size'] = sizes
profile['SizePct'] = (sizes / sizes.sum() * 100).round(1)
print(profile)
profile.to_csv('/home/claude/segmentation/cluster_profile_summary.csv')

# ---- Auto-label clusters based on Monetary & Recency & Frequency rank ----
labels = {}
mon_rank = profile['Monetary'].rank(ascending=False)
rec_rank = profile['Recency'].rank(ascending=True)   # lower recency = better
freq_rank = profile['Frequency'].rank(ascending=False)
score = mon_rank + rec_rank + freq_rank
order = score.sort_values().index.tolist()
name_pool = ['Champions (High-Value Loyalists)', 'Loyal Everyday Shoppers',
             'Price-Sensitive Newcomers', 'At-Risk / Dormant']
for cluster_id, name in zip(order, name_pool):
    labels[cluster_id] = name
df['SegmentName'] = df['Cluster'].map(labels)
profile['SegmentName'] = profile.index.map(labels)
print(profile[['SegmentName']])

df.to_csv('/home/claude/segmentation/customer_data_clustered.csv', index=False)
profile.to_csv('/home/claude/segmentation/cluster_profile_summary.csv')

palette = sns.color_palette('Set2', n_colors=df['Cluster'].nunique())
cluster_order = sorted(df['Cluster'].unique())
color_map = {c: palette[i] for i, c in enumerate(cluster_order)}

# ---- 1. PCA scatter of segments ----
plt.figure(figsize=(7, 6))
for c in cluster_order:
    sub = df[df['Cluster'] == c]
    plt.scatter(sub['PCA1'], sub['PCA2'], s=18, alpha=0.6,
                color=color_map[c], label=labels[c])
plt.title('Customer Segments (PCA Projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(fontsize=8, loc='best')
plt.tight_layout()
plt.savefig('/home/claude/segmentation/chart_pca_segments.png', dpi=150)
plt.close()

# ---- 2. Segment size ----
plt.figure(figsize=(6, 5))
order_by_size = profile['SizePct'].sort_values(ascending=False)
bars = plt.bar([labels[i] for i in order_by_size.index], order_by_size.values,
               color=[color_map[i] for i in order_by_size.index])
plt.ylabel('% of Customers')
plt.title('Segment Size Distribution')
plt.xticks(rotation=25, ha='right', fontsize=8)
for b, v in zip(bars, order_by_size.values):
    plt.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v}%', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('/home/claude/segmentation/chart_segment_sizes.png', dpi=150)
plt.close()

# ---- 3. Behavioral profile heatmap (z-scored for comparability) ----
z_profile = (profile[behavior_cols] - profile[behavior_cols].mean()) / profile[behavior_cols].std()
plt.figure(figsize=(8, 4.5))
sns.heatmap(z_profile.rename(index=labels), annot=profile[behavior_cols].rename(index=labels),
            fmt='.1f', cmap='RdYlGn', center=0, cbar_kws={'label': 'Relative Level (z-score)'})
plt.title('Behavioral Profile by Segment (annotated with actual mean values)')
plt.tight_layout()
plt.savefig('/home/claude/segmentation/chart_behavior_heatmap.png', dpi=150)
plt.close()

# ---- 4. Monetary vs Frequency bubble chart (bubble size = avg order value) ----
plt.figure(figsize=(7, 6))
for c in cluster_order:
    sub = profile.loc[[c]]
    plt.scatter(sub['Frequency'], sub['Monetary'], s=sub['AvgOrderValue']*1.5,
                color=color_map[c], alpha=0.75, edgecolor='black', linewidth=0.5,
                label=labels[c])
plt.xlabel('Avg. Purchase Frequency (per year)')
plt.ylabel('Avg. Annual Monetary Value ($)')
plt.title('Segment Value Map (bubble size = avg order value)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('/home/claude/segmentation/chart_value_map.png', dpi=150)
plt.close()

# ---- 5. Preferred category mix by segment ----
cat_mix = pd.crosstab(df['SegmentName'], df['PreferredCategory'], normalize='index') * 100
plt.figure(figsize=(8, 5))
cat_mix.plot(kind='bar', stacked=True, colormap='tab20c', ax=plt.gca())
plt.ylabel('% of Segment')
plt.title('Preferred Product Category Mix by Segment')
plt.xticks(rotation=20, ha='right', fontsize=8)
plt.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('/home/claude/segmentation/chart_category_mix.png', dpi=150)
plt.close()

# ---- 6. Age & Income distribution by segment ----
name_order = [labels[i] for i in cluster_order]
name_color_map = {labels[i]: color_map[i] for i in cluster_order}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(data=df, x='SegmentName', y='Age', hue='SegmentName',
            palette=name_color_map, order=name_order,
            hue_order=name_order, legend=False, ax=axes[0])
axes[0].set_title('Age Distribution by Segment')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=25, labelsize=8)

sns.boxplot(data=df, x='SegmentName', y='AnnualIncome', hue='SegmentName',
            palette=name_color_map, order=name_order,
            hue_order=name_order, legend=False, ax=axes[1])
axes[1].set_title('Annual Income Distribution by Segment')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=25, labelsize=8)
plt.tight_layout()
plt.savefig('/home/claude/segmentation/chart_demographics.png', dpi=150)
plt.close()

print("\nAll charts saved.")
print(profile[['SegmentName', 'Size', 'SizePct']])
