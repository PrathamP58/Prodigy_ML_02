import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('sales_d.csv')
print("Data loaded. Shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

features = ['customer_id', 'purchase', 'spending', 'age']
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=58)
data['Cluster'] = kmeans.fit_predict(X_scaled)

for cluster in range(n_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"  Number of customers: {len(cluster_data)}")
    for feature in features:
        print(f"  Average {feature}: {cluster_data[feature].mean():.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['Cluster'], cmap='viridis')
plt.xlabel('Total Spends')
plt.ylabel('Number of Purchases')
plt.title('Customer Segments')
plt.colorbar(ticks=range(n_clusters), label='Cluster')
plt.show()
