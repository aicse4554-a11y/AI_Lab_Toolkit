from sklearn.cluster import KMeans

# Data
X = [[1], [2], [3], [10], [11], [12]]

# Create K-Means model
model = KMeans(n_clusters=2, random_state=0, n_init=10)

# Train the model
model.fit(X)

# Print cluster labels
print("Cluster Labels:", model.labels_)

# Predict cluster for new data
print("New Data Cluster:", model.predict([[4]])[0])
