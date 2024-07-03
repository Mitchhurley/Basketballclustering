import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('nba_player_stats.csv')

# Handle missing values for numeric columns only
numeric_features = ['age', 'player_height', 'player_weight', 'pts', 'reb', 'ast',
                    'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']
data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

# Define the features to be used for clustering
features_to_cluster = numeric_features

# Group data by season
seasons = data['season'].unique()

season_clusters = {}

for season in seasons:
    season_data = data[data['season'] == season][features_to_cluster]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(season_data)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Store the scaled and reduced data for clustering
    season_clusters[season] = X_pca


# Function to determine the optimal number of clusters using the Elbow Method
def determine_optimal_clusters(X):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Determining Optimal Number of Clusters')
    plt.show()


# Function to process and visualize clusters for a subset of seasons
def process_and_visualize_clusters(seasons_subset):
    for season in seasons_subset:
        X_pca = season_clusters[season]

        # Determine the optimal number of clusters
        #determine_optimal_clusters(X_pca)

        # Use the optimal number of clusters (example: 3)
        k = 4
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_pca)
        labels = kmeans.labels_

        # Evaluate the clustering
        silhouette_avg = silhouette_score(X_pca, labels)
        davies_bouldin_avg = davies_bouldin_score(X_pca, labels)
        print(f'Season: {season}')
        print(f'Silhouette Score: {silhouette_avg}')
        print(f'Davies-Bouldin Score: {davies_bouldin_avg}')

        # PCA for visualization (already done)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'K-means Clustering of NBA Draft Prospects ({season})')
        plt.colorbar()
        plt.show()


# Example: Process and visualize clusters for the first 5 seasons
seasons_subset = seasons[:5]
process_and_visualize_clusters(seasons_subset)

# You can repeat the above function call for other subsets of seasons
# Example: Process and visualize clusters for the next 5 seasons
# seasons_subset = seasons[5:10]
# process_and_visualize_clusters(seasons_subset)
