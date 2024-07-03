import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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

    # Store the scaled data for clustering
    season_clusters[season] = X_scaled


# Function to process and visualize clusters for a subset of seasons
def process_and_visualize_clusters(seasons_subset):
    for season in seasons_subset:
        X_scaled = season_clusters[season]

        # K-means clustering
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_

        # PCA for visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'K-means Clustering of NBA Draft Prospects ({season})')
        plt.colorbar()
        plt.show()

        # Evaluate the clustering
        score = silhouette_score(X_scaled, labels)
        print(f'Silhouette Score for {season}: {score}')


# Example: Process and visualize clusters for the first 5 seasons
seasons_subset = seasons[:1]
process_and_visualize_clusters(seasons_subset)

# You can repeat the above function call for other subsets of seasons
# Example: Process and visualize clusters for the next 5 seasons
# seasons_subset = seasons[5:10]
# process_and_visualize_clusters(seasons_subset)
