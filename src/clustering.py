from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib


# build path relative to the project and read file (avoids backslash escape issues)
data_path = Path(__file__).resolve().parent.parent / 'data' / 'u.item'
df = pd.read_csv(data_path, header=None, sep='|', encoding='latin-1')
movies = df.drop(columns=[2, 3, 4]).reset_index(drop=True)  # drops columns by index
movies.columns = [
    'movie_id', 'title',
    'unknown', 'action', 'adventure', 'animation',
    'childrens', 'comedy', 'crime', 'documentary', 'drama',
    'fantasy', 'noir', 'horror', 'musical', 'mystery',
    'romance', 'scifi', 'thriller', 'war', 'western'
]

X=movies[['unknown','action', 'adventure', 'animation',
    'childrens', 'comedy', 'crime', 'documentary', 'drama',
    'fantasy', 'noir', 'horror', 'musical', 'mystery',
    'romance', 'scifi', 'thriller', 'war', 'western'
]]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
inertias=[]
K=range(1,11)
for k in K:
    km=KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(list(K),inertias,marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

k=6
final_kmeans=KMeans(n_clusters=k,random_state=42,n_init=10)

clusters=final_kmeans.fit_predict(X_scaled)
score=silhouette_score(X_scaled,clusters)
print("Sihoutee Score: ",score)

pca=PCA(n_components=2)

X_pca=pca.fit_transform(X_scaled)
centers_pca=pca.transform(final_kmeans.cluster_centers_)
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

plt.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    marker="X",
    s=200
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.title("KMeans Clusters after PCA")

plt.show()

movies['cluster'] = clusters
movies['pca_x'] = X_pca[:, 0]
movies['pca_y'] = X_pca[:, 1]

def get_cluster_movies(cluster_number):
    filtered = movies[movies['cluster'] == cluster_number]
    return filtered['title'].tolist()


models_path = Path(__file__).resolve().parent.parent / 'models'
models_path.mkdir(exist_ok=True)

joblib.dump(final_kmeans, models_path / 'kmeans_model.pkl')
joblib.dump(score, models_path / 'silhouette_score.pkl')
joblib.dump(movies[['movie_id', 'title', 'cluster', 'pca_x', 'pca_y']], models_path / 'clustered_movies.pkl')

print(f"Silhouette Score: {round(score, 4)}")
print("\nMovies in cluster 0:")
for title in get_cluster_movies(0)[:10]:
    print(f"  {title}")