# ── imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from pathlib import Path

# ── load models ───────────────────────────────────────────────────────────────
models_path = Path(__file__).resolve().parent / 'models'

similarity       = joblib.load(models_path / 'similarity_matrix.pkl')
movies           = joblib.load(models_path / 'movies.pkl')
rmse             = joblib.load(models_path / 'rmse_score.pkl')
kmeans           = joblib.load(models_path / 'kmeans_model.pkl')
sil_score        = joblib.load(models_path / 'silhouette_score.pkl')
clustered_movies = joblib.load(models_path / 'clustered_movies.pkl')

# ── load ratings data ─────────────────────────────────────────────────────────
data_path = Path(__file__).resolve().parent / 'data' / 'u.data'
ratings = pd.read_csv(data_path, header=None, sep='\t')
ratings = ratings.drop(columns=[3]).reset_index(drop=True)
ratings.columns = ['user_id', 'movie_id', 'rating']


@st.cache_resource
def load_collab_model():
    try:
        return joblib.load(models_path / 'collab_model.pkl')
    except ModuleNotFoundError as exc:
        if exc.name == 'surprise':
            return None
        raise

# ── functions ─────────────────────────────────────────────────────────────────

def content_recommender(movie_title, n=10):
    # find row index of selected movie
    idx = movies[movies['title'] == movie_title].index[0]
    # get similarity scores for this movie vs all others
    scores = list(enumerate(similarity[idx]))
    # sort highest first
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # skip the movie itself, take top n
    top = scores[1:n+1]
    # get movie indexes
    movie_indexes = [i[0] for i in top]
    # return titles
    return movies['title'].iloc[movie_indexes].tolist()

def collab_recommender(user_id, collab_model, n=10):
    # find movies this user already rated
    rated   = set(ratings[ratings['user_id'] == user_id]['movie_id'])
    # find all movie ids
    all_mov = set(ratings['movie_id'].unique())
    # find unseen movies
    unseen  = all_mov - rated
    # predict rating for each unseen movie
    preds = []
    for movie_id in unseen:
        pred = collab_model.predict(user_id, movie_id)
        preds.append((movie_id, pred.est))
    # sort by predicted rating highest first
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    # take top n
    top = preds[:n]
    # look up titles and return with ratings
    results = []
    for movie_id, rating in top:
        title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        results.append((title, round(rating, 2)))
    return results

# ── app config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineMatch", layout="wide")
st.title("CineMatch — Movie Recommendation Engine")

# sidebar navigation
page = st.sidebar.selectbox(
    "Navigate",
    [
        "Content Recommender",
        "Collaborative Recommender",
        "Cluster Explorer",
        "Model Evaluation"
    ]
)

# ── page 1 — content recommender ─────────────────────────────────────────────
if page == "Content Recommender":

    st.header("Find Similar Movies")
    st.write("Select a movie you like and we will find 10 similar ones based on genre.")

    # dropdown of all movie titles
    selected_movie = st.selectbox("Select a movie", movies['title'].tolist())

    if st.button("Get Recommendations"):
        results = content_recommender(selected_movie)
        st.success(f"Top 10 movies similar to '{selected_movie}':")
        for i, title in enumerate(results, 1):
            st.write(f"{i}. {title}")

# ── page 2 — collaborative recommender ───────────────────────────────────────
elif page == "Collaborative Recommender":

    st.header("Personalized Recommendations")
    st.write("Enter your user ID and we will predict movies you will enjoy.")

    collab_model = load_collab_model()
    if collab_model is None:
        st.warning(
            "The collaborative recommender is unavailable because the 'surprise' package is not installed in this Python environment."
        )

    user_id = st.number_input(
        "Enter user ID (1 to 943)",
        min_value=1,
        max_value=943,
        value=1,
        step=1
    )

    if st.button("Get Recommendations", disabled=collab_model is None):
        with st.spinner("Predicting your recommendations..."):
            results = collab_recommender(user_id, collab_model)
        st.success(f"Top 10 recommendations for user {user_id}:")
        df_results = pd.DataFrame(results, columns=['Movie Title', 'Predicted Rating'])
        df_results.index = df_results.index + 1
        st.dataframe(df_results)

# ── page 3 — cluster explorer ─────────────────────────────────────────────────
elif page == "Cluster Explorer":

    st.header("Explore Movie Clusters")
    st.write("Movies are grouped into 6 clusters based on genre similarity.")

    # interactive plotly scatter plot
    fig = px.scatter(
        clustered_movies,
        x='pca_x',
        y='pca_y',
        color='cluster',
        hover_data=['title'],
        title="Movie Clusters (PCA Visualization)",
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, width='stretch')

    # cluster selector
    st.subheader("Movies in each cluster")
    cluster_num = st.selectbox("Select a cluster", [0, 1, 2, 3, 4, 5])
    cluster_movies = clustered_movies[
        clustered_movies['cluster'] == cluster_num
    ]['title'].tolist()
    st.write(f"Cluster {cluster_num} contains {len(cluster_movies)} movies:")
    st.dataframe(pd.DataFrame(cluster_movies, columns=['Movie Title']))

# ── page 4 — model evaluation ─────────────────────────────────────────────────
elif page == "Model Evaluation":

    st.header("Model Evaluation")
    st.write("Performance metrics for each engine.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Collaborative Filter RMSE",
            value=round(rmse, 4)
        )
        st.write(
            "Average prediction error on a 1 to 5 rating scale. "
            "Lower is better. Below 1.0 is considered good."
        )

    with col2:
        st.metric(
            label="Clustering Silhouette Score",
            value=round(sil_score, 4)
        )
        st.write(
            "Measures how well separated the clusters are. "
            "Range is -1 to 1. Higher is better. "
            "Above 0.2 is acceptable for genre-based data."
        )