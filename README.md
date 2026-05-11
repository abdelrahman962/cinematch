# CineMatch — Movie Recommendation Engine

A machine learning web application that recommends movies using three different
recommendation techniques learned in the Machine Learning Specialization
(DeepLearning.AI & Stanford University).

---

## Live Demo

Run locally using the instructions below.

---

## What the app does

CineMatch analyzes the MovieLens 100K dataset and provides personalized movie
recommendations through three separate engines:

- Engine 1 finds movies with similar genres to one you already like
- Engine 2 predicts movies you will enjoy based on ratings from similar users
- Engine 3 groups all movies into clusters and lets you explore by taste group

---

## The three engines

### Engine 1 — Content-Based Filtering
Converts each movie's genres into a text vector using CountVectorizer then
measures similarity between every pair of movies using cosine similarity.
Returns the 10 most similar movies to any selected title.

### Engine 2 — Collaborative Filtering
Uses SVD matrix factorization from the Surprise library to learn hidden
patterns between users and movies from 100,000 ratings. Predicts what rating
a user would give to movies they have not seen yet and returns the top 10.

### Engine 3 — Cluster Explorer
Groups all 1682 movies into 6 clusters using KMeans based on genre similarity.
Uses PCA to reduce 19 genre dimensions to 2 for interactive visualization.

---

## ML techniques used

| Technique | Library | Purpose |
|-----------|---------|---------|
| CountVectorizer | scikit-learn | Convert genres to vectors |
| Cosine Similarity | scikit-learn | Measure movie similarity |
| SVD | scikit-surprise | Collaborative filtering |
| KMeans | scikit-learn | Movie clustering |
| PCA | scikit-learn | Dimensionality reduction |
| StandardScaler | scikit-learn | Feature scaling |

---

## Dataset

MovieLens 100K — collected by GroupLens Research at University of Minnesota.

- 100,000 ratings from 943 users on 1,682 movies
- Each user rated at least 20 movies
- Ratings scale: 1 to 5

Download: https://grouplens.org/datasets/movielens/100k/

---

## Project structure

```
cinematch/
├── data/
│   ├── u.data                  # 100,000 ratings
│   ├── u.item                  # movie info and genres
│   └── u.user                  # user demographic info
│
├── src/
│   ├── content_based.py        # Engine 1 — similarity matrix training
│   ├── collaborative.py        # Engine 2 — SVD model training
│   └── clustering.py           # Engine 3 — KMeans model training
│
├── models/
│   ├── similarity_matrix.pkl   # Content-based model
│   ├── movies.pkl              # Movie data cache
│   ├── collab_model.pkl        # Collaborative filtering model
│   ├── rmse_score.pkl          # Model evaluation score
│   ├── kmeans_model.pkl        # Clustering model
│   ├── silhouette_score.pkl    # Clustering evaluation
│   └── clustered_movies.pkl    # Movie cluster assignments
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md
```


---

## How to run locally

### 1 — Clone the repository

git clone

https://github.com/abdelrahman962/cinematch.git cd cinematch

### 2 — Install dependencies

pip install -r requirements.txt

### 3 — Download the dataset

Download MovieLens 100K from https://grouplens.org/datasets/movielens/100k/
and place these 3 files inside the data/ folder:

u.data
u.item
u.user

### 4 — Train the models

Run each src file once to generate the saved models:

python src/content_based.py
python src/collaborative.py
python src/clustering.py

### 5 — Launch the app

streamlit run app.py

Open your browser at http://localhost:8501

---

## Model performance

| Metric | Value | Meaning |
|--------|-------|---------|
| Collaborative Filter RMSE | 0.9394 | Average prediction error on 1-5 scale |
| Clustering Silhouette Score | 0.2512 | Cluster separation quality |

RMSE below 1.0 on a 1-5 rating scale is considered good performance.
Silhouette score above 0.2 is acceptable for genre-based movie data where
genres naturally overlap.

---

## What I would improve with more time

- Add hybrid recommendations combining content and collaborative engines
- Use movie descriptions and cast data instead of genres only for richer similarity
- Add user registration so new users can rate movies and get recommendations
- Deploy to Streamlit Cloud so no local setup is needed
- Add more evaluation metrics like Precision@K and Recall@K

---


## Author

Abdalrahman
GitHub: https://github.com/abdelrahman962