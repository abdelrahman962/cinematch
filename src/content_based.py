
from pathlib import Path
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
genre_cols=['unknown','action', 'adventure', 'animation',
    'childrens', 'comedy', 'crime', 'documentary', 'drama',
    'fantasy', 'noir', 'horror', 'musical', 'mystery',
    'romance', 'scifi', 'thriller', 'war', 'western'
]


movies["genres_text"]=movies[genre_cols].apply(
    lambda row: " ".join(row[row==1].index.tolist()),
    axis=1
)

vectorizer=CountVectorizer()
x=vectorizer.fit_transform(movies["genres_text"])
similarity=cosine_similarity(x)
def content_recommender(movie_title,n=10):
    mask=movies["title"]==movie_title
    idx=movies[mask].index[0]
    scores=list(enumerate(similarity[idx]))
    scores=sorted(scores,key=lambda x:x[1],reverse=True)
    top=scores[1:n+1]
    movie_indexes=[i[0] for i in top]
    return movies['title'].iloc[movie_indexes].tolist()


models_path=Path(__file__).resolve().parent.parent /'models'

models_path.mkdir(exist_ok=True)

joblib.dump(similarity, models_path /'similarity_matrix.pkl')
joblib.dump(movies[['movie_id','title','genres_text']],models_path/ 'movies.pkl')

results=content_recommender('Toy Story (1995)')
print("Recommendations for Toy Story (1995):")
for i ,title in enumerate(results,1):
    print(f" {i}. {title}")