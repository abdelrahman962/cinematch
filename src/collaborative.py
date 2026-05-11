from surprise import Dataset,SVD,Reader,accuracy
from surprise.model_selection import cross_validate,train_test_split

from pathlib import Path
import pandas as pd
from pathlib import Path
import joblib

# build path relative to the project and read file (avoids backslash escape issues)
data_path = Path(__file__).resolve().parent.parent / 'data' / 'u.data'
df = pd.read_csv(data_path, header=None, sep='\t', encoding='latin-1')
df=df.drop(columns=[3]).reset_index(drop=True)

df.columns=['user_id','movie_id','rating']
models_path = Path(__file__).resolve().parent.parent / 'models'
movies = joblib.load(models_path / 'movies.pkl')
reader=Reader(rating_scale=(1,5))
data=Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']],reader=reader)
trainset,testset=train_test_split(data,test_size=0.2)

svd_model=SVD()
svd_model.fit(trainset)
predictions=svd_model.test(testset)


def collab_recommender(user_id):
    result=df[df['user_id']==user_id]
    rated_movies=set(result['movie_id'])
    all_movies=set(df['movie_id'].unique())
    unseen_movies=all_movies-rated_movies
    all_predictions=[]
    for mov_id in unseen_movies:
        estimated_predictions=svd_model.predict(user_id,mov_id)
        all_predictions.append((mov_id,estimated_predictions.est))
    final_pred=sorted(all_predictions,key= lambda x:x[1],reverse=True)
    top=final_pred[:10]
    last_rs=[]
    for movie_id, rating in top:
        title=movies[movies['movie_id']==movie_id]['title'].values[0]
        last_rs.append((title,round(rating,2)))
    return last_rs


joblib.dump(svd_model, models_path / 'collab_model.pkl')

rmse = accuracy.rmse(predictions)
joblib.dump(rmse, models_path / 'rmse_score.pkl')
results = collab_recommender(1)
print("Recommendations for user 1:")
for title, rating in results:
    print(f"  {title} — predicted rating: {rating}")