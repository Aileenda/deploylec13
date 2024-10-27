
import streamlit as st
import pickle
from surprise import SVD

# โหลดข้อมูลจากไฟล์
with open('66130701921recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# UI สำหรับรับ user ID
st.title('Movie Recommendations System')
user_id = st.number_input('Enter User ID:', min_value=1, step=1)

# ถ้า User ID มีการระบุ
if user_id:
    # หาหนังที่ผู้ใช้เคยให้คะแนนแล้ว
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    # หาหนังที่ผู้ใช้ยังไม่ได้ให้คะแนน
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    
    # คำนวณการให้คะแนนโดย SVD model
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    
    # จัดลำดับการทำนายคะแนนจากสูงไปต่ำ
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)

    # เลือก 10 หนังที่แนะนำ
    top_recommendations = sorted_predictions[:10]

    # แสดงผลหนังแนะนำ
    st.subheader(f'Top 10 movie recommendations for User {user_id}')
    for recommendation in top_recommendations:
        movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
        st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")


