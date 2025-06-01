import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_career(user_skills_text, df):
    # Combine all career skills into a list
    career_skills_list = df['Skills'].astype(str).tolist()
    # Add user input as the last document
    documents = career_skills_list + [user_skills_text]
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Compute cosine similarity between user input and all careers
    user_vec = tfidf_matrix[-1]
    career_vecs = tfidf_matrix[:-1]
    similarities = cosine_similarity(user_vec, career_vecs).flatten()
    
    # Add similarity scores to DataFrame
    df['similarity_score'] = similarities
    
    # Get top 10 recommendations
    recommendations = df.sort_values('similarity_score', ascending=False).head(10)
    top_scores = recommendations['similarity_score'].values
    return recommendations, top_scores
