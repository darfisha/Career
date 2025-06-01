import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import joblib

# Load the pre-trained model and preprocessors
try:
    clf = joblib.load('random_forest_model.pkl')
    mlb = joblib.load('multilabel_binarizer.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    df = pd.read_csv('career_path_in_all_field.csv')
except FileNotFoundError:
    st.error("Model or preprocessors not found. Please ensure 'random_forest_model.pkl', 'multilabel_binarizer.pkl', 'scaler.pkl', 'label_encoder.pkl', and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()

# --- Data Preprocessing Functions ---
def clean_skills(skills):
    if pd.isnull(skills):
        return []
    return [re.sub(r'[^a-zA-Z0-9 ]', '', s.strip().lower()) for s in skills.split(',') if s.strip()]

df['skills'] = df['skills'].apply(clean_skills)

# --- Prediction/Recommendation Function ---
def get_career_recommendations(user_interest_text, user_experience_years, df, clf, mlb, scaler, tfidf):
    user_interest_vec = tfidf.transform([user_interest_text]).toarray()[0]
    user_interests = [mlb.classes_[i] for i in np.where(user_interest_vec > 0)[0]]

    user_skills_vec = mlb.transform([user_interests])
    user_experience = scaler.transform([[user_experience_years]])
    user_vector = np.hstack([user_skills_vec, user_experience])

    # aptitude_prediction = clf.predict(user_vector) # Uncomment if you want to use this

    skills_encoded = mlb.transform(df['skills'])
    experience_encoded = scaler.transform(df[['experience']])
    career_vectors = np.hstack([skills_encoded, experience_encoded])

    similarities = cosine_similarity(user_vector, career_vectors)
    top_indices = similarities.argsort()[0][::-1][:3]

    recommendations = []
    for idx in top_indices:
        recommendations.append({
            'career': df.iloc[idx]['career'],
            'similarity': similarities[0, idx]
        })

    return recommendations, user_interests # , aptitude_prediction # uncomment if aptitude is returned

def get_skill_gap_analysis(top_career, user_skills, df):
    career_skills = set(df[df['career'] == top_career]['skills'].iloc[0])
    user_skills_set = set(user_skills)
    missing_skills = career_skills - user_skills_set
    return missing_skills

# --- Streamlit App ---
st.title("AI-Powered Career Guidance Engine")
st.header("Enter Your Information")

user_interest_text = st.text_area("Describe your interests, skills, and aspirations:")
user_experience_years = st.number_input("Enter your years of experience:", min_value=0.0, step=0.5)

if st.button("Get Career Recommendations"):
    if not user_interest_text:
        st.warning("Please describe your interests.")
    else:
        recommendations, user_interests = get_career_recommendations(user_interest_text, user_experience_years, df, clf, mlb, scaler, tfidf) # Adjust return values if aptitude is used

        st.header("Top Career Recommendations:")
        for rec in recommendations:
            st.write(f"- **{rec['career']}** (Similarity: {rec['similarity']:.2f})")

        if recommendations:
            top_career_name = recommendations[0]['career']
            missing_skills = get_skill_gap_analysis(top_career_name, user_interests, df)

            st.header(f"Skill Gap Analysis for '{top_career_name}':")
            if missing_skills:
                st.write("Missing Skills:")
                for skill in missing_skills:
                    st.write(f"- {skill.title()}")
                st.subheader("Recommended Learning Resources:")
                for skill in missing_skills:
                    st.write(f"- Learn {skill.title()} on Coursera/Udemy/edX")
            else:
                st.write("None! You match all required skills for this career.")
