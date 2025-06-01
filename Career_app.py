import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

# === Data Loading & Cleaning ===
df = pd.read_csv('career_path_in_all_field.csv')

# Define skill/description columns
desc_cols = [
    'GPA', 'Extracurricular_Activities', 'Internships', 'Projects', 'Leadership_Positions',
    'Field_Specific_Courses', 'Research_Experience', 'Coding_Skills', 'Communication_Skills',
    'Problem_Solving_Skills', 'Teamwork_Skills', 'Analytical_Skills', 'Presentation_Skills',
    'Networking_Skills', 'Industry_Certifications'
]

# Clean column names and string columns
df.columns = df.columns.str.strip()
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

# Fill missing numerical values with 0
num_cols = list(df.select_dtypes(include=[np.number]).columns)
df[num_cols] = df[num_cols].fillna(0)
df[num_cols] = df[num_cols].astype(float)

# Drop rows with missing Career
df = df.dropna(subset=['Career']).reset_index(drop=True)

# Combine all relevant skill columns into a single string for each career
def row_to_description(row):
    desc = []
    for col in desc_cols:
        val = str(row.get(col, '')).replace('-', ' ').strip()
        if val and val.lower() != 'nan' and val != '0':
            desc.append(f"{col.replace('_', ' ')}: {val}")
    return "; ".join(desc)

df['description'] = df.apply(row_to_description, axis=1)

# Use TfidfVectorizer for text embedding
vectorizer = TfidfVectorizer()
career_embeddings = vectorizer.fit_transform(df['description'].tolist())

def recommend_career(user_skills_text):
    user_embedding = vectorizer.transform([user_skills_text])
    similarities = cosine_similarity(user_embedding, career_embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    recommendations = df.iloc[top_indices]
    return recommendations, similarities[top_indices]

# Helper function to render a progression bar using circles
def render_progress_bar(value, max_value=5):
    try:
        value = int(float(value))
    except:
        value = 0
    value = max(0, min(value, max_value))
    filled = '●' * value
    empty = '○' * (max_value - value)
    return f"{filled}{empty}"

# === Streamlit App ===
st.title("Career Recommendation Engine")

user_input = st.text_input("Enter your skills and experience (comma-separated):", "")

if st.button("Recommend Careers"):
    if user_input.strip():
        recs, scores = recommend_career(user_input)
        st.write("### Recommended Careers for You:")
        for idx, row in recs.iterrows():
            st.write(f"**{row['Career']}** (Similarity: {scores[list(recs.index).index(idx)]:.2f})")
            for col in desc_cols:
                if col in num_cols:
                    st.write(f"- {col.replace('_', ' ')}: {render_progress_bar(row.get(col, 0))}")
                else:
                    val = str(row.get(col, '')).replace('-', ' ').strip()
                    if val and val.lower() != 'nan' and val != '0':
                        st.write(f"- {col.replace('_', ' ')}: {val}")
