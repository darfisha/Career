import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Data Loading & Cleaning ===
df = pd.read_csv('career_path_in_all_field.csv')

desc_cols = [
    'GPA', 'Extracurricular_Activities', 'Internships', 'Projects', 'Leadership_Positions',
    'Field_Specific_Courses', 'Research_Experience', 'Coding_Skills', 'Communication_Skills',
    'Problem_Solving_Skills', 'Teamwork_Skills', 'Analytical_Skills', 'Presentation_Skills',
    'Networking_Skills', 'Industry_Certifications'
]

df.columns = df.columns.str.strip()
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
num_cols = list(df.select_dtypes(include=[np.number]).columns)
df[num_cols] = df[num_cols].fillna(0).astype(float)
df = df.dropna(subset=['Career']).reset_index(drop=True)

def row_to_description(row):
    desc = []
    for col in desc_cols:
        val = str(row.get(col, '')).replace('-', ' ').strip()
        if val and val.lower() != 'nan' and val != '0':
            desc.append(f"{col.replace('_', ' ')}: {val}")
    return "; ".join(desc)

df['description'] = df.apply(row_to_description, axis=1)
vectorizer = TfidfVectorizer()
career_embeddings = vectorizer.fit_transform(df['description'].tolist())

def recommend_career(user_skills_text, top_n=3):
    user_embedding = vectorizer.transform([user_skills_text])
    similarities = cosine_similarity(user_embedding, career_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices]
    return recommendations, similarities[top_indices]

def render_progress_bar(value, max_value=5):
    try:
        value = int(float(value))
    except:
        value = 0
    value = max(0, min(value, max_value))
    filled = '🟢' * value
    empty = '⚪' * (max_value - value)
    return f"{filled}{empty}"

# === Streamlit App ===
st.set_page_config(page_title="Career Recommendation Engine", page_icon="🎯")
st.title("🎯 Career Recommendation Engine")
st.write("Welcome! Get personalized career suggestions based on your skills and experiences. 🚀")

with st.form("career_form"):
    user_input = st.text_area("📝 Enter your skills and experience (comma-separated):", "")
    top_n = st.slider("How many career recommendations would you like?", 1, 5, 3)
    submitted = st.form_submit_button("🔍 Recommend Careers")

if submitted and user_input.strip():
    recs, scores = recommend_career(user_input, top_n=top_n)
    st.success("Here are your recommended careers! 🌟")
    for idx, row in recs.iterrows():
        st.markdown(f"### {row['Career']}  \n_Similarity: {scores[list(recs.index).index(idx)]:.2f}_")
        with st.expander("See required skills and experiences"):
            for col in desc_cols:
                if col in num_cols:
                    st.write(f"- {col.replace('_', ' ')}: {render_progress_bar(row.get(col, 0))}")
                else:
                    val = str(row.get(col, '')).replace('-', ' ').strip()
                    if val and val.lower() != 'nan' and val != '0':
                        st.write(f"- {col.replace('_', ' ')}: {val}")
else:
    st.info("👈 Enter your skills and click 'Recommend Careers' to get started!")
