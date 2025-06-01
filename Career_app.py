import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# === Data Loading & Cleaning ===
df = pd.read_csv('career_path_in_all_field.csv')

# Clean DataFrame
df.columns = df.columns.str.strip()
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(0).astype(float)
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
df = df.dropna(subset=['Career']).reset_index(drop=True)

# Encode Field if present
if 'Field' in df.columns:
    le_field = LabelEncoder()
    df['Field_encoded'] = le_field.fit_transform(df['Field'])
else:
    df['Field_encoded'] = 0

# === Feature Engineering ===
desc_cols = [
    'GPA', 'Extracurricular_Activities', 'Internships', 'Projects', 'Leadership_Positions',
    'Field_Specific_Courses', 'Research_Experience', 'Coding_Skills', 'Communication_Skills',
    'Problem_Solving_Skills', 'Teamwork_Skills', 'Analytical_Skills', 'Presentation_Skills',
    'Networking_Skills', 'Industry_Certifications'
]

def row_to_description(row):
    desc = []
    for col in desc_cols:
        val = str(row[col]).replace('-', ' ').strip()
        if val and val.lower() != 'nan' and val != '0':
            desc.append(f"{col.replace('_', ' ')}: {val}")
    return "; ".join(desc)

df['description'] = df.apply(row_to_description, axis=1)

# === Clustering Careers ===
exclude_cols = ['Career', 'Field']
feature_cols = [col for col in df.columns if col not in exclude_cols + ['Field_encoded']]
numeric_feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
X = df[numeric_feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X_scaled)

# Find best k using silhouette score
k_values = range(2, 11)
sil_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    sil_scores.append(silhouette_score(X_pca, labels))

best_k = k_values[np.argmax(sil_scores)]
final_kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = final_kmeans.fit_predict(X_pca)
df['Cluster'] = cluster_labels

# Prepare cluster career lists
all_cluster_careers = []
for i in range(best_k):
    careers_in_cluster = df[df['Cluster'] == i]['Career'].tolist()
    all_cluster_careers.append(careers_in_cluster)

# === Text Embedding for Clusters ===
cluster_vectorizer = TfidfVectorizer()
all_career_texts = [" ".join(careers) for careers in all_cluster_careers]
cluster_embeddings = cluster_vectorizer.fit_transform(all_career_texts)

# === Career Skill Map & Learning Resources (shortened for brevity) ===
career_skill_map = {
    'Software Engineer': ['programming', 'algorithms', 'data structures', 'problem solving', 'system design'],
    'Data Scientist': ['python', 'statistics', 'machine learning', 'data visualization', 'sql'],
    # ... (add more as needed)
}
learning_resources = {
    'python': ['Python Programming from Coursera', 'Automate the Boring Stuff with Python'],
    'statistics': ['Khan Academy Statistics', 'Statistics for Data Science by Udemy'],
    # ... (add more as needed)
}

# === Streamlit App ===
st.title("🚀 Career Guidance Engine")
st.write("Enter your skills and experience (comma-separated):")

user_text = st.text_input("Your skills (e.g., programming, problem solving, sql):")
if st.button("🔍 Recommend Careers"):
    if user_text.strip():
        user_skills = [s.strip().lower() for s in user_text.split(',') if s.strip()]
        user_embedding = cluster_vectorizer.transform([user_text])
        similarities = cosine_similarity(user_embedding, cluster_embeddings)[0]
        best_cluster = np.argmax(similarities)
        st.success(f"🎯 Most compatible cluster: Cluster {best_cluster+1} (Similarity: {similarities[best_cluster]:.2f})")
        st.write("### 👩‍💼 Recommended Careers for You:")
        for career in all_cluster_careers[best_cluster]:
            st.write(f"- {career}")

        # Show missing skills and resources
        cluster_skill_set = set()
        for career in all_cluster_careers[best_cluster]:
            cluster_skill_set.update(career_skill_map.get(career, []))
        missing_skills = cluster_skill_set - set(user_skills)
        if missing_skills:
            st.warning("⚠️ Skills you might want to develop for this cluster:")
            for skill in sorted(missing_skills):
                st.write(f"- {skill.capitalize()}")
            st.info("📚 Suggested learning resources:")
            for skill in sorted(missing_skills):
                resources = learning_resources.get(skill, [f"Search online courses for {skill.capitalize()}"])
                st.write(f"**{skill.capitalize()}**:")
                for res in resources:
                    st.write(f"- {res}")
        else:
            st.balloons()
            st.success("✅ You already have most of the key skills for this cluster!")

# (Optional) Plot Silhouette Scores
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(k_values), y=sil_scores, marker='o', ax=ax)
ax.set_title("Silhouette Score vs Number of Clusters")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Silhouette Score")
ax.grid(True)
st.pyplot(fig)
