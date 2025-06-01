import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import streamlit as st

# Placeholder: Load Indian career profiles, exams, and academic streams
# Create a larger fake Indian career profiles dataset for testing
fake_data = {
    'career': [
        'Civil Servant', 'Software Engineer', 'Doctor', 'Teacher', 'Lawyer',
        'Chartered Accountant', 'Mechanical Engineer', 'Journalist', 'Pharmacist', 'Architect',
        'Data Scientist', 'Nurse', 'Graphic Designer', 'Banker', 'Entrepreneur',
        'Psychologist', 'Pilot', 'Fashion Designer', 'Dentist', 'Statistician'
    ],
    'required_skills': [
        'leadership, general knowledge, law, public administration',
        'programming, problem solving, algorithms, teamwork',
        'biology, empathy, medical knowledge, communication',
        'communication, teaching, subject knowledge, patience',
        'law, critical thinking, argumentation, research',
        'accounting, finance, auditing, analytical thinking',
        'mechanical design, physics, problem solving, teamwork',
        'writing, communication, investigation, critical thinking',
        'pharmacy, chemistry, attention to detail, communication',
        'design, creativity, mathematics, visualization',
        'statistics, programming, machine learning, data analysis',
        'nursing, empathy, medical knowledge, patience',
        'creativity, design, software, communication',
        'finance, analytical thinking, customer service, mathematics',
        'innovation, risk taking, management, leadership',
        'psychology, empathy, research, communication',
        'navigation, technical skills, decision making, communication',
        'fashion, creativity, design, trend analysis',
        'dentistry, medical knowledge, precision, communication',
        'statistics, mathematics, analytical thinking, data analysis'
    ],
    'stream': [
        'Arts', 'Science', 'Science', 'Arts', 'Commerce',
        'Commerce', 'Science', 'Arts', 'Science', 'Science',
        'Science', 'Science', 'Arts', 'Commerce', 'Any',
        'Arts', 'Science', 'Arts', 'Science', 'Science'
    ],
    'exams': [
        'UPSC', 'JEE', 'NEET', 'CTET', 'CLAT',
        'CA Foundation', 'GATE', 'IIMC Entrance', 'GPAT', 'NATA',
        'GATE, CAT', 'AIIMS Nursing', 'NID', 'IBPS', 'None',
        'CUET', 'CPL', 'NIFT', 'NEET', 'ISI Admission Test'
    ]
}
career_profiles = pd.DataFrame(fake_data)

# --- Aptitude Estimation Model (Supervised ML) ---
class AptitudeEstimator:
    def __init__(self, model_path='aptitude_model.pkl'):
        if model_path is not None:
            self.model = joblib.load(model_path)
        else:
            self.model = None
    
    def predict(self, test_data):
        return self.model.predict_proba([test_data])[0]

# --- NLP-driven Goal & Interest Extraction ---
class GoalInterestExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.career_clusters = career_profiles['career'].tolist()
        self.vectorizer.fit(self.career_clusters)
    
    def extract(self, user_text):
        user_vec = self.vectorizer.transform([user_text])
        cluster_vecs = self.vectorizer.transform(self.career_clusters)
        sims = cosine_similarity(user_vec, cluster_vecs).flatten()
        top_idx = np.argmax(sims)
        return self.career_clusters[top_idx]

# --- Skill & Experience Mapping using Embeddings ---
class SkillMapper:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model  # e.g., SentenceTransformer
    
    def embed_skills(self, skills_list):
        return self.embedding_model.encode(skills_list)
    
    def match_careers(self, user_skills):
        user_vec = self.embed_skills([user_skills])
        career_vecs = self.embed_skills(career_profiles['required_skills'].tolist())
        sims = cosine_similarity(user_vec, career_vecs).flatten()
        top_indices = sims.argsort()[-5:][::-1]
        return career_profiles.iloc[top_indices][['career', 'stream', 'exams']]

# --- AI-Driven Career Recommendation Engine ---
class CareerRecommender:
    def __init__(self, aptitude_model, goal_extractor, skill_mapper):
        self.aptitude_model = aptitude_model
        self.goal_extractor = goal_extractor
        self.skill_mapper = skill_mapper
    
    def recommend(self, user_data):
        aptitude_scores = self.aptitude_model.predict(user_data['aptitude_test'])
        goal = self.goal_extractor.extract(user_data['aspirations'])
        skill_matches = self.skill_mapper.match_careers(user_data['skills'])
        # Combine scores (simple example)
        recommendations = skill_matches.copy()
        recommendations['aptitude_score'] = aptitude_scores[:len(recommendations)]
        recommendations['goal_match'] = recommendations['career'] == goal
        return recommendations.sort_values(['goal_match', 'aptitude_score'], ascending=False)

# --- Skill Gap Analysis ---
def skill_gap_analysis(user_skills, target_career_skills):
    user_set = set(user_skills.lower().split(','))
    target_set = set(target_career_skills.lower().split(','))
    missing = target_set - user_set
    return list(missing)

# --- Example Usage ---
# Load or initialize models
# aptitude_model = AptitudeEstimator('aptitude_model.pkl')
# goal_extractor = GoalInterestExtractor()
# skill_mapper = SkillMapper(embedding_model)  # e.g., SentenceTransformer('all-MiniLM-L6-v2')
# recommender = CareerRecommender(aptitude_model, goal_extractor, skill_mapper)

# user_data = {
#     'aptitude_test': [0.7, 0.5, 0.8],  # Example features
#     'aspirations': "I want to become a civil servant and help society.",
#     'skills': "communication, leadership, general knowledge"
# }
# recommendations = recommender.recommend(user_data)
# print(recommendations)

# For skill gap analysis:
# missing_skills = skill_gap_analysis(user_data['skills'], "leadership, general knowledge, law, public administration")
# print("Skill gaps:", missing_skills)
# --- Generate Fake Data for Testing ---

# Create a fake Indian career profiles dataset
fake_data = {
    'career': [
        'Civil Servant', 'Software Engineer', 'Doctor', 'Teacher', 'Lawyer'
    ],
    'required_skills': [
        'leadership, general knowledge, law, public administration',
        'programming, problem solving, algorithms, teamwork',
        'biology, empathy, medical knowledge, communication',
        'communication, teaching, subject knowledge, patience',
        'law, critical thinking, argumentation, research'
    ],
    'stream': [
        'Arts', 'Science', 'Science', 'Arts', 'Commerce'
    ],
    'exams': [
        'UPSC', 'JEE', 'NEET', 'CTET', 'CLAT'
    ]
}
career_profiles = pd.DataFrame(fake_data)
career_profiles.to_excel('career_data.xlsx', index=False)

# --- Dummy Models for Testing ---
class DummyAptitudeModel:
    def predict_proba(self, X):
        # Return random probabilities for 5 careers
        return [[0.8, 0.6, 0.4, 0.7, 0.5]]

class DummyEmbeddingModel:
    def encode(self, texts):
        # Return random vectors for each text
        return np.random.rand(len(texts), 5)

# Instantiate dummy models
aptitude_model = AptitudeEstimator(model_path=None)
aptitude_model.model = DummyAptitudeModel()
goal_extractor = GoalInterestExtractor()
skill_mapper = SkillMapper(DummyEmbeddingModel())
recommender = CareerRecommender(aptitude_model, goal_extractor, skill_mapper)

# --- Streamlit App ---
st.set_page_config(page_title="Career Guidance Engine (India)", page_icon="🎓", layout="centered")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/career.png", width=80)
    st.title("Career Guidance Engine 🇮🇳")
    st.markdown("""
    **Empowering your career journey!**
    
    - Personalized recommendations
    - Skill gap analysis
    - NEP-aligned learning pathways
    
    ---
    Made with ❤️ using Streamlit
    """)

st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>🚀 Career Guidance Engine (India)</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Find your best-fit career, bridge skill gaps, and plan your learning journey!</h4>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    aspirations = st.text_area("🎯 What are your aspirations?", 
        "I want to become a civil servant and help society.",
        help="Describe your dream career or what motivates you.")
with col2:
    skills = st.text_area("🛠️ List your skills (comma separated)", 
        "communication, leadership, general knowledge",
        help="E.g., programming, teamwork, creativity, etc.")

aptitude_test = st.text_input("📊 Aptitude Test Scores (comma separated)", 
    "0.7, 0.5, 0.8", 
    help="Enter your scores or estimates for aptitude (e.g., 0.7, 0.5, 0.8)")

st.markdown("---")

if st.button("✨ Get My Career Recommendations"):
    with st.spinner("Analyzing your profile and finding the best matches..."):
        try:
            aptitude_scores = [float(x.strip()) for x in aptitude_test.split(",")]
            user_data = {
                'aptitude_test': aptitude_scores,
                'aspirations': aspirations,
                'skills': skills
            }
            recommendations = recommender.recommend(user_data)
            st.success("Here are your top career recommendations! 🎉")
            st.dataframe(recommendations.style.highlight_max(axis=0, color='#D6EAF8'))

            top_career = recommendations.iloc[0]['career']
            st.balloons()
            st.markdown(f"<h3 style='color:#27AE60;'>🏆 Top Career Recommendation: {top_career}</h3>", unsafe_allow_html=True)

            # Skill gap for top career
            top_skills = career_profiles[career_profiles['career'] == top_career]['required_skills'].values[0]
            missing_skills = skill_gap_analysis(skills, top_skills)
            st.info(f"🧩 **Missing Skills for {top_career}:** {', '.join(missing_skills) if missing_skills else 'None'}")

            # Show relevant stream and exams
            stream = career_profiles[career_profiles['career'] == top_career]['stream'].values[0]
            exams = career_profiles[career_profiles['career'] == top_career]['exams'].values[0]
            st.write(f"📚 **Relevant Academic Stream:** `{stream}`")
            st.write(f"📝 **Relevant Competitive Exam(s):** `{exams}`")

            # Suggest learning pathways (simple example)
            if missing_skills:
                with st.expander("🎓 Recommended Learning Pathways"):
                    for skill in missing_skills:
                        st.write(f"- Consider NEP-aligned courses or online resources to build: **{skill.strip().capitalize()}**")
                    st.write(f"📖 You may also prepare for competitive exams like: `{exams}`")
        except Exception as e:
            st.error(f"❌ Error: {e}")

with st.expander("🔍 Sample Data Preview"):
    st.dataframe(career_profiles)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Made with ❤️ by your AI Career Guide</p>", unsafe_allow_html=True)

# To run this app with Streamlit:
# 1. Open a terminal or command prompt.
# 2. Navigate to the folder containing this file.
# 3. Run the following command:
#    streamlit run "import numpy as np.py"
# (Ensure you have Streamlit installed: pip install streamlit)
