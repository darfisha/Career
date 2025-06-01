import streamlit as st
import pandas as pd

# Example: Load your dataset
@st.cache_data
def load_data():
    # Replace with your actual dataset path
    df = pd.read_csv('career_path_in_all_field.csv')
    return df

df = load_data()

# Columns for description (including Skills)
desc_cols = [
    'GPA', 'Extracurricular_Activities', 'Internships', 'Projects', 'Leadership_Positions',
    'Field_Specific_Courses', 'Research_Experience', 'Coding_Skills', 'Communication_Skills',
    'Problem_Solving_Skills', 'Teamwork_Skills', 'Analytical_Skills', 'Presentation_Skills',
    'Networking_Skills', 'Industry_Certifications', 'Skills'
]

# Numeric columns for progress bars
num_cols = [
    'GPA', 'Extracurricular_Activities', 'Internships', 'Projects', 'Leadership_Positions',
    'Field_Specific_Courses', 'Research_Experience', 'Coding_Skills', 'Communication_Skills',
    'Problem_Solving_Skills', 'Teamwork_Skills', 'Analytical_Skills', 'Presentation_Skills',
    'Networking_Skills'
]

def render_progress_bar(value, max_value=10):
    pct = min(max(float(value) / max_value, 0), 1)
    bar = f'<div style="background:#eee;width:100%;border-radius:5px;"><div style="background:#4CAF50;width:{pct*100}%;height:16px;border-radius:5px;"></div></div>'
    return bar

def recommend_career(user_input, top_n=3):
    # Dummy recommendation logic for demonstration
    # Replace with your actual recommendation logic
    # Example: filter by user_input if not empty
    if user_input.strip():
        filtered = df[df.apply(lambda row: user_input.lower() in str(row).lower(), axis=1)]
        if not filtered.empty:
            return filtered.head(top_n)
    return df.head(top_n)

# Streamlit UI
st.title("Career Guidance Engine")

user_input = st.text_input("Describe your interests, skills, or career goals:")
top_n = st.slider("Number of recommendations", 1, 5, 3)
submitted = st.button("Recommend Careers")

if submitted and user_input.strip():
    recs = recommend_career(user_input, top_n=top_n)
    st.success("Here are your recommended careers! 🌟")
    for idx, row in recs.iterrows():
        st.markdown(f'<div class="animated-career"><h3>{row["Career"]}</h3></div>', unsafe_allow_html=True)
        with st.expander("See required skills and experiences"):
            # Show the three skills
            skills = str(row.get('Skills', '')).split(',')
            skills = [s.strip() for s in skills if s.strip()]
            if skills:
                st.write("**Key Skills:**")
                for skill in skills:
                    st.write(f"- {skill}")
            # Show other columns as before
            for col in desc_cols:
                if col == 'Skills':
                    continue  # Already shown above
                if col in num_cols:
                    st.markdown(f"- {col.replace('_', ' ')}: {render_progress_bar(row.get(col, 0))}", unsafe_allow_html=True)
                else:
                    val = str(row.get(col, '')).replace('-', ' ').strip()
                    if val and val.lower() != 'nan' and val != '0':
                        st.write(f"- {col.replace('_', ' ')}: {val}")
