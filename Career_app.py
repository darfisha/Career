import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# === Data Loading & Cleaning ===
df = pd.read_csv('career_path_in_all_field.csv')

# Remove torch and SentenceTransformer dependencies.
# Use sklearn's TfidfVectorizer for text embedding and cosine_similarity for similarity.


# No torch or SentenceTransformer needed!
def embed_descriptions(descriptions):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(descriptions)
    return embeddings, vectorizer

def embed_user_input(user_text, vectorizer):
    return vectorizer.transform([user_text])

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
# Combine all relevant skill columns into a single string for each career
desc_cols = [
    'GPA', 'Extracurricular_Activities', 'Internships', 'Projects', 'Leadership_Positions',
    'Field_Specific_Courses', 'Research_Experience', 'Coding_Skills', 'Communication_Skills',
    'Problem_Solving_Skills', 'Teamwork_Skills', 'Analytical_Skills', 'Presentation_Skills',
    'Networking_Skills', 'Industry_Certifications'
]

def row_to_description(row):
    # Replace underscores with spaces in column names for readability
    desc = []
    for col in desc_cols:
        val = str(row[col]).replace('-', ' ').strip()
        if val and val.lower() != 'nan' and val != '0':
            desc.append(f"{col.replace('_', ' ')}: {val}")
    return "; ".join(desc)

df['description'] = df.apply(row_to_description, axis=1)
career_embeddings = sentence_model.encode(df['description'].tolist())

def recommend_career(user_skills_text):
    # Embed user input
    user_embedding = sentence_model.encode([user_skills_text])
    
    # Compute cosine similarity with career descriptions
    similarities = util.cos_sim(user_embedding, career_embeddings)[0].cpu().numpy()
    
    # Get top 3 career matches
    top_indices = similarities.argsort()[-3:][::-1]
    recommendations = df.iloc[top_indices]
    
    return recommendations, similarities[top_indices]

# Streamlit app starts here
st.title("Career Path Recommendation System")

st.write("Enter your skills or interests, and get personalized career path recommendations!")

user_input = st.text_area("Enter your skills (comma separated or descriptive text):")

if st.button("Recommend Careers"):
    if user_input.strip():
        recs, scores = recommend_career(user_input)
        
        st.write("### Recommended Careers for You:")
        for idx, row in recs.iterrows():
            st.write(f"**{row['career']}** (Similarity: {scores[list(recs.index).index(idx)]:.2f})")
            st.write(f"- {row['description']}")

print("Initial info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Fill missing numerical values with 0 (or median/mean as you prefer)
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(0)

# Strip whitespace from column names and string columns
df.columns = df.columns.str.strip()
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

# Convert numerical columns to float explicitly
df[num_cols] = df[num_cols].astype(float)

print("\nCleaned DataFrame head:\n", df.head())

df = df.dropna(subset=['Career']).reset_index(drop=True)

if 'Field' in df.columns:
    le_field = LabelEncoder()
    df['Field_encoded'] = le_field.fit_transform(df['Field'])
else:
    df['Field_encoded'] = 0

exclude_cols = ['Career', 'Field']
feature_cols = [col for col in df.columns if col not in exclude_cols + ['Field_encoded']]
X = df[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

exclude_cols = ['Career', 'Field']
feature_cols = [col for col in df.columns if col not in exclude_cols + ['Field_encoded']]
# Only use numeric columns for scaling
numeric_feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
X = df[numeric_feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Define the range of k values to try for clustering
k_values = range(2, 11)
sil_scores = []

# Compute X_pca using PCA with n_components=15
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X_scaled)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    sil_scores.append(score)
    print(f"Silhouette Score for {k} clusters: {score:.4f}")

plt.figure(figsize=(10, 5))
sns.lineplot(x=list(k_values), y=sil_scores, marker='o')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()


best_k = k_values[np.argmax(sil_scores)]
print(f"\n✅ Best number of clusters (k): {best_k}")

final_kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = final_kmeans.fit_predict(X_pca)
df['Cluster'] = cluster_labels

all_cluster_careers = []
for i in range(best_k):
    cluster_career_counts = df[df['Cluster'] == i]['Career'].value_counts()
    print(f"\nCluster {i} careers:\n{cluster_career_counts.head(10)}")
    all_cluster_careers.append(list(cluster_career_counts.index))

model = SentenceTransformer('all-MiniLM-L6-v2')

cluster_embeddings = []
for careers in all_cluster_careers:
    career_text = " ".join(careers)
    emb = model.encode(career_text, convert_to_tensor=True)
    cluster_embeddings.append(emb)

# === Define career to skill mapping here manually ===
career_skill_map = {
    'Data Scientist': ['python', 'statistics', 'machine learning', 'data visualization', 'sql'],
    'Chartered Accountant': ['accounting', 'taxation', 'finance', 'auditing', 'gst'],
    'Civil Services (UPSC)': ['general knowledge', 'current affairs', 'essay writing', 'history', 'geography', 'political science'],
    'Doctor (NEET)': ['biology', 'chemistry', 'physics', 'anatomy', 'physiology'],
    'Software Engineer': ['programming', 'algorithms', 'data structures', 'problem solving', 'system design'],
    'Mechanical Engineer': ['mechanics', 'thermodynamics', 'material science', 'maths', 'design'],
    'Marketing Manager': ['marketing strategy', 'brand management', 'analytics', 'seo', 'communication'],
    'Teacher': ['lesson planning', 'classroom management', 'curriculum design', 'student assessment', 'subject expertise'],
    'Doctor': ['clinical skills', 'diagnosis', 'patient care', 'medical ethics', 'emergency procedures'],
    'Paralegal': ['legal research', 'document drafting', 'case management', 'legal terminology', 'attention to detail'],
    'Software Developer': ['python', 'java', 'problem solving', 'algorithms', 'version control'],
    'Investment Banker': ['financial modeling', 'valuation', 'excel', 'economics', 'mergers and acquisitions'],
    'Curriculum Developer': ['instructional design', 'pedagogy', 'learning theory', 'assessment design', 'education technology'],
    'Microbiologist': ['lab skills', 'microbial techniques', 'microscopy', 'data analysis', 'biosafety'],
    'Mechanical Engineer': ['cad', 'thermodynamics', 'mechanics', 'manufacturing processes', 'matlab'],
    'Biochemist': ['protein analysis', 'molecular biology', 'spectroscopy', 'lab techniques', 'enzymology'],
    'Nuclear Physicist': ['quantum mechanics', 'radiation safety', 'reactor physics', 'mathematics', 'simulations'],
    'Aerospace Engineer': ['aerodynamics', 'cad', 'propulsion systems', 'materials science', 'simulation tools'],
    'Animator': ['2d animation', '3d modeling', 'storyboarding', 'adobe animate', 'character design'],
    'Sound Engineer': ['audio editing', 'sound mixing', 'microphone setup', 'pro tools', 'acoustics'],
    'Financial Advisor': ['investment planning', 'retirement planning', 'tax knowledge', 'financial products', 'client communication'],
    'Landscape Architect': ['autocad', 'ecological design', 'plant science', 'urban planning', 'land use planning'],
    'Ecologist': ['field work', 'data analysis', 'environmental impact', 'biodiversity knowledge', 'statistical tools'],
    'Legal Analyst': ['legal research', 'case law analysis', 'writing skills', 'contracts', 'regulatory knowledge'],
    'Industrial-Organizational Psychologist': ['workplace behavior', 'data analysis', 'psychological testing', 'hr consulting', 'survey design'],
    'AI Researcher': ['machine learning', 'deep learning', 'neural networks', 'mathematics', 'research writing'],
    'Marketing Specialist': ['campaign execution', 'seo/sem', 'analytics', 'market research', 'crm tools'],
    'Risk Analyst': ['risk modeling', 'data analysis', 'regulatory knowledge', 'finance', 'problem solving'],
    'Music Therapist': ['music skills', 'therapy techniques', 'empathy', 'communication', 'client assessment'],
    'Entrepreneur': ['business strategy', 'fundraising', 'marketing', 'product development', 'risk management'],
    'Physicist': ['quantum physics', 'mathematics', 'research', 'experiment design', 'data analysis'],
    'Lawyer': ['legal knowledge', 'advocacy', 'research', 'negotiation', 'ethics'],
    'Dentist': ['oral health', 'dental procedures', 'patient care', 'anatomy', 'precision'],
    'Biomedical Engineer': ['biomedical devices', 'engineering principles', 'anatomy', 'regulations', 'design software'],
    'Astronomer': ['observational techniques', 'astrophysics', 'data analysis', 'telescopes', 'research'],
    'Zoologist': ['animal behavior', 'field work', 'ecology', 'data collection', 'species identification'],
    'Legal Consultant': ['legal compliance', 'contract review', 'corporate law', 'client advising', 'documentation'],
    'Construction Manager': ['project management', 'blueprint reading', 'safety standards', 'budgeting', 'team leadership'],
    'Pharmacist': ['pharmacology', 'prescription handling', 'patient counseling', 'regulations', 'drug interactions'],
    'Principal': ['school leadership', 'educational policy', 'teacher supervision', 'student welfare', 'community relations'],
    'Fluid Mechanics Engineer': ['fluid dynamics', 'cfd tools', 'mathematics', 'experimental methods', 'design'],
    'Advertising Manager': ['branding', 'media planning', 'creative strategy', 'budgeting', 'client communication'],
    'Architectural Technologist': ['technical drawing', 'building codes', 'cad', 'construction knowledge', 'materials science'],
    'Musician': ['instrument skills', 'music theory', 'practice discipline', 'performance', 'creativity'],
    'Quantum Physicist': ['quantum theory', 'mathematics', 'research', 'experimental design', 'data interpretation'],
    'Composer': ['music theory', 'composition techniques', 'notation software', 'instrumentation', 'creativity'],
    'Marketing Manager': ['marketing strategy', 'brand management', 'analytics', 'seo', 'communication'],
    'Teacher': ['lesson planning', 'classroom management', 'curriculum design', 'student assessment', 'subject expertise'],
    'Doctor': ['clinical skills', 'diagnosis', 'patient care', 'medical ethics', 'emergency procedures'],
    'Paralegal': ['legal research', 'document drafting', 'case management', 'legal terminology', 'attention to detail'],
    'Software Developer': ['python', 'java', 'problem solving', 'algorithms', 'version control'],
    'Investment Banker': ['financial modeling', 'valuation', 'excel', 'economics', 'mergers and acquisitions'],
    'Curriculum Developer': ['instructional design', 'pedagogy', 'learning theory', 'assessment design', 'education technology'],
    'Microbiologist': ['lab skills', 'microbial techniques', 'microscopy', 'data analysis', 'biosafety'],
    'Mechanical Engineer': ['cad', 'thermodynamics', 'mechanics', 'manufacturing processes', 'matlab'],
    'Biochemist': ['protein analysis', 'molecular biology', 'spectroscopy', 'lab techniques', 'enzymology'],
    'Nuclear Physicist': ['quantum mechanics', 'radiation safety', 'reactor physics', 'mathematics', 'simulations'],
    'Aerospace Engineer': ['aerodynamics', 'cad', 'propulsion systems', 'materials science', 'simulation tools'],
    'Animator': ['2d animation', '3d modeling', 'storyboarding', 'adobe animate', 'character design'],
    'Sound Engineer': ['audio editing', 'sound mixing', 'microphone setup', 'pro tools', 'acoustics'],
    'Financial Advisor': ['investment planning', 'retirement planning', 'tax knowledge', 'financial products', 'client communication'],
    'Landscape Architect': ['autocad', 'ecological design', 'plant science', 'urban planning', 'land use planning'],
    'Ecologist': ['field work', 'data analysis', 'environmental impact', 'biodiversity knowledge', 'statistical tools'],
    'Legal Analyst': ['legal research', 'case law analysis', 'writing skills', 'contracts', 'regulatory knowledge'],
    'Industrial-Organizational Psychologist': ['workplace behavior', 'data analysis', 'psychological testing', 'hr consulting', 'survey design'],
    'AI Researcher': ['machine learning', 'deep learning', 'neural networks', 'mathematics', 'research writing'],
    'Marketing Specialist': ['campaign execution', 'seo/sem', 'analytics', 'market research', 'crm tools'],
    'Risk Analyst': ['risk modeling', 'data analysis', 'regulatory knowledge', 'finance', 'problem solving'],
    'Music Therapist': ['music skills', 'therapy techniques', 'empathy', 'communication', 'client assessment'],
    'Entrepreneur': ['business strategy', 'fundraising', 'marketing', 'product development', 'risk management'],
    'Physicist': ['quantum physics', 'mathematics', 'research', 'experiment design', 'data analysis'],
    'Lawyer': ['legal knowledge', 'advocacy', 'research', 'negotiation', 'ethics'],
    'Dentist': ['oral health', 'dental procedures', 'patient care', 'anatomy', 'precision'],
    'Biomedical Engineer': ['biomedical devices', 'engineering principles', 'anatomy', 'regulations', 'design software'],
    'Astronomer': ['observational techniques', 'astrophysics', 'data analysis', 'telescopes', 'research'],
    'Zoologist': ['animal behavior', 'field work', 'ecology', 'data collection', 'species identification'],
    'Legal Consultant': ['legal compliance', 'contract review', 'corporate law', 'client advising', 'documentation'],
    'Construction Manager': ['project management', 'blueprint reading', 'safety standards', 'budgeting', 'team leadership'],
    'Pharmacist': ['pharmacology', 'prescription handling', 'patient counseling', 'regulations', 'drug interactions'],
    'Principal': ['school leadership', 'educational policy', 'teacher supervision', 'student welfare', 'community relations'],
    'Fluid Mechanics Engineer': ['fluid dynamics', 'cfd tools', 'mathematics', 'experimental methods', 'design'],
    'Advertising Manager': ['branding', 'media planning', 'creative strategy', 'budgeting', 'client communication'],
    'Architectural Technologist': ['technical drawing', 'building codes', 'cad', 'construction knowledge', 'materials science'],
    'Musician': ['instrument skills', 'music theory', 'practice discipline', 'performance', 'creativity'],
    'Quantum Physicist': ['quantum theory', 'mathematics', 'research', 'experimental design', 'data interpretation'],
    'Composer': ['music theory', 'composition techniques', 'notation software', 'instrumentation', 'creativity'],
    'Cyber Law Expert': ['cybersecurity laws', 'legal drafting', 'compliance', 'it knowledge', 'case analysis'],
    'Policy Analyst': ['policy research', 'data interpretation', 'writing', 'stakeholder analysis', 'impact evaluation'],
    'Public Relations Officer': ['media communication', 'branding', 'writing', 'event coordination', 'crisis management'],
    'Event Planner': ['budgeting', 'logistics', 'vendor coordination', 'scheduling', 'client communication'],
    'Tour Guide': ['local knowledge', 'storytelling', 'foreign languages', 'interpersonal skills', 'safety awareness'],
    'Insurance Underwriter': ['risk assessment', 'financial analysis', 'attention to detail', 'policy knowledge', 'communication'],
    'HR Specialist': ['recruitment', 'interviewing', 'employee relations', 'hr laws', 'onboarding'],
    'Retail Manager': ['inventory management', 'sales techniques', 'customer service', 'staff supervision', 'store operations'],
    'Public Health Officer': ['epidemiology', 'health programs', 'communication', 'community engagement', 'policy knowledge'],
    'AI Ethics Officer': ['ethical frameworks', 'bias detection', 'legal compliance', 'data privacy', 'ai awareness'],
    'Bioinformatician': ['python', 'biological data analysis', 'genomics', 'machine learning', 'databases'],
    'Telemedicine Specialist': ['remote diagnostics', 'clinical knowledge', 'telehealth tools', 'patient privacy', 'communication'],
    'Drone Technician': ['drone maintenance', 'hardware troubleshooting', 'battery systems', 'soldering', 'gps calibration'],
    'Waste Management Officer': ['environmental laws', 'recycling systems', 'logistics', 'data tracking', 'community outreach'],
    'Environmental Lawyer': ['environmental policy', 'legal research', 'court representation', 'drafting', 'regulatory knowledge'],
    'Marine Engineer': ['naval architecture', 'hydrodynamics', 'mechanical systems', 'ship design', 'problem solving'],
    'Astrobiologist': ['astrobiology', 'microbiology', 'planetary science', 'data interpretation', 'research skills'],
    'Sports Physiotherapist': ['injury rehab', 'anatomy', 'exercise therapy', 'patient handling', 'communication'],
    'Audiovisual Technician': ['sound systems', 'video recording', 'equipment setup', 'lighting', 'live events'],
    'Clinical Data Manager': ['clinical trials', 'data validation', 'database management', 'compliance', 'documentation'],
    'AI Prompt Engineer': ['nlp knowledge', 'prompt design', 'logical reasoning', 'data querying', 'tool usage'],
    'Sustainability Officer': ['green technologies', 'impact reporting', 'compliance', 'environmental strategy', 'stakeholder communication'],
    'Archivist': ['digitization', 'historical preservation', 'metadata standards', 'recordkeeping', 'data curation'],
    'Librarian': ['cataloging', 'research assistance', 'digital archives', 'user engagement', 'library management'],
    'Ethical Hacker': ['penetration testing', 'network security', 'vulnerability scanning', 'ethical hacking tools', 'compliance'],
    'App Developer': ['mobile frameworks', 'ui/ux design', 'api integration', 'debugging', 'app deployment'],
    'Botanist': ['plant identification', 'lab techniques', 'fieldwork', 'data collection', 'species classification'],
    'Geologist': ['rock analysis', 'mapping', 'seismic interpretation', 'field equipment', 'geospatial tools'],
    'Textile Designer': ['fabric knowledge', 'pattern creation', 'color theory', 'design software', 'trends'],
    'Voice UX Designer': ['voice interaction design', 'dialogue flow', 'nlp integration', 'usability testing', 'accessibility'],
    'Content Strategist': ['seo', 'editorial planning', 'analytics', 'user personas', 'storytelling'],
    'Digital Archivist': ['metadata tagging', 'digital preservation', 'information retrieval', 'system management', 'record control'],
    'Aquaculture Specialist': ['fish farming', 'water quality testing', 'breeding', 'disease management', 'sustainability'],
    'Cyber Forensic Analyst': ['digital evidence handling', 'file recovery', 'cyber laws', 'forensic tools', 'chain of custody'],
    'Climate Scientist': ['climate modeling', 'satellite data', 'statistics', 'field sampling', 'environmental policy'],
    'E-learning Developer': ['instructional design', 'scorm compliance', 'animation tools', 'video editing', 'lms management'],
    'Blockchain Architect': ['blockchain architecture', 'decentralized apps', 'smart contracts', 'tokenomics', 'scalability'],
    'Health Informatics Specialist': ['emr systems', 'data integration', 'clinical knowledge', 'privacy laws', 'data standards'],
    'AI Trainer': ['data labeling', 'annotation tools', 'model feedback', 'domain knowledge', 'consistency checking'],
    'Precision Agriculture Specialist': ['iot', 'drone usage', 'gps mapping', 'crop analytics', 'farm software']
}



learning_resources = {
    'python': ['Python Programming from Coursera', 'Automate the Boring Stuff with Python'],
    'statistics': ['Khan Academy Statistics', 'Statistics for Data Science by Udemy'],
    'machine learning': ['Andrew Ng ML Course (Coursera)', 'Hands-On ML with Scikit-Learn'],
    'data visualization': ['Tableau Official Tutorials', 'Data Visualization with Python (Matplotlib, Seaborn)'],
    'sql': ['SQL for Data Science by Coursera', 'LeetCode SQL Practice'],
    'accounting': ['Tally ERP Course', 'CA Foundation Books'],
    'taxation': ['GST and Income Tax online courses', 'CA Inter Taxation Notes'],
    'finance': ['Corporate Finance by Aswath Damodaran', 'Finance for Non-Finance'],
    'auditing': ['Auditing Theory and Practice', 'CA Audit Books'],
    'gst': ['GST Portal Tutorials', 'CA GST Modules'],
    'general knowledge': ['Manorama Yearbook', 'Current Affairs by Insights IAS'],
    'current affairs': ['Daily News Digest', 'The Hindu Newspaper'],
    'essay writing': ['IELTS Essay Writing Guide', 'General Essay Topics'],
    'history': ['India After Gandhi by Ramachandra Guha', 'NCERT History Books'],
    'geography': ['NCERT Geography Books', 'GC Leong Physical Geography'],
    'political science': ['Introduction to Political Theory', 'Indian Polity by Laxmikanth'],
    'biology': ['Trueman’s Biology', 'NEET Biology Online Lectures'],
    'chemistry': ['O.P. Tandon Chemistry Book', 'NEET Chemistry Crash Course'],
    'physics': ['HC Verma Concepts of Physics', 'NTA NEET Physics Notes'],
    'anatomy': ['Gray’s Anatomy', 'Human Anatomy for NEET'],
    'physiology': ['Guyton Physiology', 'Human Physiology for NEET'],
    'programming': ['CS50 by Harvard', 'Introduction to Programming in Python'],
    'algorithms': ['CLRS Book', 'Algorithms by GeeksforGeeks'],
    'data structures': ['Data Structures Easy to Advanced (Udemy)', 'GeeksforGeeks Tutorials'],
    'problem solving': ['CodeChef Practice', 'LeetCode Problems'],
    'system design': ['Grokking the System Design Interview', 'System Design Primer on GitHub'],
    'mechanics': ['Engineering Mechanics by Beer and Johnston', 'Mechanics Tutorials'],
    'thermodynamics': ['Thermodynamics Textbooks', 'NPTEL Thermodynamics Course'],
    'material science': ['Material Science by Callister', 'Material Science Lectures'],
    'maths': ['NCERT Maths Books', 'Higher Engineering Mathematics'],
    'design': ['CAD Tutorials', 'Machine Design Notes'],
    'marketing strategy': ['Digital Marketing Specialization (Coursera)', 'HubSpot Academy', 'Kotler’s Marketing Management Book'],
    'brand management': ['Brand Management by IE Business School (Coursera)', 'Udemy Branding Courses', 'LinkedIn Learning Branding'],
    'analytics': ['Google Data Analytics Certificate', 'Tableau Public Resources', 'IBM Data Analyst Certificate (Coursera)'],
    'seo': ['Moz SEO Guide', 'Google SEO Starter Guide', 'Yoast Academy'],
    'communication': ['Effective Communication Skills (Udemy)', 'Toastmasters International', 'TED Talks'],
    'lesson planning': ['Edutopia Resources', 'Teachers Pay Teachers', 'Coursera Teaching Courses'],
    'classroom management': ['The Classroom Management Book by Wong', 'EdX Classroom Strategies', 'TeachThought'],
    'curriculum design': ['Curriculum Development (edX)', 'Instructional Design on Coursera', 'NPTEL Curriculum Planning'],
    'student assessment': ['Assessment Strategies (ASCD)', 'NPTEL Assessment and Evaluation', 'Formative Assessment Toolkit'],
    'subject expertise': ['Khan Academy', 'CrashCourse on YouTube', 'MIT OpenCourseWare by Subject'],
    'clinical skills': ['Stanford Clinical Skills Guide', 'OSCE Cases', 'Marrow Clinical Videos'],
    'diagnosis': ['BMJ Best Practice', 'Differential Diagnosis App (Diagnosaurus)', 'Case Files Series'],
    'patient care': ['Patient Care by MedCram', 'Nursing Skills (Coursera)', 'Clinical Care Guidelines by WHO'],
    'medical ethics': ['NPTEL Bioethics', 'AMA Code of Medical Ethics', 'Coursera Ethics in Healthcare'],
    'emergency procedures': ['Basic Life Support (BLS) Certification', 'First Aid Guide by Red Cross', 'Advanced Trauma Life Support (ATLS)'],
     'fire safety': ['Fire Safety Training – National Fire Academy', 'Introduction to Fire Safety – Alison', 'NFPA Fire Safety Tips'],
    'risk assessment': ['Risk Assessment and Management – FutureLearn', 'ISO 31000 Risk Management Training', 'Health and Safety Executive (HSE) Guidelines'],
    'first aid': ['First Aid Manual – St. John Ambulance', 'First Aid Basics – Red Cross', 'CPR and First Aid – Udemy'],
    'law': ['Introduction to Law – Harvard edX', 'Indian Constitution – NPTEL', 'LegalEdge Law Prep Materials'],
    'criminology': ['Introduction to Criminology – Saylor Academy', 'Criminology – NPTEL', 'Criminal Justice Courses – Coursera'],
    'investigation techniques': ['Crime Scene Investigation – NIJ.gov', 'Forensic Investigation – FutureLearn', 'Detective Skills – Udemy'],
    'psychology': ['Psychology – MIT OpenCourseWare', 'Introduction to Psychology – Yale (YouTube)', 'Psychology 101 – Coursera'],
    'child psychology': ['Child Psychology – OpenLearn', 'Developmental Psychology – Coursera', 'Child Psychology Basics – Alison'],
    'counseling': ['Basic Counseling Skills – Udemy', 'Counseling Psychology – NPTEL', 'Psychological First Aid – Coursera'],
    'media ethics': ['Media Ethics and Governance – Coursera', 'Journalism Ethics – Poynter', 'Digital Media Ethics – UNESCO'],
    'journalism': ['Journalism Skills – BBC Academy', 'Journalism for Beginners – Coursera', 'Investigative Journalism – Udemy'],
    'public speaking': ['Public Speaking – TEDx Talks (YouTube)', 'Introduction to Public Speaking – Coursera', 'Toastmasters International Resources'],
    'writing': ['Creative Writing – Wesleyan Coursera', 'Effective Writing – NPTEL', 'Grammarly Blog and Resources'],
    'graphic design': ['Graphic Design Specialization – Coursera', 'Canva Design School', 'Adobe Creative Cloud Tutorials'],
    'illustration': ['Illustration Techniques – Domestika', 'Drawing & Illustration – Skillshare', 'Procreate Tutorials – YouTube'],
    'animation': ['Animation for Beginners – Udemy', '2D Animation with Adobe Animate – Coursera', 'Blender Animation Tutorials – YouTube'],
    'computer graphics': ['Computer Graphics – Udacity', 'Interactive Computer Graphics – SIGGRAPH Courses', 'OpenGL Programming Guide'],
    'networking': ['Computer Networking – Stanford Online', 'Networking Fundamentals – Cisco NetAcad', 'CompTIA Network+ Training'],
    'cybersecurity': ['Introduction to Cybersecurity – Cisco', 'Cybersecurity Specialization – Coursera', 'CompTIA Security+'],
    'web development': ['The Odin Project – Web Dev Curriculum', 'Full-Stack Web Development – Coursera', 'Frontend Masters Bootcamp'],
    'ui/ux design': ['Google UX Design – Coursera', 'Interaction Design Foundation Courses', 'Adobe XD Tutorials – YouTube'],
    'mobile app development': ['Android Development – Google Developers', 'iOS App Development – Stanford iTunes U', 'Flutter & Dart – Udemy'],
    'game development': ['Game Development with Unity – Coursera', 'GameDev.tv Courses – Udemy', 'Unreal Engine Online Learning'],
    'ai': ['AI for Everyone – Andrew Ng (Coursera)', 'Artificial Intelligence – MIT OpenCourseWare', 'DeepLearning.AI Specialization'],
    'deep learning': ['Deep Learning Specialization – Coursera', 'Fast.ai Practical Deep Learning', 'Deep Learning with PyTorch – Udacity'],
    'nlp': ['Natural Language Processing – Coursera', 'NLP with Transformers – HuggingFace Course', 'Speech and Language Processing – Book'],
    'computer vision': ['Computer Vision Basics – Coursera', 'PyImageSearch Tutorials', 'CS231n – Stanford CNNs'],
    'cloud computing': ['Cloud Computing Basics – IBM Coursera', 'AWS Training and Certification', 'Google Cloud Skills Boost'],
    'devops': ['DevOps on AWS – Coursera', 'CI/CD Pipelines – Udemy', 'DevOps Bootcamp – KodeKloud'],
    'blockchain': ['Blockchain Basics – Coursera', 'Ethereum & Solidity – Udemy', 'Blockchain Developer Path – Alchemy'],
    'robotics': ['Modern Robotics – Northwestern University (Coursera)', 'Robotics – MIT OpenCourseWare', 'ROS Tutorials – The Construct'],
    'iot': ['Introduction to IoT – Cisco NetAcad', 'IoT Specialization – Coursera', 'IoT Projects – Hackster.io']
    # Add more skills & resources as needed
}

# === User input for skills ===
user_text = input("\nEnter your skills and experience (comma-separated): ")
user_skills = [s.strip().lower() for s in user_text.split(',') if s.strip()]

user_embedding = model.encode(user_text, convert_to_tensor=True)

similarities = [util.cos_sim(user_embedding, cluster_emb)[0][0].item() for cluster_emb in cluster_embeddings]
best_cluster = np.argmax(similarities)

print(f"\n🎯 Most compatible cluster: Cluster {best_cluster} with similarity score: {similarities[best_cluster]:.4f}")
print("\nRecommended careers for you from this cluster:")
for career in all_cluster_careers[best_cluster]:
    print(f" - {career}")

# Aggregate all skills required for careers in best cluster
cluster_skill_set = set()
for career in all_cluster_careers[best_cluster]:
    cluster_skill_set.update(career_skill_map.get(career, []))

missing_skills = cluster_skill_set - set(user_skills)

if missing_skills:
    print("\n⚠️ Skills you might want to develop for this cluster:")
    for skill in sorted(missing_skills):
        print(f" - {skill.capitalize()}")

    print("\n📚 Suggested learning resources:")
    for skill in sorted(missing_skills):
        resources = learning_resources.get(skill, [f"Search online courses for {skill.capitalize()}"])
        print(f"\n{skill.capitalize()}:")
        for res in resources:
            print(f"  * {res}")
        # Also show in Streamlit if running in Streamlit context
        try:
            st.write(f"**{skill.capitalize()}**:")
            for res in resources:
                st.write(f"- {res}")
        except Exception:
            pass

        else:
            print("\n✅ You already have most of the key skills for this cluster!")
