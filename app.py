from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Job description
job_desc = "Looking for a software engineer with experience in Python and machine learning."

# Sample resumes (you can replace these with actual resume text)
resumes = [
    "Experienced in Java, web development, and databases.",
    "Expert in Python, AI, and deep learning models.",
    "Worked on cloud computing and DevOps automation.",
    "Machine learning engineer with strong Python knowledge.",
    "Software developer with experience in C++, Java, and Python."
]
# Convert job description and resumes to embeddings
job_embedding = model.encode(job_desc)
resume_embeddings = model.encode(resumes)
# Compute cosine similarity
similarity_scores = cosine_similarity([job_embedding], resume_embeddings)[0]
# Rank resumes (higher score = better match)
ranked_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
ranked_resumes = [(resumes[i], similarity_scores[i]) for i in ranked_indices]

# Display results
print("\n🔹 Ranked Resumes (Higher Score = Better Match):\n")
for rank, (resume, score) in enumerate(ranked_resumes, 1):
    print(f"Rank {rank}: Score = {score:.4f} | Resume: {resume}")
# Set a similarity threshold (e.g., 0.5) or pick top N candidates
threshold = 0.5
top_candidates = [res for res, score in ranked_resumes if score >= threshold]

print("\n🔹 Shortlisted Candidates:")
for i, candidate in enumerate(top_candidates, 1):
    print(f"{i}. {candidate}")
