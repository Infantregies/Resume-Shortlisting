from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import PyPDF2
import re

# Load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Job description
job_desc = "looking for a java developer"

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Function to extract candidate name from the PDF file name
def extract_name_from_filename(file_name):
    # Assume the candidate's name is after the first " - " and before ".pdf"
    match = re.search(r" - (.+?)\.pdf", file_name)
    if match:
        return match.group(1)
    return file_name.replace(".pdf", "")  # Fallback if no pattern match

# Directory containing resumes in PDF format
resumes_dir = "C:/Users/jenit/PycharmProjects/bitHack/resumes"  # Replace with the actual directory path

# Extract text from all PDFs in the directory
resumes = []
candidate_names = []
for file in os.listdir(resumes_dir):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(resumes_dir, file)
        resume_text = extract_text_from_pdf(pdf_path)
        if resume_text.strip():  # Add to the list if the text is not empty
            resumes.append(resume_text)
            candidate_names.append(extract_name_from_filename(file))  # Extract candidate name

# Convert job description and resumes to embeddings
job_embedding = model.encode(job_desc)
resume_embeddings = model.encode(resumes)

# Compute cosine similarity
similarity_scores = cosine_similarity([job_embedding], resume_embeddings)[0]

# Rank resumes (higher score = better match)
ranked_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
ranked_resumes = [(candidate_names[i], resumes[i], similarity_scores[i]) for i in ranked_indices]

# Display results
print("\n🔹 Ranked Resumes (Higher Score = Better Match):\n")
for rank, (candidate_name, resume, score) in enumerate(ranked_resumes, 1):
    print(f"Rank {rank}: Score = {score:.4f} | Candidate: {candidate_name}")

# Set a similarity threshold (e.g., 0.3) or pick top N candidates
threshold = 0.3
top_candidates = [(candidate_name, score) for candidate_name, _, score in ranked_resumes if score >= threshold]

# Display shortlisted candidates
print("\n🔹 Shortlisted Candidates:")
if top_candidates:
    for i, (candidate_name, score) in enumerate(top_candidates, 1):
        print(f"{i}. Candidate: {candidate_name} | Score: {score:.4f}")
else:
    print("No candidates met the threshold.")
