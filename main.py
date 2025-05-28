from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import picklefrom fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import pickle
from models.embedding_model import model
from sklearn.preprocessing import normalize  # tambahkan ini

app = FastAPI()

# Load FAISS index & job_id_map
faiss_index = faiss.read_index("faiss_index.bin")
with open("job_id_map.pkl", "rb") as f:
    job_id_map = pickle.load(f)

class CVInput(BaseModel):
    cv_text: str

@app.post("/recommend")
def recommend(data: CVInput):
    if not faiss_index:
        return {"error": "Index not loaded"}, 500

    #  Encode + normalisasi CV
    query_embedding = model.encode([data.cv_text])
    query_embedding = normalize(query_embedding, norm='l2').astype("float32")

    #  Cari top 50 berdasarkan cosine similarity (inner product)
    D, I = faiss_index.search(query_embedding, 50)
    

    job_score_pairs = []
    for idx, score in zip(I[0], D[0]):
        if idx in job_id_map:
            job_score_pairs.append((job_id_map[idx], float(score)))

    #  Ambil yang paling mirip (skor > 0.3)
    # filtered_jobs = [pair for pair in job_score_pairs if pair[1] > 0.7]
    filtered_jobs = [pair for pair in job_score_pairs if pair[1] > 0.3]

    #  Urutkan dari skor tertinggi (semakin mirip)
    top_jobs = sorted(filtered_jobs, key=lambda x: x[1], reverse=True)[:10]

    #  Tampilkan ke console
    print("📋 Filtered Jobs:", filtered_jobs)
    print("🏆 Top 10 Jobs:", top_jobs)

    return {
        "recommended_job_ids": [job_id for job_id, score in top_jobs],
        "similarities": [score for job_id, score in top_jobs]
    }

from models.embedding_model import model

app = FastAPI()

# Load FAISS index & job_id_map
faiss_index = faiss.read_index("faiss_index.bin")
with open("job_id_map.pkl", "rb") as f:
    job_id_map = pickle.load(f)

class CVInput(BaseModel):
    cv_text: str

@app.post("/recommend")
def recommend(data: CVInput):
    if not faiss_index:
        return {"error": "Index not loaded"}, 500

    # Encode CV
    query_embedding = model.encode([data.cv_text]).astype("float32")

    # Search top 50 (biar bisa difilter dulu)
    D, I = faiss_index.search(query_embedding, 50)

    # Zip job IDs and scores
    job_score_pairs = []
    for idx, score in zip(I[0], D[0]):
        if idx in job_id_map:
            job_score_pairs.append((job_id_map[idx], float(score)))

    # Filter yang skornya > 0.3 (30%)
    filtered_jobs = [pair for pair in job_score_pairs if pair[1] > 0.3]
    # filtered_jobs = [pair for pair in job_score_pairs if 0.5 <= pair[1] <= 1.0]
    # Ambil 10 terbaik
    top_jobs = filtered_jobs[:10]

    return {
        "recommended_job_ids": [job_id for job_id, score in top_jobs],
        "similarities": [score for job_id, score in top_jobs]
    }
