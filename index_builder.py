import mysql.connector
import faiss
import numpy as np
import pickle
from models.embedding_model import model
from sklearn.preprocessing import normalize  # ✅ Tambahkan ini

def fetch_jobs_from_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # ganti sesuai
        database="jobsportal_db"
    )
    cursor = conn.cursor(dictionary=True)

    # ✅ JOIN dengan tabel functional_areas
    cursor.execute("""
        SELECT jobs.id, jobs.description, jobs.benefits, jobs.functional_area_id, functional_areas.functional_area
        FROM jobs
        LEFT JOIN functional_areas ON jobs.functional_area_id = functional_areas.functional_area_id
    """)
    jobs = cursor.fetchall()
    conn.close()
    return jobs

def rebuild_faiss_index():
    jobs = fetch_jobs_from_db()

    # ✅ Gabungkan semua informasi relevan untuk embedding
    texts = [
        (job['description'] or '') + ' ' + 
        (job.get('benefits', '') or '') + ' ' + 
        (job.get('functional_area', '') or '')
        for job in jobs
    ]

    embeddings = model.encode(texts)

    # ✅ Normalisasi ke unit vector (cosine sim)
    embeddings = normalize(np.array(embeddings), norm='l2').astype("float32")

    # ✅ Ganti ke IndexFlatIP (inner product = cosine sim jika normalized)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "faiss_index.bin")

    job_id_map = {i: job["id"] for i, job in enumerate(jobs)}
    with open("job_id_map.pkl", "wb") as f:
        pickle.dump(job_id_map, f)

    print("✅ FAISS cosine similarity index updated.")

def delete_from_faiss(job_id_to_delete):
    index = faiss.read_index("faiss_index.bin")
    with open("job_id_map.pkl", "rb") as f:
        job_id_map = pickle.load(f)

    reverse_map = {job_id: i for i, job_id in job_id_map.items()}

    if job_id_to_delete not in reverse_map:
        print(f"⚠️ job_id {job_id_to_delete} tidak ditemukan dalam index FAISS.")
        return

    delete_idx = reverse_map[job_id_to_delete]

    all_ids = sorted(job_id_map.keys())
    keep_indices = [i for i in all_ids if job_id_map[i] != job_id_to_delete]

    all_embeddings = index.reconstruct_n(0, index.ntotal)
    embeddings_to_keep = np.array([all_embeddings[i] for i in keep_indices]).astype("float32")

    # ✅ JANGAN LUPA: tetap gunakan IndexFlatIP saat rebuild
    new_index = faiss.IndexFlatIP(embeddings_to_keep.shape[1])
    new_index.add(embeddings_to_keep)
    faiss.write_index(new_index, "faiss_index.bin")

    new_job_id_map = {i: job_id_map[old_i] for i, old_i in enumerate(keep_indices)}
    with open("job_id_map.pkl", "wb") as f:
        pickle.dump(new_job_id_map, f)

    print(f"🗑️ FAISS index updated — job_id {job_id_to_delete} removed.")
