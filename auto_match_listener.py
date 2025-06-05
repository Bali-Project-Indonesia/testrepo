import redis
import json
import mysql.connector
import pickle
import numpy as np
import faiss
import os
import fitz  # PyMuPDF
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, util
from cover_letter_generator import generate_cover_letter

# Redis setup
r = redis.Redis(host='localhost', port=6379, db=0)
pubsub = r.pubsub()
pubsub.subscribe("job_updates")

# Load FAISS
faiss_index = faiss.read_index("faiss_index.bin")
with open("job_id_map.pkl", "rb") as f:
    job_id_map = pickle.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

print("👂 Listening to job_updates for new job posts...")

CV_FOLDER_PATH = "C:\\xampp\\htdocs\\openjob_dev_jupli\\public\\storage\\cvs"

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="jobsportal_db"
    )

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            page_text = page.get_text("text")
            text += page_text
        clean_text = text.replace("�", "").strip()
        return clean_text
    except Exception as e:
        print(f"❌ Gagal membaca PDF: {e}")
        return ""

def find_file_insensitive(folder, filename):
    for f in os.listdir(folder):
        if f.lower() == filename.lower():
            return os.path.join(folder, f)
    return None

for message in pubsub.listen():
    if message['type'] != 'message':
        continue

    try:
        print(f"📨 Menerima pesan Redis: {message}")

        payload = json.loads(message['data'])
        action = payload.get('action')
        job_id = payload.get('job_id')

        if action not in ['posted', 'updated']:
            print(f"ℹ️ Aksi tidak relevan: {action}")
            continue

        print(f"🆕 Job baru diproses: ID {job_id}")

        conn = connect_db()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT 
                jobs.*, 
                companies.name AS company_name, 
                companies.location AS company_location,
                companies.website
            FROM jobs
            JOIN companies ON jobs.company_id = companies.id
            WHERE jobs.id = %s
        """, (job_id,))
        job = cursor.fetchone()
        print(f"📄 Data job diambil: {job}")

        if not job:
            print("⚠️ Job tidak ditemukan di database.")
            continue

        job_text = f"""
        Job Title: {job.get("title", "")}
        Company: {job.get("company_name", "")}
        Location: {job.get("company_location", "")}
        Employment Type: {job.get("type", "")}
        Level: {job.get("level", "")}
        Benefits: {job.get("benefits", "")}

        Job Description:
        {job.get("description", "")}
        """
        job_vector = model.encode([job_text])
        job_vector = normalize(job_vector, norm="l2").astype("float32")

        cursor.execute("""
            SELECT profile_cvs.id as cv_id,
                   profile_cvs.cv_file as cv_file,
                   users.id as user_id,
                   users.first_name,
                   users.last_name,
                   users.current_salary,
                   users.expected_salary,
                   users.salary_currency
            FROM profile_cvs
            JOIN users ON users.id = profile_cvs.user_id
            WHERE users.is_active = 1
              AND users.package_end_date > NOW()
              AND users.jobs_quota > users.availed_jobs_quota;
        """)
        cvs = cursor.fetchall()

        print(f"👤 Total CV premium yang diproses: {len(cvs)}")

        for cv in cvs:
            full_name = f"{cv['first_name']} {cv['last_name']}"
            cv_path = os.path.join(CV_FOLDER_PATH, cv['cv_file'])
            print(f"📁 Lokasi file CV awal: {cv_path}")

            if not os.path.exists(cv_path):
                cv_path_ci = find_file_insensitive(CV_FOLDER_PATH, cv['cv_file'])
                if cv_path_ci:
                    print(f"🔍 File ditemukan dengan pencarian case-insensitive: {cv_path_ci}")
                    cv_path = cv_path_ci
                else:
                    print(f"⚠️ File CV tidak ditemukan: {cv_path}")
                    continue

            print(f"📁 Lokasi file CV yang digunakan: {cv_path}")
            cv_text = extract_text_from_pdf(cv_path)
            if not cv_text:
                print(f"⚠️ CV kosong/gagal dibaca: {cv['cv_file']}")
                continue

            print(f"📄 Isi CV (max 1500 chars) untuk {full_name}:")
            print(cv_text[:1500])
            print("-" * 50)

            cv_vector = model.encode([cv_text])
            cv_vector = normalize(cv_vector, norm="l2").astype("float32")
            score = float(util.cos_sim(job_vector, cv_vector)[0][0])

            if score > 0.01:
                print(f"🤝 Cocok! {full_name} (skor: {score:.2f})")

                cursor.execute("""
                    SELECT COUNT(*) as count FROM job_apply
                    WHERE user_id = %s
                      AND DATE(created_at) = CURDATE()
                      AND status = 'auto_applied'
                """, (cv["user_id"],))
                auto_applied_today = cursor.fetchone()["count"]

                if auto_applied_today >= 5:
                    print(f"⛔ Melebihi limit harian auto-apply: {full_name}")
                    continue

                cursor.execute("""
                    INSERT INTO job_apply (user_id, job_id, cv_id, current_salary, expected_salary, salary_currency, status, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                """, (
                    cv["user_id"],
                    job_id,
                    cv["cv_id"],
                    cv["current_salary"],
                    cv["expected_salary"],
                    cv["salary_currency"],
                    "auto_applied"
                ))
                conn.commit()
                jobapply_id = cursor.lastrowid

                cursor.execute("""
                    UPDATE users SET availed_jobs_quota = availed_jobs_quota + 1
                    WHERE id = %s
                """, (cv["user_id"],))
                conn.commit()

                try:
                    cover_letter = generate_cover_letter(
                        cv_text=cv_text,
                        job_text=job_text,
                        full_name=full_name,
                        company_name=job.get("company_name", "Company"),
                        location=job.get("company_location", "Location"),
                        email=None,
                        phone=None
                    )
                except Exception as e:
                    print(f"❌ Error saat generate_cover_letter: {e}")
                    cover_letter = None

                print(f"🖊️ Cover letter untuk {full_name} (apply_id: {jobapply_id}):")
                print(cover_letter[:12000] + ("..." if len(cover_letter) > 12000 else ""))

                if cover_letter and not cover_letter.startswith("Gagal"):
                    try:
                        cursor.execute("""
                            INSERT INTO cover_letters (jobapply_id, cover_letter, created_at)
                            VALUES (%s, %s, NOW())
                        """, (jobapply_id, cover_letter))
                        conn.commit()
                        print(f"✅ Cover letter tersimpan di database.")
                    except mysql.connector.Error as err:
                        print(f"❌ MySQL Error: {err}")
                        conn.rollback()
                else:
                    print("⛔ Cover letter tidak valid, tidak disimpan.")

        cursor.close()
        conn.close()

    except json.JSONDecodeError:
        print("❌ Gagal decode pesan Redis (bukan JSON valid)")
    except Exception as e:
        print(f"❌ Error umum: {e}")
