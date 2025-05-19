from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    cv_text = data['cv_text']
    jobs = data['jobs']

    cv_embedding = model.encode(cv_text, convert_to_tensor=True)

    # Ubah 'requirements' dengan 'benefits' atau kolom lain yang ada di database jika 'requirements' tidak ada
    job_embeddings = model.encode(
        [job['description'] + ' ' + (job.get('requirements', '') or job.get('benefits', '')) for job in jobs],
        convert_to_tensor=True
    )

    scores = util.cos_sim(cv_embedding, job_embeddings)[0].tolist()
    
    # Menggabungkan ID pekerjaan dengan skor
    job_score_pairs = list(zip([job['id'] for job in jobs], scores))
    job_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # Mengambil 10 pekerjaan terbaik
    filtered_jobs = [pair for pair in job_score_pairs if pair[1] > 0.3]

    top_jobs = filtered_jobs[:10]

    return jsonify({
        'recommended_job_ids': [job_id for job_id, score in top_jobs],
        'similarities': [score for job_id, score in top_jobs]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
