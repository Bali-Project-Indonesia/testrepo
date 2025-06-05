from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re
import traceback
from bs4 import BeautifulSoup

# === 1. Load Model Cover_genie ===
print("🚀 Memuat model Cover_genie untuk cover letter...")
tokenizer = AutoTokenizer.from_pretrained("Hariharavarshan/Cover_genie")
model = AutoModelForSeq2SeqLM.from_pretrained("Hariharavarshan/Cover_genie")

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)
print("✅ Model dan pipeline berhasil dimuat.")

# === 2. Pembersih Output ===
def clean_cover_letter(text):
    # Hapus tanggal
    text = re.sub(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b", "", text)
    # Hapus email
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text)
    # Hapus nomor telepon
    text = re.sub(r"\(?\+?\d{1,3}?\)?[-.\s]?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}", "", text)
    return text.strip()

def remove_repeated_phrases(text):
    # Menghilangkan kalimat yang berulang berturut-turut
    sentences = re.split(r'(?<=[.!?]) +', text)
    cleaned = []
    prev = ""
    for s in sentences:
        if s != prev:
            cleaned.append(s)
        prev = s
    return " ".join(cleaned)

# === 3. Fungsi bantu strip HTML ===
def strip_html(raw_html):
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")

# === 4. Generate Cover Letter ===
def generate_cover_letter(cv_text, job_text, full_name, company_name, location, email=None, phone=None):
    try:
        # === Preprocess Text ===
        cv_clean = strip_html(cv_text)[:1000]
        job_clean = strip_html(job_text)[:1000]

        # === Prompt lebih ringkas untuk AI, hanya isi tengah surat ===
        ai_prompt = (
            f"Generate the main body of a cover letter for {full_name} applying at {company_name} in {location}. "
            f"Use the resume and job description below. Be concise, avoid repeating phrases.\n\n"
            f"Resume:\n{cv_clean}\n\n"
            f"Job Description:\n{job_clean}\n\n"
            f"Cover Letter Body:"
        )
        print("=== PROMPT HYBRID YANG DIKIRIM ===")
        print(ai_prompt)

        # === Generate using model ===
        output = pipe(
            ai_prompt,
            max_length=350,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.92,
            top_k=40,
            no_repeat_ngram_size=3
        )
        print("=== OUTPUT MENTAH PIPE ===")
        print(output)

        # === Clean Output ===
        body = clean_cover_letter(output[0]['generated_text'])
        body = remove_repeated_phrases(body)

        # === Build Final Letter with Template ===
        contact_header = f"{full_name}"
        if email:
            contact_header += f" | {email}"
        if phone:
            contact_header += f" | {phone}"

        letter = (
            f"{contact_header}\n\n"
            f"Dear Hiring Team at {company_name},\n\n"
            f"{body}\n\n"
            f"Warm regards,\n"
            f"{full_name}"
        )

        return letter

    except Exception as e:
        print("❌ Terjadi kesalahan saat generate_cover_letter:")
        traceback.print_exc()
        return "Gagal membuat surat lamaran karena error internal."
