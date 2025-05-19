import os
import fitz
from flask import Flask, request, send_file, jsonify
import io
import logging
import re

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
logging.basicConfig(level=logging.INFO)

@app.route('/highlight', methods=['POST'])
def highlight_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "PDF file is required"}), 400
            
        pdf_file = request.files['pdf']
        keywords_str = request.form.get('keywords', '')

        # Ekstrak kata dari hasil AI, dan filter panjang > 3
        all_keywords = [
            word for word in set(re.findall(r'\b\w+\b', keywords_str.lower()))
            if len(word) > 3
        ]

        # Tambahkan keyword penting manual (agar tetap di-highlight meskipun tidak muncul dari AI)
        section_headers = ['education', 'experience', 'skills', 'projects', 'summary']  # <-- Tambahan
        all_keywords = list(set(all_keywords + section_headers))  # <-- Tambahan

        print(f"Keywords setelah pemrosesan: {all_keywords}")

        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Invalid file type"}), 400

        pdf_data = pdf_file.read()
        if len(pdf_data) == 0:
            return jsonify({"error": "Empty PDF file"}), 400

        doc = fitz.open(stream=pdf_data, filetype="pdf")
        
        for page in doc:
            for keyword in all_keywords:
                text_instances = page.search_for(keyword, flags=1)
                for inst in text_instances:
                    print(f"Highlight: {keyword} -> {inst}")
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=(1, 0, 0))  # Merah

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        # === Tambahan: Simpan ke folder lokal ===
         output_dir = r'C:\xampp\htdocs\openjob_dev_jupli\storage\cvs'
        os.makedirs(output_dir, exist_ok=True)

        # Tentukan path file output
        output_path = os.path.join(output_dir, 'highlighted.pdf')
        with open(output_path, 'wb') as f:
            f.write(buffer.getbuffer())

        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='highlighted.pdf'
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": "Processing error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
