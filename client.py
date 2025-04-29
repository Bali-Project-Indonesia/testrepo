# client.py

import requests

url = 'http://localhost:5000/highlight'  # Alamat endpoint Flask
pdf_path = r'C:\Users\user\Downloads\highlighted_cv_85_1745548307259.pdf'

# Kirim file dan data keywords
with open(pdf_path, 'rb') as f:
    files = {'pdf': f}
    data = {'keywords': 'tes,highlight'}
    response = requests.post(url, files=files, data=data)

# Simpan hasil jika sukses
if response.status_code == 200:
    with open('hasil_highlight.pdf', 'wb') as out_file:
        out_file.write(response.content)
    print("✅ File hasil disimpan sebagai 'hasil_highlight.pdf'")
else:
    print("❌ Gagal memproses PDF")
    print("Response:", response.text)
