# sign-language-detector-python
# 🖐️ ASL Sign Language Detector

Deteksi alfabet bahasa isyarat Amerika (ASL) secara real-time menggunakan Python, Mediapipe, dan model machine learning berbasis Random Forest.

---

## 📷 Apa yang dilakukan?
- Menggunakan webcam untuk mendeteksi gesture tangan ASL.
- Mengenali huruf.
- Memberikan feedback suara (opsional) dan menampilkan overlay gambar (opsional).
- Mendukung dua mode: **Normal** dan **Interactive** (dengan kategori Fruit/Animal) **OPSIONAL**.

---

## 🛠️ Fitur Utama
✅ Real-time ASL detection  
✅ GUI fullscreen berbasis tkinter  
✅ Feedback suara per huruf  
✅ Gambar overlay di atas wajah pengguna  
✅ Dataset collect dan training model custom

---

## 📁 Struktur Folder (rekomendasi)
SIGN-LANGUAGE-DETECTOR-PYTHON/
├── collect_imgs.py
├── create_dataset.py
├── dataset_show.py
├── train_classifier.py
├── asl_detector_fix.py
├── model.p
├── requirements.txt
├── LICENSE
├── .gitignore
├── README.md

---

## 📦 Resource tambahan
Untuk gambar overlay dan file audio feedback, siapkan sendiri dengan struktur:
images/
normal/A.png, normal/B.png, ...
fruit/A.png, fruit/B.png, ...
animal/A.png, animal/B.png, ...

voice/
a.mp3, b.mp3, ...
af.mp3, bf.mp3, ... (untuk Fruit mode)
aa.mp3, ba.mp3, ... (untuk Animal mode)

silahkan disesuaikan dengan keperluan

**Catatan:**  
- `A.png` dll harus memiliki channel alpha (PNG transparan).
- Nama file audio harus sesuai huruf yang dikenali.
- Letakkan folder `images/` dan `voice/` di root project.

---

## 🚀 Cara Install & Jalankan
1. Clone repo ini:
   ```bash
   git clone https://github.com/rizdhandevcode/sign-language-detector-python.git
   cd sign-language-detector-python
2. Buat virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
3. Install dependensi:
   ```bash
   pip install -r requirements.txt

## 🚀 Workflow Lengkap

### 1️⃣ Collect Data
Jalankan untuk mengumpulkan dataset gesture dari webcam:
```bash
python collect_imgs.py
```

Akan menyimpan gambar ke data/0/, data/1/, dst.

### 2️⃣ Buat Dataset Fitur
Konversi gambar ke dataset numerik (84 fitur per gambar):
```bash
python create_dataset.py
```

### 3️⃣ (Opsional) Cek Dataset
Pastikan landmark tangan terdeteksi di gambar hasil collect:
```bash
python dataset_show.py
```

### 4️⃣ Latih Model
Latih model Random Forest dari dataset:
```bash
python train_classifier.py
```

### 5️⃣ Jalankan Detektor ASL
Deteksi gesture secara real-time:
```bash
python asl_detector_fix.py
```

---

## 🙏 Kontribusi
Pull request dan masukan sangat dipersilakan! Silakan fork repo ini dan buat PR untuk kontribusi.

---

## 📞 Kontak
Untuk pertanyaan atau kerjasama, hubungi saya melalui Issues di repo ini.

---
