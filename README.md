# Serenity — Deteksi Indikasi Depresi dari Teks

Aplikasi Streamlit untuk mendeteksi indikasi depresi pada teks singkat (gaya tweet),
menggunakan model **Stacking Ensemble** (Logistic Regression + Linear SVM + XGBoost +
CatBoost) hasil dari notebook riset `NotebookV1StackingModel90Acc.ipynb`.

---

## 1. Struktur Folder

```
deploy/
├── app.py                   # Aplikasi utama Streamlit (UI + logika)
├── preprocessing.py         # Pipeline pembersihan teks (identik dengan training)
├── feature_engineering.py   # TF-IDF + fitur numerik + feature selection
├── requirements.txt         # Daftar dependency
├── models/                  # LETAKKAN file .pkl hasil training di sini
│   ├── tfidf_vectorizer.pkl
│   ├── numeric_scaler.pkl
│   ├── feature_selector.pkl
│   └── best_model.pkl
└── README.md
```

## 2. PENTING — Artefak Model yang Dibutuhkan

Notebook Anda **hanya diupload sebagai file `.ipynb`**, sedangkan file hasil
training (`.pkl`) tersimpan lokal di environment Colab/Jupyter Anda saat notebook
dijalankan (`tfidf_vectorizer.pkl`, `numeric_scaler.pkl`, `feature_selector.pkl`,
`best_model.pkl`, dst).

Karena itu, **sebelum aplikasi ini bisa berjalan**, Anda perlu:

1. Menjalankan ulang notebook `NotebookV1StackingModel90Acc.ipynb` sampai akhir
   (atau di lingkungan yang sudah punya file `.pkl` tersebut).
2. Mengambil 4 file berikut, lalu menyalinnya ke folder `models/` pada proyek ini:
   - `tfidf_vectorizer.pkl`
   - `numeric_scaler.pkl`
   - `feature_selector.pkl`
   - `best_model.pkl` (model terbaik yang otomatis dipilih notebook
     berdasarkan skor F1 tertinggi — pada eksperimen Anda kemungkinan besar
     ini adalah **Stacking Classifier**)

Jika di Google Colab, Anda bisa mengunduhnya dengan menambahkan sel berikut
di akhir notebook:

```python
from google.colab import files
for f in ["tfidf_vectorizer.pkl", "numeric_scaler.pkl",
          "feature_selector.pkl", "best_model.pkl"]:
    files.download(f)
```

Lalu pindahkan ke folder `models/` proyek Streamlit ini.

> ⚠️ Tanpa keempat file ini, aplikasi tetap akan berjalan (tidak crash) namun
> akan menampilkan pesan bahwa konfigurasi belum lengkap.

## 3. Instalasi & Menjalankan Secara Lokal

```bash
# 1. Buat virtual environment (disarankan)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependency
pip install -r requirements.txt

# 3. Download resource NLTK yang dibutuhkan (sekali saja)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 4. Pastikan folder models/ sudah berisi 4 file .pkl (lihat bagian 2)

# 5. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka otomatis di `http://localhost:8501`.

## 4. Deploy ke Streamlit Community Cloud

1. Push seluruh folder `deploy/` (termasuk `models/*.pkl`) ke repository GitHub.
   - Jika file `.pkl` berukuran besar (misalnya TF-IDF dengan 20.000 fitur),
     gunakan **Git LFS** agar tidak melebihi batas ukuran repo GitHub.
2. Buka [streamlit.io/cloud](https://streamlit.io/cloud), login dengan GitHub.
3. Klik **New app**, pilih repository dan branch Anda.
4. Set **Main file path** ke `app.py`.
5. Tambahkan file `packages.txt` (opsional) jika ada dependency sistem —
   untuk kombinasi library di atas biasanya tidak diperlukan.
6. Klik **Deploy**. Streamlit Cloud akan otomatis membaca `requirements.txt`
   dan menginstal seluruh dependency, termasuk mengunduh resource NLTK saat
   `preprocessing.py` pertama kali dipanggil (karena kode sudah menangani
   auto-download jika resource belum ada — lihat `preprocessing.py`).

## 5. Catatan Teknis Penting

### a. Perbaikan bug preprocessing saat inference

Pada notebook riset asli, fungsi `predict_tweet()` di sel terakhir memakai
`simple_clean()` — versi pembersihan teks yang **lebih sederhana** (tidak ada
stopword removal maupun lemmatization) dibanding `clean_text()` yang dipakai
saat membangun fitur training.

Ini menyebabkan **data-training mismatch**: model dilatih dengan teks yang
sudah di-stopword-removal + lemmatization, namun saat inference teks tidak
diproses dengan cara yang sama — sehingga prediksi bisa menjadi kurang akurat.

**Pada aplikasi deployment ini, bug tersebut sudah diperbaiki.** Modul
`preprocessing.py` di sini menggunakan `clean_text()` versi lengkap (identik
dengan proses training), sehingga konsisten dan hasil prediksi lebih dapat
diandalkan.

### b. Urutan fitur numerik

Urutan 6 fitur numerik pada `feature_engineering.py`
(`word_count, char_count, avg_word_length, uppercase_ratio, polarity, subjectivity`)
harus **sama persis** dengan urutan kolom `numeric_cols` pada notebook saat
`StandardScaler` di-fit. Jangan diubah urutannya, karena scaler bergantung
pada urutan kolom yang konsisten.

### c. Model yang dipakai

Aplikasi memuat `best_model.pkl` — file yang pada notebook dipilih secara
otomatis dari model dengan F1-score tertinggi di antara: Logistic Regression,
Linear SVM, XGBoost, CatBoost, Soft Voting, dan Stacking Classifier.

## 6. Disclaimer

Aplikasi ini dibuat untuk tujuan edukasi dan eksplorasi riset. **Bukan alat
diagnosis klinis** dan tidak boleh menggantikan konsultasi dengan psikolog,
psikiater, atau tenaga profesional kesehatan mental lainnya.
