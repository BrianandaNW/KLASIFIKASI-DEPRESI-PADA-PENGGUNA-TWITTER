"""
feature_engineering.py
=======================
Menyatukan seluruh proses transformasi fitur untuk inference, identik
dengan pipeline pada notebook training:

1. Cleaning teks           -> preprocessing.clean_text()
2. TF-IDF (1-3 ngram)       -> tfidf_vectorizer.pkl
3. Fitur numerik tambahan   -> word_count, char_count, avg_word_length,
                                uppercase_ratio, polarity, subjectivity
4. Scaling fitur numerik    -> numeric_scaler.pkl
5. Gabung TF-IDF + numerik  -> hstack
6. Feature selection        -> feature_selector.pkl (SelectKBest)

Fungsi utama: `build_features(raw_text, artifacts)` mengembalikan matrix
sparse siap pakai oleh model klasifikasi.
"""

from scipy.sparse import csr_matrix, hstack
from textblob import TextBlob

from preprocessing import clean_text, uppercase_ratio

NUMERIC_COLS = [
    "word_count",
    "char_count",
    "avg_word_length",
    "uppercase_ratio",
    "polarity",
    "subjectivity",
]


def extract_numeric_features(raw_text: str, clean: str) -> list:
    """Menghitung 6 fitur numerik tambahan, urutannya harus SAMA PERSIS
    dengan urutan saat training (lihat NUMERIC_COLS)."""
    words = clean.split()
    word_count = len(words)
    char_count = len(clean)
    avg_word_length = char_count / max(word_count, 1)
    upper_ratio = uppercase_ratio(raw_text)

    blob = TextBlob(clean)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    return [word_count, char_count, avg_word_length, upper_ratio, polarity, subjectivity]


def build_features(raw_text: str, artifacts: dict):
    """
    Parameters
    ----------
    raw_text : str
        Teks tweet mentah dari input pengguna.
    artifacts : dict
        Dict berisi objek-objek hasil training:
        {
            "tfidf": TfidfVectorizer,
            "scaler": StandardScaler,
            "selector": SelectKBest
        }

    Returns
    -------
    X : scipy.sparse matrix
        Matrix fitur final, siap di-.predict() oleh model.
    clean : str
        Teks yang sudah dibersihkan (untuk ditampilkan ke user jika perlu).
    """
    clean = clean_text(raw_text)

    tfidf = artifacts["tfidf"]
    scaler = artifacts["scaler"]
    selector = artifacts["selector"]

    # Jika hasil cleaning kosong, tetap proses dengan string kosong
    # supaya vectorizer tidak error dan tetap menghasilkan vector nol.
    X_text = tfidf.transform([clean])

    numeric_values = extract_numeric_features(raw_text, clean)
    X_num = scaler.transform([numeric_values])
    X_num = csr_matrix(X_num)

    X = hstack([X_text, X_num])
    X = selector.transform(X)

    return X, clean
