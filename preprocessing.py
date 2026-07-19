"""
preprocessing.py
=================
Modul preprocessing teks yang REPLIKA PERSIS dari pipeline yang digunakan
saat training pada notebook (NotebookV1StackingModel90Acc.ipynb).

Tahapan (harus sama persis urutannya dengan training, jangan diubah):
1. Perbaikan contractions (opsional, jika library tersedia)
2. Lowercase
3. Hapus URL, mention (@user), tag HTML
4. Hapus karakter '#'
5. Hapus emoji (opsional, jika library tersedia)
6. Hapus angka
7. Normalisasi karakter berulang (misal "yaaaay" -> "yaay")
8. Hapus tanda baca
9. Rapikan whitespace
10. Stopword removal + lemmatization (NLTK)

Catatan penting:
Pada notebook asli, fungsi `predict_tweet` di sel terakhir memakai
`simple_clean()` (versi minimalis) untuk inference, PADAHAL model dilatih
menggunakan fitur hasil `clean_text()` (full preprocessing + stopword +
lemmatization). Ini adalah inkonsistensi/bug pada notebook riset.
Pada deployment ini, kita SELALU memakai `clean_text()` versi lengkap
agar konsisten dengan proses training dan menghasilkan prediksi yang akurat.
"""

import re
import string

import nltk

# Pastikan resource NLTK tersedia (aman dipanggil berulang kali, sudah ada cache)
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(
            f"corpora/{resource}"
            if resource != "stopwords"
            else f"corpora/stopwords"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import contractions
    HAS_CONTRACTIONS = True
except ImportError:
    HAS_CONTRACTIONS = False

try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = r"http\S+|www\S+"
MENTION_PATTERN = r"@\w+"
HTML_PATTERN = r"<.*?>"


def normalize_repeat(text: str) -> str:
    """Meredam karakter berulang 3x atau lebih menjadi 2x. Contoh: 'sooo' -> 'soo'."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def clean_text(text) -> str:
    """
    Pipeline pembersihan teks LENGKAP — identik dengan yang dipakai
    saat feature engineering pada notebook training.
    """
    text = str(text)

    if HAS_CONTRACTIONS:
        text = contractions.fix(text)

    text = text.lower()

    text = re.sub(URL_PATTERN, " ", text)
    text = re.sub(MENTION_PATTERN, " ", text)
    text = re.sub(HTML_PATTERN, " ", text)

    text = text.replace("#", " ")

    if HAS_EMOJI:
        text = emoji.replace_emoji(text, replace=" ")

    text = re.sub(r"\d+", " ", text)

    text = normalize_repeat(text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for word in text.split():
        if word not in STOP_WORDS:
            word = LEMMATIZER.lemmatize(word)
            tokens.append(word)

    return " ".join(tokens)


def uppercase_ratio(text: str) -> float:
    """Rasio huruf kapital terhadap total karakter pada teks ASLI (sebelum cleaning)."""
    text = str(text)
    if len(text) == 0:
        return 0.0
    return sum(c.isupper() for c in text) / len(text)
