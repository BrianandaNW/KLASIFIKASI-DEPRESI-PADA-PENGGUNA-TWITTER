"""
app.py
======
Aplikasi Streamlit untuk deteksi indikasi depresi dari teks tweet,
menggunakan model terbaik (Stacking Ensemble) hasil penelitian.

Jalankan dengan:
    streamlit run app.py
"""

import os

import joblib
import streamlit as st

from feature_engineering import build_features

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Serenity | Deteksi Indikasi Depresi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Path dihitung relatif terhadap LOKASI FILE app.py ini (bukan terhadap
# current working directory). Ini penting karena di Streamlit Community
# Cloud, working directory selalu berada di ROOT repository meskipun
# app.py berada di dalam subfolder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

ARTIFACT_PATHS = {
    "tfidf": os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
    "scaler": os.path.join(MODEL_DIR, "numeric_scaler.pkl"),
    "selector": os.path.join(MODEL_DIR, "feature_selector.pkl"),
    "model": os.path.join(MODEL_DIR, "best_model.pkl"),
}

CRISIS_KEYWORDS = [
    "bunuh diri", "suicide", "ingin mati", "mengakhiri hidup",
    "kill myself", "end my life", "self harm", "menyakiti diri",
]


# ----------------------------------------------------------------------------
# STYLING — palet colorful, tetap elegan (gradient teal -> violet -> coral)
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --violet: #7c5cff;
    --violet-deep: #5a3fd6;
    --teal: #0fb8a3;
    --coral: #ff6f61;
    --gold: #f5a623;
    --ink: #1a1b2e;
    --ink-soft: #5a5c78;
    --paper: #faf9ff;
}

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }

.stApp {
    background:
        radial-gradient(circle at 8% -5%, rgba(124,92,255,0.14), transparent 40%),
        radial-gradient(circle at 95% 0%, rgba(15,184,163,0.13), transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(255,111,97,0.08), transparent 45%),
        var(--paper);
}

#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }

h1, h2, h3 { font-family: 'Fraunces', serif; color: var(--ink); }

/* Hero title gradient text */
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 3rem;
    line-height: 1.1;
    background: linear-gradient(100deg, var(--violet) 10%, var(--teal) 55%, var(--coral) 95%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.6rem;
}

.hero-eyebrow {
    font-size: 0.78rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--violet-deep); font-weight: 800; margin-bottom: 0.7rem;
}

.hero-sub { font-size: 1.05rem; color: var(--ink-soft); line-height: 1.7; max-width: 680px; }

/* Card container (native st.container border works, this styles it further) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 20px !important;
    box-shadow: 0 16px 40px -20px rgba(90,63,214,0.25);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(120deg, var(--violet), var(--violet-deep));
    color: white; border: none; border-radius: 999px;
    padding: 0.7rem 2rem; font-weight: 700; font-size: 0.95rem;
    box-shadow: 0 10px 24px -8px rgba(90,63,214,0.5);
    transition: transform 0.15s ease;
}
.stButton > button:hover { transform: translateY(-2px); color: white; }

/* Text area */
.stTextArea textarea {
    border-radius: 14px !important;
    border: 1.5px solid rgba(124,92,255,0.25) !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--violet) !important;
    box-shadow: 0 0 0 3px rgba(124,92,255,0.15) !important;
}

/* Badge chips for "how it works" */
.step-chip {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: white; border-radius: 14px; padding: 0.8rem 1rem;
    margin-bottom: 0.6rem; width: 100%;
    border: 1px solid rgba(26,27,46,0.06);
    box-shadow: 0 4px 14px -8px rgba(26,27,46,0.15);
}
.step-num {
    flex-shrink: 0; width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.85rem; color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(200deg, #241b45 0%, #1a1b2e 100%);
}
section[data-testid="stSidebar"] * { color: #ece9ff !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(236,233,255,0.15); }
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# LOAD ARTIFACTS (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    missing = [name for name, path in ARTIFACT_PATHS.items() if not os.path.exists(path)]
    if missing:
        return None, missing

    artifacts = {
        "tfidf": joblib.load(ARTIFACT_PATHS["tfidf"]),
        "scaler": joblib.load(ARTIFACT_PATHS["scaler"]),
        "selector": joblib.load(ARTIFACT_PATHS["selector"]),
    }
    model = joblib.load(ARTIFACT_PATHS["model"])
    return {"artifacts": artifacts, "model": model}, []


def contains_crisis_language(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


def predict(text: str, bundle: dict):
    X, clean = build_features(text, bundle["artifacts"])
    model = bundle["model"]

    pred = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
    else:
        prob = None

    return pred, prob, clean


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
def render_sidebar(bundle):
    with st.sidebar:
        st.markdown("### 🧠 Serenity")
        st.caption("MENTAL WELLNESS TEXT ANALYTICS")
        st.divider()

        st.write(
            "Aplikasi ini menganalisis pola linguistik pada teks singkat "
            "(gaya tweet) untuk memperkirakan kemungkinan indikasi depresi, "
            "menggunakan model *stacking ensemble* yang dilatih pada gabungan "
            "fitur TF-IDF dan fitur linguistik."
        )

        st.markdown("**RINGKASAN MODEL**")

        if bundle is not None:
            model_name = type(bundle["model"]).__name__
            st.metric("Arsitektur Model", model_name)
            st.metric("Sumber Fitur", "TF-IDF + Linguistik")
            st.metric("Imbalance & Seleksi", "SMOTE + SelectKBest")
        else:
            st.warning("Model belum dimuat. Lengkapi file di folder `models/`.")

        st.divider()
        st.markdown("**Disclaimer**")
        st.caption(
            "Alat ini bersifat eksploratif dan edukatif, bukan alat diagnosis "
            "klinis. Hasil prediksi tidak menggantikan konsultasi dengan "
            "tenaga profesional kesehatan mental."
        )


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    inject_css()

    bundle_result, missing = load_artifacts()
    bundle = bundle_result["artifacts"] if bundle_result else None
    full_bundle = bundle_result if bundle_result else None

    render_sidebar(full_bundle)

    # ---------------- HERO ----------------
    st.markdown('<div class="hero-eyebrow">Analisis Teks · Kesehatan Mental</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Dengarkan apa yang tersirat di balik kata</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Tempelkan sebuah tweet atau tulisan singkat di bawah ini. '
        'Model <b>stacking ensemble</b> kami — hasil kombinasi Logistic Regression, '
        'SVM, XGBoost, dan CatBoost — akan menelaah pola bahasa dan nada emosional '
        'untuk memperkirakan kemungkinan indikasi depresi.</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    if missing:
        st.warning("**Konfigurasi diperlukan** — beberapa file model belum ditemukan.")
        st.markdown(
            "Letakkan file berikut di dalam folder `models/` pada direktori aplikasi ini "
            "agar prediksi dapat berjalan:"
        )
        for m in missing:
            st.code(ARTIFACT_PATHS[m], language=None)
        st.caption(
            "File-file ini dihasilkan dari notebook penelitian "
            "(`tfidf_vectorizer.pkl`, `numeric_scaler.pkl`, `feature_selector.pkl`, "
            "dan `best_model.pkl`)."
        )
        return

    # ---------------- INPUT + INFO ----------------
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("##### 📝 Langkah 1 — Tulis atau tempel teks")

            example_prompt = st.selectbox(
                "Gunakan contoh (opsional)",
                [
                    "— Pilih contoh —",
                    "I feel hopeless and nobody loves me.",
                    "Had such a wonderful day at the beach with my family!",
                    "I can't get out of bed anymore, everything feels pointless.",
                    "Just finished a great workout, feeling energized and happy.",
                ],
            )

            default_text = "" if example_prompt == "— Pilih contoh —" else example_prompt

            user_text = st.text_area(
                "Teks yang akan dianalisis",
                value=default_text,
                height=170,
                placeholder="Contoh: 'I feel so tired of pretending everything is okay...'",
                label_visibility="collapsed",
            )

            analyze_clicked = st.button("✨ Analisis Teks")

    with right:
        with st.container(border=True):
            st.markdown("##### ⚙️ Alur Analisis")

            steps = [
                ("1", "#7c5cff", "Pembersihan Teks", "Normalisasi, hapus URL/mention/emoji, stopword removal & lemmatization."),
                ("2", "#0fb8a3", "Ekstraksi Fitur", "TF-IDF (1-3 gram) digabung fitur linguistik: panjang teks, huruf kapital, sentimen."),
                ("3", "#f5a623", "Seleksi Fitur", "Pemilihan fitur paling informatif via mutual information."),
                ("4", "#ff6f61", "Prediksi", "Model stacking ensemble menghasilkan probabilitas indikasi depresi."),
            ]

            for num, color, title, desc in steps:
                col_a, col_b = st.columns([0.12, 0.88])
                with col_a:
                    st.markdown(
                        f'<div class="step-num" style="background:{color};">{num}</div>',
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.markdown(f"**{title}**")
                    st.caption(desc)

    # ---------------- RESULT ----------------
    if analyze_clicked:
        if not user_text or not user_text.strip():
            st.warning("Mohon masukkan teks terlebih dahulu sebelum melakukan analisis.")
        else:
            with st.spinner("Menelaah pola bahasa..."):
                pred, prob, clean = predict(user_text, full_bundle)

            st.write("")
            is_risk = pred == 1
            prob_pct = round(prob * 100, 1) if prob is not None else None

            if is_risk:
                st.error("### 😔 Indikasi Depresi Terdeteksi")
            else:
                st.success("### 🌤️ Tidak Terindikasi Depresi")

            st.caption("Berdasarkan analisis pola linguistik dan sentimen pada teks yang diberikan.")

            if prob_pct is not None:
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    st.progress(prob_pct / 100)
                with col2:
                    st.metric("Tingkat Keyakinan Model", f"{prob_pct}%")

            if is_risk and contains_crisis_language(user_text):
                st.info(
                    "**Catatan penting:** Jika kamu atau seseorang yang kamu kenal sedang "
                    "dalam krisis atau memiliki pikiran untuk menyakiti diri sendiri, segera "
                    "hubungi layanan darurat setempat atau hotline kesehatan jiwa "
                    "(di Indonesia: **119 ext. 8**, Kemenkes). Kamu tidak sendirian."
                )

            with st.expander("🔍 Lihat detail teks setelah preprocessing"):
                st.code(clean if clean else "(hasil pembersihan kosong)", language=None)

    st.write("")
    st.write("")
    st.caption("Dibangun dengan Streamlit · Model Stacking Ensemble (LR + SVM + XGBoost + CatBoost)")


if __name__ == "__main__":
    main()
