"""
app.py
======
Aplikasi Streamlit untuk deteksi indikasi depresi dari teks tweet,
menggunakan model terbaik (Stacking Ensemble) hasil penelitian.

Jalankan dengan:
    streamlit run app.py
"""

import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_engineering import build_features
from preprocessing import clean_text

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Serenity | Deteksi Indikasi Depresi",
    page_icon="✧",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "models"

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
# STYLING — "Serenity" design system
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600&family=Manrope:wght@300;400;500;600;700&display=swap');

        :root {
            --ink:        #14181f;
            --ink-soft:    #4a5262;
            --paper:      #f6f4ef;
            --paper-deep: #eeebe2;
            --deep-teal:  #1f3d3a;
            --deep-teal-2:#2c524d;
            --gold:       #b98d4f;
            --gold-soft:  #d9bd8e;
            --line:       rgba(20, 24, 31, 0.09);
            --shadow:     0 20px 60px -25px rgba(20, 40, 38, 0.35);
        }

        html, body, [class*="css"]  {
            font-family: 'Manrope', sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% -10%, rgba(31,61,58,0.10), transparent 45%),
                radial-gradient(circle at 100% 0%, rgba(185,141,79,0.10), transparent 40%),
                var(--paper);
        }

        h1, h2, h3, .display-font {
            font-family: 'Fraunces', serif;
            font-weight: 500;
            letter-spacing: -0.01em;
            color: var(--ink);
        }

        #MainMenu, footer, header {visibility: hidden;}

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        /* ---------- HERO ---------- */
        .hero-eyebrow {
            font-family: 'Manrope', sans-serif;
            font-size: 0.78rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--gold);
            font-weight: 700;
            margin-bottom: 0.9rem;
        }

        .hero-title {
            font-family: 'Fraunces', serif;
            font-size: 3.1rem;
            line-height: 1.08;
            font-weight: 500;
            color: var(--ink);
            margin-bottom: 0.9rem;
        }
        .hero-title em {
            font-style: italic;
            color: var(--deep-teal);
        }

        .hero-sub {
            font-size: 1.05rem;
            color: var(--ink-soft);
            max-width: 640px;
            line-height: 1.7;
            font-weight: 400;
        }

        .hero-divider {
            width: 64px;
            height: 2px;
            background: linear-gradient(90deg, var(--gold), transparent);
            margin: 1.6rem 0 1.6rem 0;
        }

        /* ---------- CARD ---------- */
        .glass-card {
            background: rgba(255,255,255,0.62);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 2.1rem 2.3rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(6px);
        }

        .section-label {
            font-size: 0.72rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: var(--gold);
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .section-title {
            font-family: 'Fraunces', serif;
            font-size: 1.5rem;
            font-weight: 500;
            margin-bottom: 0.3rem;
        }

        /* ---------- TEXTAREA ---------- */
        .stTextArea textarea {
            background: rgba(255,255,255,0.85) !important;
            border: 1.5px solid var(--line) !important;
            border-radius: 14px !important;
            font-family: 'Manrope', sans-serif !important;
            font-size: 1rem !important;
            color: var(--ink) !important;
            padding: 1rem 1.1rem !important;
        }
        .stTextArea textarea:focus {
            border-color: var(--deep-teal) !important;
            box-shadow: 0 0 0 3px rgba(31,61,58,0.12) !important;
        }

        /* ---------- BUTTON ---------- */
        .stButton > button {
            background: linear-gradient(135deg, var(--deep-teal), var(--deep-teal-2));
            color: #f6f4ef;
            border: none;
            border-radius: 999px;
            padding: 0.72rem 2.1rem;
            font-family: 'Manrope', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
            box-shadow: 0 12px 30px -10px rgba(31,61,58,0.55);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 36px -8px rgba(31,61,58,0.6);
            color: #f6f4ef;
        }

        /* ---------- RESULT BADGES ---------- */
        .result-card {
            border-radius: 22px;
            padding: 2rem 2.2rem;
            box-shadow: var(--shadow);
            border: 1px solid var(--line);
        }

        .result-card.risk {
            background: linear-gradient(160deg, #fdf3ef 0%, #f9e7e1 100%);
            border-color: rgba(178, 84, 54, 0.25);
        }
        .result-card.safe {
            background: linear-gradient(160deg, #f1f7f3 0%, #e6f0ea 100%);
            border-color: rgba(31, 61, 58, 0.2);
        }

        .result-label {
            font-family: 'Fraunces', serif;
            font-size: 1.7rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
        }
        .result-label.risk { color: #a3492c; }
        .result-label.safe { color: var(--deep-teal); }

        .result-meta { color: var(--ink-soft); font-size: 0.92rem; }

        .prob-bar-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(20,24,31,0.08);
            overflow: hidden;
            margin: 0.9rem 0 0.3rem 0;
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 999px;
        }
        .prob-bar-fill.risk { background: linear-gradient(90deg, #d97757, #a3492c); }
        .prob-bar-fill.safe { background: linear-gradient(90deg, var(--gold-soft), var(--deep-teal)); }

        /* ---------- SIDEBAR ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(185deg, var(--deep-teal) 0%, #16302d 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #eae6da !important;
        }
        section[data-testid="stSidebar"] .stMarkdown hr {
            border-color: rgba(234,230,218,0.18);
        }

        .sb-title {
            font-family: 'Fraunces', serif;
            font-size: 1.4rem;
            font-weight: 500;
            color: #f6f4ef !important;
            margin-bottom: 0.1rem;
        }
        .sb-caption {
            font-size: 0.82rem;
            color: rgba(234,230,218,0.65) !important;
            letter-spacing: 0.05em;
        }

        .metric-pill {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(234,230,218,0.14);
            border-radius: 14px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
        }
        .metric-pill .m-value {
            font-family: 'Fraunces', serif;
            font-size: 1.3rem;
            color: var(--gold-soft) !important;
        }
        .metric-pill .m-label {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(234,230,218,0.6) !important;
        }

        .footnote {
            font-size: 0.8rem;
            color: var(--ink-soft);
            line-height: 1.6;
        }

        .crisis-box {
            border-left: 3px solid #a3492c;
            background: rgba(163,73,44,0.06);
            padding: 1rem 1.2rem;
            border-radius: 10px;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        st.markdown('<div class="sb-title">✧ Serenity</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sb-caption">MENTAL WELLNESS TEXT ANALYTICS</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="footnote" style="color:rgba(234,230,218,0.75);">
            Aplikasi ini menganalisis pola linguistik pada teks singkat
            (gaya tweet) untuk memperkirakan kemungkinan indikasi depresi,
            menggunakan model <em>stacking ensemble</em> yang dilatih pada
            gabungan fitur TF‑IDF dan fitur linguistik.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="sb-caption">RINGKASAN MODEL</div>', unsafe_allow_html=True)

        if bundle is not None:
            model_name = type(bundle["model"]).__name__
            st.markdown(
                f"""
                <div class="metric-pill">
                    <div class="m-value">{model_name}</div>
                    <div class="m-label">Arsitektur Model</div>
                </div>
                <div class="metric-pill">
                    <div class="m-value">TF-IDF + Linguistik</div>
                    <div class="m-label">Sumber Fitur</div>
                </div>
                <div class="metric-pill">
                    <div class="m-value">SMOTE + SelectKBest</div>
                    <div class="m-label">Penanganan Imbalance & Seleksi Fitur</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Model belum dimuat. Lengkapi file di folder `models/`.")

        st.markdown("<br/><hr/>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="footnote" style="color:rgba(234,230,218,0.65);">
            <strong style="color:#eae6da;">Disclaimer</strong><br/>
            Alat ini bersifat eksploratif dan edukatif, <em>bukan</em>
            alat diagnosis klinis. Hasil prediksi tidak menggantikan
            konsultasi dengan tenaga profesional kesehatan mental.
            </div>
            """,
            unsafe_allow_html=True,
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
    st.markdown(
        '<div class="hero-title">Dengarkan apa yang<br/><em>tersirat</em> di balik kata.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Tempelkan sebuah tweet atau tulisan singkat di bawah ini. '
        'Model <em>stacking ensemble</em> kami — hasil kombinasi Logistic Regression, '
        'SVM, XGBoost, dan CatBoost — akan menelaah pola bahasa dan nada emosional '
        'untuk memperkirakan kemungkinan indikasi depresi.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    if missing:
        missing_items = "".join(f"<li><code>{ARTIFACT_PATHS[m]}</code></li>" for m in missing)
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-label">Konfigurasi Diperlukan</div>
                <div class="section-title">Beberapa file model belum ditemukan</div>
                <p class="footnote">
                Letakkan file berikut di dalam folder <code>models/</code> pada direktori
                aplikasi ini agar prediksi dapat berjalan:
                </p>
                <ul class="footnote">
                {missing_items}
                </ul>
                <p class="footnote">
                File-file ini dihasilkan dari notebook penelitian
                (<code>tfidf_vectorizer.pkl</code>, <code>numeric_scaler.pkl</code>,
                <code>feature_selector.pkl</code>, dan <code>best_model.pkl</code>).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ---------------- INPUT CARD ----------------
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.6rem;">'
            '<div style="width:34px; height:34px; border-radius:50%; background:var(--deep-teal); '
            'color:#f6f4ef; display:flex; align-items:center; justify-content:center; '
            'font-family:Fraunces,serif; font-weight:600; font-size:1rem; flex-shrink:0;">01</div>'
            '<div><div class="section-label" style="margin-bottom:0.1rem;">Langkah Pertama</div>'
            '<div class="section-title" style="margin-bottom:0;">Tulis atau tempel teks</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        example_prompt = st.selectbox(
            "Gunakan contoh (opsional)",
            [
                "— Pilih contoh —",
                "I feel hopeless and nobody loves me.",
                "Had such a wonderful day at the beach with my family!",
                "I can't get out of bed anymore, everything feels pointless.",
                "Just finished a great workout, feeling energized and happy.",
            ],
            label_visibility="collapsed",
        )

        default_text = "" if example_prompt == "— Pilih contoh —" else example_prompt

        user_text = st.text_area(
            "Teks",
            value=default_text,
            height=170,
            placeholder="Contoh: 'I feel so tired of pretending everything is okay...'",
            label_visibility="collapsed",
        )

        analyze_clicked = st.button("✧  Analisis Teks", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown(
            '<div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.6rem;">'
            '<div style="width:34px; height:34px; border-radius:50%; background:var(--gold); '
            'color:#14181f; display:flex; align-items:center; justify-content:center; '
            'font-family:Fraunces,serif; font-weight:600; font-size:1.1rem; flex-shrink:0;">✦</div>'
            '<div><div class="section-label" style="margin-bottom:0.1rem;">Bagaimana Cara Kerjanya</div>'
            '<div class="section-title" style="margin-bottom:0;">Alur Analisis</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        steps = [
            ("①", "Pembersihan teks", "Normalisasi, penghapusan URL/mention/emoji, stopword removal, dan lemmatization."),
            ("②", "Ekstraksi fitur", "TF-IDF (1–3 gram) digabung fitur linguistik: panjang teks, rasio huruf kapital, polaritas & subjektivitas sentimen."),
            ("③", "Seleksi fitur", "Pemilihan fitur paling informatif melalui mutual information."),
            ("④", "Prediksi", "Model stacking ensemble menghasilkan probabilitas indikasi depresi."),
        ]

        for num, title, desc in steps:
            st.markdown(
                f'<div style="display:flex; gap:0.8rem; margin-bottom:0.9rem;">'
                f'<div style="font-family:Fraunces,serif; font-size:1.1rem; color:var(--gold); '
                f'flex-shrink:0; width:1.4rem;">{num}</div>'
                f'<div><div style="font-weight:600; font-size:0.92rem; margin-bottom:0.15rem;">{title}</div>'
                f'<div class="footnote" style="margin:0;">{desc}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RESULT ----------------
    if analyze_clicked:
        if not user_text or not user_text.strip():
            st.warning("Mohon masukkan teks terlebih dahulu sebelum melakukan analisis.")
        else:
            with st.spinner("Menelaah pola bahasa..."):
                pred, prob, clean = predict(user_text, full_bundle)

            st.markdown("<br/>", unsafe_allow_html=True)
            is_risk = pred == 1
            css_class = "risk" if is_risk else "safe"
            label_text = "Indikasi Depresi Terdeteksi" if is_risk else "Tidak Terindikasi Depresi"
            icon = "◐" if is_risk else "✦"

            prob_pct = round(prob * 100, 1) if prob is not None else None

            # ---------------------------------------------------------
            # Kartu hasil: judul + deskripsi (HTML, styling penuh)
            # ---------------------------------------------------------
            st.markdown(
                f'<div class="result-card {css_class}">'
                f'<div class="result-label {css_class}">{icon}&nbsp;&nbsp;{label_text}</div>'
                f'<div class="result-meta">Berdasarkan analisis pola linguistik dan sentimen '
                f'pada teks yang diberikan.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ---------------------------------------------------------
            # Tingkat keyakinan: pakai st.progress bawaan Streamlit
            # (lebih stabil ketimbang bar HTML custom) + label metrik.
            # ---------------------------------------------------------
            if prob_pct is not None:
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                m_left, m_right = st.columns([0.75, 0.25])
                with m_left:
                    st.progress(min(max(prob / 1.0, 0.0), 1.0))
                with m_right:
                    st.markdown(
                        f"<div style='text-align:right; font-family:Fraunces,serif; "
                        f"font-size:1.15rem; font-weight:600; color:"
                        f"{'#a3492c' if is_risk else '#1f3d3a'};'>{prob_pct}%</div>",
                        unsafe_allow_html=True,
                    )
                st.caption("Tingkat keyakinan model terhadap prediksi di atas.")

            if is_risk and contains_crisis_language(user_text):
                st.markdown(
                    """
                    <div class="crisis-box footnote">
                    <strong>Catatan penting:</strong> Jika kamu atau seseorang yang kamu kenal
                    sedang dalam krisis atau memiliki pikiran untuk menyakiti diri sendiri,
                    segera hubungi layanan darurat setempat atau hotline kesehatan jiwa
                    (di Indonesia: <strong>119 ext. 8</strong>, Kemenkes). Kamu tidak sendirian.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("Lihat detail teks setelah preprocessing"):
                st.code(clean if clean else "(hasil pembersihan kosong)", language=None)

    st.markdown("<br/><br/>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="footnote" style="text-align:center; opacity:0.7;">
        Dibangun dengan Streamlit · Model Stacking Ensemble (LR + SVM + XGBoost + CatBoost)
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
