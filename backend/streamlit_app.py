import streamlit as st
import time
import fitz
from docx import Document
import io
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Sumy imports
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from rouge_score import rouge_scorer

# ─── Optional Transformers ─────────────────────────────────
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ─── MapReduce Helper ──────────────────────────────────────
def chunk_text(text, max_words=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def mapreduce_summarize(text, summarize_fn, max_words=400):
    words = text.split()
    if len(words) <= max_words:
        return summarize_fn(text)
    chunks = chunk_text(text, max_words)
    chunk_summaries = []
    for chunk in chunks:
        summary = summarize_fn(chunk)
        chunk_summaries.append(summary)
    combined = " ".join(chunk_summaries)
    final_summary = summarize_fn(combined)
    return final_summary

# ─── Load Models Once ──────────────────────────────────────
@st.cache_resource
def load_t5():
    if not TRANSFORMERS_AVAILABLE:
        return None
    return pipeline("summarization", model="t5-small")

@st.cache_resource
def load_bart():
    if not TRANSFORMERS_AVAILABLE:
        return None
    return pipeline("summarization", model="facebook/bart-base")

@st.cache_resource
def load_distilbart():
    if not TRANSFORMERS_AVAILABLE:
        return None
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="+SumIt Up!",
    page_icon="🍓",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;700;800;900&display=swap');

* { font-family: 'Nunito', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #1a0533 0%, #2d1b69 50%, #1a0533 100%);
    background-size: 400% 400%;
    animation: gradientShift 8s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.title {
    font-family: 'Fredoka One', cursive !important;
    font-size: 3.5rem !important;
    color: #FFE44D !important;
    text-align: center;
    text-shadow: 4px 4px 0px #2d1b69, 0 0 30px rgba(255,228,77,0.5);
    letter-spacing: 0.04em;
    animation: titlePop 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
}

@keyframes titlePop {
    0% { transform: scale(0.5); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

.subtitle {
    text-align: center;
    color: #ffd6f5;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 0.03em;
}

.card {
    background: rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px;
    border: 2px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}

.card-tfidf { border-top: 4px solid #FF9DE2; }
.card-textrank { border-top: 4px solid #82CFFF; }
.card-lexrank { border-top: 4px solid #FFD93D; }
.card-sumbasic { border-top: 4px solid #FF9F43; }
.card-t5 { border-top: 4px solid #A8F5A0; }
.card-bart { border-top: 4px solid #C77DFF; }
.card-distilbart { border-top: 4px solid #FF6B6B; }

.algo-title {
    font-family: 'Fredoka One', cursive;
    font-size: 1.1rem;
    color: #FFE44D;
    margin-bottom: 10px;
    letter-spacing: 0.04em;
}

.section-header {
    font-family: 'Fredoka One', cursive;
    font-size: 1.6rem;
    color: #FFE44D;
    margin: 20px 0 10px 0;
    text-shadow: 2px 2px 0px rgba(0,0,0,0.3);
}

.metric-box {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 10px 6px;
    text-align: center;
    margin: 4px;
    backdrop-filter: blur(5px);
    transition: transform 0.2s ease;
}

.metric-box:hover {
    transform: scale(1.05);
    background: rgba(255,255,255,0.12);
}

.metric-value {
    font-family: 'Fredoka One', cursive;
    font-size: 1.4rem;
    color: #FFE44D;
}

.metric-label {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

.neural-unavailable {
    background: rgba(255,107,107,0.1);
    border: 2px dashed rgba(255,107,107,0.4);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    color: rgba(255,255,255,0.6);
    font-size: 13px;
}

hr { border-color: rgba(255,255,255,0.1) !important; }

.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 2px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: white !important;
}

.stCheckbox label {
    color: rgba(255,255,255,0.8) !important;
    font-weight: 600 !important;
}

.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 50px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}

.badge-extractive {
    background: rgba(130, 207, 255, 0.2);
    color: #82CFFF;
    border: 1px solid #82CFFF;
}

.badge-abstractive {
    background: rgba(199, 125, 255, 0.2);
    color: #C77DFF;
    border: 1px solid #C77DFF;
}
</style>
""", unsafe_allow_html=True)

# ─── File Extraction ───────────────────────────────────────
def extract_text_from_pdf(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text.strip()

def extract_text_from_docx(file):
    doc = Document(io.BytesIO(file.read()))
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text.strip()

# ─── Extractive Algorithms ─────────────────────────────────
def _sumy_summarize(text, SummarizerClass, sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = SummarizerClass(stemmer)
    summarizer.stop_words = get_stop_words("english")
    result = summarizer(parser.document, sentences)
    return " ".join(str(s) for s in result)

def summarize_tfidf(text):
    start = time.time()
    fn = lambda t: _sumy_summarize(t, LsaSummarizer)
    summary = mapreduce_summarize(text, fn)
    return summary, round((time.time() - start) * 1000)

def summarize_textrank(text):
    start = time.time()
    fn = lambda t: _sumy_summarize(t, TextRankSummarizer)
    summary = mapreduce_summarize(text, fn)
    return summary, round((time.time() - start) * 1000)

def summarize_lexrank(text):
    start = time.time()
    fn = lambda t: _sumy_summarize(t, LexRankSummarizer)
    summary = mapreduce_summarize(text, fn)
    return summary, round((time.time() - start) * 1000)

def summarize_sumbasic(text):
    start = time.time()
    try:
        fn = lambda t: _sumy_summarize(t, SumBasicSummarizer)
        summary = mapreduce_summarize(text, fn)
        if not summary.strip():
            raise ValueError("Empty summary")
    except Exception:
        fn = lambda t: _sumy_summarize(t, LsaSummarizer)
        summary = mapreduce_summarize(text, fn)
    return summary, round((time.time() - start) * 1000)

# ─── Abstractive Algorithms ────────────────────────────────
def _transformer_summarize(model_fn, text):
    model = model_fn()
    if model is None:
        return None
    words = text.split()
    max_len = min(130, max(30, len(words) // 2))
    min_len = min(30, max_len - 10)
    result = model(text[:1024], max_length=max_len, min_length=min_len, do_sample=False)
    return result[0]["summary_text"]

def summarize_t5(text):
    if not TRANSFORMERS_AVAILABLE:
        return "⚠️ Neural models unavailable on Streamlit Cloud. Run locally to use T5-Small.", 0
    start = time.time()
    fn = lambda t: _transformer_summarize(load_t5, t)
    summary = mapreduce_summarize(text, fn, max_words=200)
    return summary, round((time.time() - start) * 1000)

def summarize_bart(text):
    if not TRANSFORMERS_AVAILABLE:
        return "⚠️ Neural models unavailable on Streamlit Cloud. Run locally to use BART-Base.", 0
    start = time.time()
    fn = lambda t: _transformer_summarize(load_bart, t)
    summary = mapreduce_summarize(text, fn, max_words=300)
    return summary, round((time.time() - start) * 1000)

def summarize_distilbart(text):
    if not TRANSFORMERS_AVAILABLE:
        return "⚠️ Neural models unavailable on Streamlit Cloud. Run locally to use DistilBART.", 0
    start = time.time()
    fn = lambda t: _transformer_summarize(load_distilbart, t)
    summary = mapreduce_summarize(text, fn, max_words=300)
    return summary, round((time.time() - start) * 1000)

# ─── ROUGE ─────────────────────────────────────────────────
def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {
        'rouge1': round(scores['rouge1'].fmeasure, 3),
        'rouge2': round(scores['rouge2'].fmeasure, 3),
        'rougeL': round(scores['rougeL'].fmeasure, 3)
    }

# ─── Result Card ───────────────────────────────────────────
def show_result_card(title, summary, time_ms, word_count, rouge=None, card_class="card"):
    is_unavailable = summary.startswith("⚠️")
    if is_unavailable:
        st.markdown(f"""
        <div class="neural-unavailable">
            <div class="algo-title">{title}</div>
            <p>{summary}</p>
            <small>Install <code>transformers</code> and <code>torch</code> to use neural models locally.</small>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="card {card_class}">
        <div class="algo-title">{title}</div>
        <div style="
            height: 120px;
            overflow: hidden;
            position: relative;
            font-size:13px;
            line-height:1.8;
            color:rgba(255,255,255,0.8);">
            {summary}
            <div style="
                position: absolute;
                bottom: 0; left: 0; right: 0;
                height: 40px;
                background: linear-gradient(transparent, rgba(26,5,51,0.9));">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Read full summary"):
        st.write(summary)

    cols = st.columns(4 if rouge else 2)
    with cols[0]:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{time_ms}ms</div><div class="metric-label">Time Taken</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{word_count}</div><div class="metric-label">Word Count</div></div>', unsafe_allow_html=True)
    if rouge:
        with cols[2]:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{rouge["rouge1"]}</div><div class="metric-label">ROUGE-1</div></div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{rouge["rougeL"]}</div><div class="metric-label">ROUGE-L</div></div>', unsafe_allow_html=True)

# ─── Main App ──────────────────────────────────────────────
st.markdown('<p class="title">🍓 +SumIt Up!</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Compare 7 summarization algorithms side by side 🚀</p>', unsafe_allow_html=True)

if not TRANSFORMERS_AVAILABLE:
    st.info("ℹ️ Running in cloud mode — extractive algorithms only. Neural models (T5, BART, DistilBART) require a local install.", icon="☁️")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📄 Input Text")
    uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
    input_text = st.text_area("Or paste your text directly", height=200, placeholder="Paste your text here...")
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            extracted = extract_text_from_pdf(uploaded_file)
        else:
            extracted = extract_text_from_docx(uploaded_file)
        if extracted:
            input_text = extracted
            st.success(f"✅ Text extracted! ({len(extracted.split())} words)")
            with st.expander("Preview extracted text"):
                st.write(extracted[:1000] + "..." if len(extracted) > 1000 else extracted)

with col2:
    st.markdown("### ⚙️ Settings")
    reference_text = st.text_area("Reference Summary (optional — for ROUGE)", height=120, placeholder="Paste a reference summary...")
    st.markdown("### 🔧 Select Algorithms")
    st.markdown("**Extractive**")
    use_tfidf = st.checkbox("TF-IDF (LSA)", value=True)
    use_textrank = st.checkbox("TextRank", value=True)
    use_lexrank = st.checkbox("LexRank", value=True)
    use_sumbasic = st.checkbox("SumBasic", value=True)
    st.markdown("**Abstractive** ☁️ local only")
    use_t5 = st.checkbox("T5-Small", value=TRANSFORMERS_AVAILABLE)
    use_bart = st.checkbox("BART-Base", value=TRANSFORMERS_AVAILABLE)
    use_distilbart = st.checkbox("DistilBART", value=TRANSFORMERS_AVAILABLE)

st.divider()

if st.button("▶ Run All Algorithms", type="primary", use_container_width=True):
    if not input_text.strip():
        st.error("Please enter some text or upload a file first!")
    else:
        # ── Extractive Results ──
        st.markdown('<p class="section-header">⊡ Extractive Algorithms <span class="badge badge-extractive">RULE BASED</span></p>', unsafe_allow_html=True)
        ext_cols = st.columns(4)
        extractive_results = []

        if use_tfidf:
            with ext_cols[0]:
                with st.spinner("TF-IDF..."):
                    s, t = summarize_tfidf(input_text)
                    r = calculate_rouge(reference_text, s) if reference_text.strip() else None
                    show_result_card("⊡ TF-IDF (LSA)", s, t, len(s.split()), r, "card-tfidf")
                    extractive_results.append({"Algorithm": "TF-IDF (LSA)", "Time (ms)": t, "Words": len(s.split()), **(r or {})})

        if use_textrank:
            with ext_cols[1]:
                with st.spinner("TextRank..."):
                    s, t = summarize_textrank(input_text)
                    r = calculate_rouge(reference_text, s) if reference_text.strip() else None
                    show_result_card("◈ TextRank", s, t, len(s.split()), r, "card-textrank")
                    extractive_results.append({"Algorithm": "TextRank", "Time (ms)": t, "Words": len(s.split()), **(r or {})})

        if use_lexrank:
            with ext_cols[2]:
                with st.spinner("LexRank..."):
                    s, t = summarize_lexrank(input_text)
                    r = calculate_rouge(reference_text, s) if reference_text.strip() else None
                    show_result_card("◇ LexRank", s, t, len(s.split()), r, "card-lexrank")
                    extractive_results.append({"Algorithm": "LexRank", "Time (ms)": t, "Words": len(s.split()), **(r or {})})

        if use_sumbasic:
            with ext_cols[3]:
                with st.spinner("SumBasic..."):
                    s, t = summarize_sumbasic(input_text)
                    r = calculate_rouge(reference_text, s) if reference_text.strip() else None
                    show_result_card("≡ SumBasic", s, t, len(s.split()), r, "card-sumbasic")
                    extractive_results.append({"Algorithm": "SumBasic", "Time (ms)": t, "Words": len(s.split()), **(r or {})})

        # ── Abstractive Results ──
        st.markdown('<p class="section-header">◈ Abstractive Algorithms <span class="badge badge-abstractive">NEURAL</span></p>', unsafe_allow_html=True)
        abs_cols = st.columns(3)
        abstractive_results = []

        if use_t5:
            with abs_cols[0]:
                with st.spinner("T5-Small..."):
                    s, t = summarize_t5(input_text)
                    r = calculate_rouge(reference_text, s) if reference_text.strip() and not s.startswith("⚠️") else None
                    show_result_card("T5-Small", s, t, len(s.split()), r, "card-t5")
                    if not s.startswith("⚠️"):
                        abstractive_results.append({"Algorithm": "T5-Small", "Time (ms)": t, "Words": len(s.split()), **(r or {})})

        if use_bart:
            with abs_cols[1]:
                with st.spinner("BART-Base..."):
                    s, t = summarize_bart(input_text)
                    r = calculate_rouge(reference_text, s) if reference_text.strip() and not s.startswith("⚠️") else None
                    show_result_card("BART-Base", s, t, len(s.split()), r, "card-bart")
                    if not s.startswith("⚠️"):
                        abstractive_results.append({"Algorithm": "BART-Base", "Time (ms)": t, "Words": len(s.split()), **(r or {})})

        if use_distilbart:
            with abs_cols[2]:
                with st.spinner("DistilBART..."):
                    s, t = summarize_distilbart(input_text)
                    r = calculate_rouge(reference_text, s) if reference_text.strip() and not s.startswith("⚠️") else None
                    show_result_card("DistilBART", s, t, len(s.split()), r, "card-distilbart")
                    if not s.startswith("⚠️"):
                        abstractive_results.append({"Algorithm": "DistilBART", "Time (ms)": t, "Words": len(s.split()), **(r or {})})

        # ── Comparison Table ──
        st.divider()
        st.markdown("### 📈 Full Comparison")
        all_results = extractive_results + abstractive_results
        if all_results:
            st.dataframe(all_results, use_container_width=True)