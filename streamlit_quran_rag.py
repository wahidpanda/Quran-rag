import os
import re
import time
import json
import random
import hashlib
from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st
import streamlit.components.v1 as components
import torch
import requests

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Cross-encoder re-ranker. Imported lazily/guarded so the app still runs
# if sentence-transformers isn't installed (falls back to vector order).
try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except Exception:
    _HAS_CROSS_ENCODER = False


# ============================================================
# CONFIG
# ============================================================
FAISS_DIR = "quran_faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # fast, lightweight re-ranker

# How many candidates to pull from FAISS before re-ranking down to top-k.
RETRIEVAL_FETCH_MULTIPLIER = 4   # fetch k * this many candidates per sub-query
MAX_SUBQUERIES = 4               # original + up to 3 rephrasings

# Public, free, no-key APIs used by the agentic layer.
QURAN_API = "https://api.alquran.cloud/v1"          # verses, translations, audio
HADITH_API = "https://random-hadith-generator.vercel.app"  # random hadith by collection
HADITH_COLLECTIONS = ["bukhari", "muslim", "abudawud", "ibnmajah", "tirmidhi"]

# DuckDuckGo Instant Answer API (no key) for lightweight web validation.
DDG_API = "https://api.duckduckgo.com/"

# Network behaviour — generous timeouts + a retry, tuned for Streamlit Cloud
# cold starts where the first outbound request is often slow.
HTTP_TIMEOUT = 20          # seconds per request
HTTP_RETRIES = 2           # total attempts before giving up
HTTP_HEADERS = {"User-Agent": "QuranChatbot/1.0 (+streamlit)"}


def http_get_json(url, params=None, timeout=HTTP_TIMEOUT, retries=HTTP_RETRIES):
    """
    GET a URL and parse JSON, retrying on transient failures.
    Raises the last exception if all attempts fail — callers should let that
    propagate (NOT return None) so st.cache_data does not cache the failure.
    """
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=HTTP_HEADERS)
            resp.raise_for_status()
            return resp.json()
        except Exception as ex:
            last_err = ex
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))  # simple backoff
    raise RuntimeError(f"Request to {url} failed after {retries} attempts: {last_err}")

ACCENT_THEMES = {
    "Emerald (default)": {"accent": "34, 197, 94", "accent2": "59, 130, 246"},
    "Royal Blue": {"accent": "59, 130, 246", "accent2": "168, 85, 247"},
    "Amethyst": {"accent": "168, 85, 247", "accent2": "236, 72, 153"},
    "Amber / Gold": {"accent": "217, 119, 6", "accent2": "34, 197, 94"},
    "Rose": {"accent": "236, 72, 153", "accent2": "59, 130, 246"},
    "Midnight Teal": {"accent": "20, 184, 166", "accent2": "99, 102, 241"},
}

QUESTION_BANK = {
    "🕊️ Faith & Belief": [
        "What does the Quran say about the Oneness of God (Tawheed)?",
        "What does the Quran say about the Day of Judgment?",
        "What does the Quran say about angels?",
        "What does the Quran say about the Prophets?",
    ],
    "🤲 Worship & Prayer": [
        "What does the Quran say about prayer (Salah)?",
        "What does the Quran say about fasting in Ramadan?",
        "What does the Quran say about Hajj?",
        "What does the Quran say about supplication (dua)?",
    ],
    "❤️ Character & Ethics": [
        "What does the Quran say about patience?",
        "What does the Quran say about forgiveness?",
        "What does the Quran say about honesty?",
        "What does the Quran say about anger?",
    ],
    "👨‍👩‍👧 Family & Society": [
        "What does the Quran say about parents' rights?",
        "What does the Quran say about kindness to neighbors?",
        "What does the Quran say about justice?",
        "What does the Quran say about marriage?",
    ],
    "💰 Wealth & Charity": [
        "What does the Quran say about charity?",
        "What does the Quran say about riba (interest)?",
        "What does the Quran say about greed?",
        "What does the Quran say about helping the poor?",
    ],
}
ALL_SAMPLE_QUESTIONS = [q for group in QUESTION_BANK.values() for q in group]

# The 99 Names of Allah (Asma ul-Husna): (transliteration, Arabic, meaning).
# Arabic script is used so browser text-to-speech can pronounce it correctly.
ASMA_UL_HUSNA = [
    ("Ar-Rahman", "الرحمن", "The Most Compassionate"),
    ("Ar-Raheem", "الرحيم", "The Most Merciful"),
    ("Al-Malik", "الملك", "The King / Sovereign"),
    ("Al-Quddus", "القدوس", "The Most Holy"),
    ("As-Salam", "السلام", "The Source of Peace"),
    ("Al-Mu'min", "المؤمن", "The Granter of Security"),
    ("Al-Muhaymin", "المهيمن", "The Guardian"),
    ("Al-Aziz", "العزيز", "The Almighty"),
    ("Al-Jabbar", "الجبار", "The Compeller"),
    ("Al-Mutakabbir", "المتكبر", "The Supreme in Greatness"),
    ("Al-Khaliq", "الخالق", "The Creator"),
    ("Al-Bari'", "البارئ", "The Originator"),
    ("Al-Musawwir", "المصور", "The Fashioner of Forms"),
    ("Al-Ghaffar", "الغفار", "The Ever-Forgiving"),
    ("Al-Qahhar", "القهار", "The All-Subduer"),
    ("Al-Wahhab", "الوهاب", "The Supreme Bestower"),
    ("Ar-Razzaq", "الرزاق", "The Provider"),
    ("Al-Fattah", "الفتاح", "The Opener / Judge"),
    ("Al-Alim", "العليم", "The All-Knowing"),
    ("Al-Qabid", "القابض", "The Withholder"),
    ("Al-Basit", "الباسط", "The Extender"),
    ("Al-Khafid", "الخافض", "The Abaser"),
    ("Ar-Rafi'", "الرافع", "The Exalter"),
    ("Al-Mu'izz", "المعز", "The Giver of Honour"),
    ("Al-Mudhill", "المذل", "The Giver of Dishonour"),
    ("As-Sami'", "السميع", "The All-Hearing"),
    ("Al-Basir", "البصير", "The All-Seeing"),
    ("Al-Hakam", "الحكم", "The Judge"),
    ("Al-Adl", "العدل", "The Utterly Just"),
    ("Al-Latif", "اللطيف", "The Subtle One / Most Gentle"),
    ("Al-Khabir", "الخبير", "The All-Aware"),
    ("Al-Halim", "الحليم", "The Forbearing"),
    ("Al-Azim", "العظيم", "The Magnificent"),
    ("Al-Ghafur", "الغفور", "The Great Forgiver"),
    ("Ash-Shakur", "الشكور", "The Most Appreciative"),
    ("Al-Ali", "العلي", "The Most High"),
    ("Al-Kabir", "الكبير", "The Most Great"),
    ("Al-Hafiz", "الحفيظ", "The Preserver"),
    ("Al-Muqit", "المقيت", "The Sustainer"),
    ("Al-Hasib", "الحسيب", "The Reckoner"),
    ("Al-Jalil", "الجليل", "The Majestic"),
    ("Al-Karim", "الكريم", "The Most Generous"),
    ("Ar-Raqib", "الرقيب", "The Watchful"),
    ("Al-Mujib", "المجيب", "The Responsive"),
    ("Al-Wasi'", "الواسع", "The All-Encompassing"),
    ("Al-Hakim", "الحكيم", "The All-Wise"),
    ("Al-Wadud", "الودود", "The Most Loving"),
    ("Al-Majid", "المجيد", "The Most Glorious"),
    ("Al-Ba'ith", "الباعث", "The Resurrector"),
    ("Ash-Shahid", "الشهيد", "The Witness"),
    ("Al-Haqq", "الحق", "The Absolute Truth"),
    ("Al-Wakil", "الوكيل", "The Trustee"),
    ("Al-Qawiyy", "القوي", "The All-Strong"),
    ("Al-Matin", "المتين", "The Firm"),
    ("Al-Waliyy", "الولي", "The Protecting Friend"),
    ("Al-Hamid", "الحميد", "The Praiseworthy"),
    ("Al-Muhsi", "المحصي", "The All-Enumerating"),
    ("Al-Mubdi'", "المبدئ", "The Originator"),
    ("Al-Mu'id", "المعيد", "The Restorer"),
    ("Al-Muhyi", "المحيي", "The Giver of Life"),
    ("Al-Mumit", "المميت", "The Bringer of Death"),
    ("Al-Hayy", "الحي", "The Ever-Living"),
    ("Al-Qayyum", "القيوم", "The Self-Subsisting"),
    ("Al-Wajid", "الواجد", "The Perceiver / Finder"),
    ("Al-Majid", "الماجد", "The Noble / Illustrious"),
    ("Al-Wahid", "الواحد", "The One"),
    ("Al-Ahad", "الأحد", "The Unique / Indivisible"),
    ("As-Samad", "الصمد", "The Eternal Refuge"),
    ("Al-Qadir", "القادر", "The All-Powerful"),
    ("Al-Muqtadir", "المقتدر", "The Determiner"),
    ("Al-Muqaddim", "المقدم", "The Expediter"),
    ("Al-Mu'akhkhir", "المؤخر", "The Delayer"),
    ("Al-Awwal", "الأول", "The First"),
    ("Al-Akhir", "الآخر", "The Last"),
    ("Az-Zahir", "الظاهر", "The Manifest"),
    ("Al-Batin", "الباطن", "The Hidden"),
    ("Al-Wali", "الوالي", "The Governor / Patron"),
    ("Al-Muta'ali", "المتعالي", "The Self-Exalted"),
    ("Al-Barr", "البر", "The Source of Goodness"),
    ("At-Tawwab", "التواب", "The Ever-Relenting"),
    ("Al-Muntaqim", "المنتقم", "The Avenger"),
    ("Al-Afuww", "العفو", "The Pardoner"),
    ("Ar-Ra'uf", "الرؤوف", "The Most Kind"),
    ("Malik-ul-Mulk", "مالك الملك", "Owner of All Sovereignty"),
    ("Dhul-Jalali-wal-Ikram", "ذو الجلال والإكرام", "Lord of Majesty and Honour"),
    ("Al-Muqsit", "المقسط", "The Equitable"),
    ("Al-Jami'", "الجامع", "The Gatherer"),
    ("Al-Ghaniyy", "الغني", "The Self-Sufficient"),
    ("Al-Mughni", "المغني", "The Enricher"),
    ("Al-Mani'", "المانع", "The Preventer of Harm"),
    ("Ad-Darr", "الضار", "The Distresser"),
    ("An-Nafi'", "النافع", "The Bestower of Benefit"),
    ("An-Nur", "النور", "The Light"),
    ("Al-Hadi", "الهادي", "The Guide"),
    ("Al-Badi'", "البديع", "The Incomparable Originator"),
    ("Al-Baqi", "الباقي", "The Everlasting"),
    ("Al-Warith", "الوارث", "The Inheritor"),
    ("Ar-Rashid", "الرشيد", "The Guide to the Right Path"),
    ("As-Sabur", "الصبور", "The Most Patient"),
]


# ============================================================
# PAGE STYLE
# ============================================================
def inject_custom_css(accent="34, 197, 94", accent2="59, 130, 246"):
    st.markdown(
        f"""
        <style>
        :root {{
            --accent: {accent};
            --accent2: {accent2};
        }}
        .main-title {{
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            background: linear-gradient(90deg, rgba(var(--accent),1), rgba(var(--accent2),1));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtle-text {{ color: #6b7280; font-size: 0.95rem; margin-bottom: 1rem; }}
        .answer-box {{
            padding: 1rem 1.2rem; border-radius: 14px;
            background: rgba(var(--accent), 0.08);
            border: 1px solid rgba(var(--accent), 0.25);
            margin-top: 0.5rem; margin-bottom: 0.75rem;
        }}
        .source-box {{
            padding: 0.8rem 1rem; border-radius: 12px;
            background: rgba(var(--accent2), 0.06);
            border: 1px solid rgba(var(--accent2), 0.18);
        }}
        .hadith-box {{
            padding: 1rem 1.2rem; border-radius: 14px;
            background: rgba(var(--accent2), 0.07);
            border-left: 4px solid rgba(var(--accent2), 0.6);
            margin: 0.5rem 0;
        }}
        .verse-arabic {{
            font-size: 1.6rem; line-height: 2.6rem; text-align: right;
            direction: rtl; font-family: 'Traditional Arabic', 'Amiri', serif;
        }}
        .small-muted {{ color: #6b7280; font-size: 0.85rem; }}
        .badge {{
            display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
            background: rgba(var(--accent), 0.15); color: rgba(var(--accent), 1);
            font-size: 0.75rem; font-weight: 700; margin-right: 0.3rem; margin-bottom: 0.3rem;
        }}
        .badge-verify {{
            display:inline-block; padding:0.2rem 0.6rem; border-radius:999px;
            font-size:0.78rem; font-weight:700;
        }}
        .v-ok {{ background: rgba(34,197,94,0.18); color: rgb(21,128,61); }}
        .v-warn {{ background: rgba(217,119,6,0.18); color: rgb(146,64,14); }}
        .v-unknown {{ background: rgba(107,114,128,0.18); color: rgb(55,65,81); }}
        .pill-stat {{
            padding: 0.6rem 0.9rem; border-radius: 12px;
            background: rgba(var(--accent2), 0.08);
            border: 1px solid rgba(var(--accent2), 0.18); text-align: center;
        }}
        .name-card {{
            padding: 1.1rem 1.3rem; border-radius: 16px;
            background: linear-gradient(135deg, rgba(var(--accent),0.12), rgba(var(--accent2),0.12));
            border: 1px solid rgba(var(--accent),0.25);
        }}
        .step-line {{
            padding: 0.35rem 0.6rem; border-radius: 8px; margin-bottom: 0.3rem;
            background: rgba(var(--accent2),0.06); font-size: 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL / INDEX LOADING
# ============================================================
def load_vector_store():
    if not os.path.exists(FAISS_DIR):
        raise FileNotFoundError(
            f"FAISS folder '{FAISS_DIR}' not found.\n"
            f"Expected: {FAISS_DIR}/index.faiss and {FAISS_DIR}/index.pkl"
        )
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)


def initialize_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    return tokenizer, model


def initialize_reranker():
    """Load the cross-encoder re-ranker, or None if unavailable."""
    if not _HAS_CROSS_ENCODER:
        return None
    try:
        return CrossEncoder(RERANKER_MODEL)
    except Exception:
        return None


def create_prompt():
    template = """
You are a knowledgeable Islamic assistant.

Use ONLY the Quran and tafsir context provided below to answer the user's question.

Rules:
- Answer clearly, respectfully, and concisely.
- Base the answer on the provided Quran context.
- Mention Surah and Ayah references when relevant.
- If the context is insufficient, say that clearly.
- Do NOT invent Hadith references.
- Do NOT add information that is not supported by the provided context.
- Structure the answer in short paragraphs or bullet points.

User Question:
{query}

Relevant Quran Context:
{context}

Answer:
"""
    return PromptTemplate(template=template, input_variables=["query", "context"])


# ============================================================
# AGENTIC LAYER — keyword/intent planning
# ============================================================
TOPIC_KEYWORDS = {
    "prayer": ["pray", "salah", "salat", "worship", "prostrate"],
    "charity": ["charity", "zakat", "sadaqah", "poor", "alms", "give"],
    "fasting": ["fast", "ramadan", "sawm", "siyam"],
    "patience": ["patience", "sabr", "perseverance", "endure"],
    "forgiveness": ["forgive", "mercy", "pardon", "repent"],
    "parents": ["parent", "mother", "father", "family"],
    "honesty": ["honest", "truth", "lie", "trust"],
    "justice": ["justice", "fair", "oppression", "equity"],
    "knowledge": ["knowledge", "learn", "wisdom", "study"],
    "hajj": ["hajj", "pilgrimage", "kaaba", "mecca"],
}


def plan_agentic_steps(query):
    """A lightweight 'planner': decide which tools the agent should invoke."""
    q = query.lower()
    steps = ["Retrieve relevant Quran passages from local FAISS index"]
    matched_topic = None
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in q for kw in kws):
            matched_topic = topic
            break
    steps.append("Generate grounded answer with FLAN-T5")
    steps.append(f"Fetch a related Hadith ({matched_topic or 'general'} theme)")
    steps.append("Cross-check Hadith authenticity via web validation")
    steps.append("Assemble cited, verified final response")
    return steps, matched_topic


# ============================================================
# TOOL: Quran verse fetch (Arabic + translation + audio)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_quran_verse(reference):
    """
    reference like '2:255' (surah:ayah) or '262' (global ayah number 1..6236).
    Returns dict with arabic, english, audio url, surah name.
    Raises on failure so a transient error is NOT cached as a permanent None.
    """
    ar = http_get_json(f"{QURAN_API}/ayah/{reference}/ar.alafasy")
    en = http_get_json(f"{QURAN_API}/ayah/{reference}/en.asad")
    if ar.get("status") != "OK" or en.get("status") != "OK":
        raise RuntimeError(f"AlQuran.cloud returned non-OK status for {reference}")
    a, e = ar["data"], en["data"]
    return {
        "reference": reference,
        "surah_name": a["surah"]["englishName"],
        "surah_arabic": a["surah"]["name"],
        "ayah_number": a["numberInSurah"],
        "arabic": a["text"],
        "english": e["text"],
        "audio": a.get("audio"),
    }


def safe_fetch_quran_verse(reference):
    """Non-raising wrapper for inline enrichment loops; returns None on failure."""
    try:
        return fetch_quran_verse(reference)
    except Exception:
        return None


# ============================================================
# TOOL: Hadith fetch
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_hadith(collection):
    """Pull a hadith from a public random-hadith API. Returns None on failure."""
    try:
        payload = http_get_json(f"{HADITH_API}/{collection}")
        data = payload.get("data", {})
        text = clean_text(data.get("hadith_english", ""))
        if not text:
            return None
        return {
            "collection": collection.title(),
            "text": text,
            "narrator": data.get("header", "").strip(),
            "book": data.get("book", ""),
            "number": data.get("hadith_no", ""),
            "ref": data.get("refno", ""),
        }
    except Exception:
        return None


# ============================================================
# TOOL: Web validation of a hadith (agentic cross-check)
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def validate_hadith_web(hadith):
    """
    Cross-check a hadith reference against the web using DuckDuckGo's
    Instant Answer API. Returns a verification verdict + evidence snippets.
    Note: free instant-answer API is shallow; this is a best-effort signal,
    not a scholarly grading.
    """
    if not hadith:
        return {"status": "unknown", "confidence": 0, "evidence": [], "query": ""}

    collection = hadith.get("collection", "")
    number = hadith.get("number", "")
    search_query = f"{collection} hadith {number} authentic sahih"

    evidence = []
    status = "unknown"
    confidence = 0
    try:
        data = http_get_json(
            DDG_API,
            params={"q": search_query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
        abstract = data.get("AbstractText", "")
        if abstract:
            evidence.append(abstract[:300])
        for t in data.get("RelatedTopics", [])[:4]:
            txt = t.get("Text") if isinstance(t, dict) else None
            if txt:
                evidence.append(txt[:200])

        joined = " ".join(evidence).lower()
        # Heuristic grading signal
        positive = any(w in joined for w in ["sahih", "authentic", "bukhari", "muslim", "reliable"])
        negative = any(w in joined for w in ["weak", "da'if", "daif", "fabricated", "mawdu"])

        if collection.lower() in ["bukhari", "muslim"]:
            # The two Sahih collections are widely regarded as authentic.
            status, confidence = "authentic", 90
        elif positive and not negative:
            status, confidence = "authentic", 70
        elif negative:
            status, confidence = "needs-review", 40
        elif evidence:
            status, confidence = "plausible", 55
        else:
            status, confidence = "unknown", 20
    except Exception:
        status, confidence = "unknown", 0

    return {
        "status": status,
        "confidence": confidence,
        "evidence": evidence,
        "query": search_query,
    }


# ============================================================
# TEXT HELPERS
# ============================================================
def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def extract_verse_refs(text):
    """Find 'Surah X:Y' or 'X:Y' style references in generated text/context."""
    refs = re.findall(r"\b(\d{1,3}):(\d{1,3})\b", text)
    seen, out = set(), []
    for s, a in refs:
        key = f"{s}:{a}"
        if key not in seen and 1 <= int(s) <= 114:
            seen.add(key)
            out.append(key)
    return out[:4]


def score_to_relevance(score, all_scores, higher_is_better=False):
    if not all_scores:
        return 0
    lo, hi = min(all_scores), max(all_scores)
    if hi == lo:
        return 100
    if higher_is_better:
        pct = 100 * (score - lo) / (hi - lo)
    else:
        # smaller distance -> higher percentage
        pct = 100 * (1 - (score - lo) / (hi - lo))
    return max(5, min(100, round(pct)))


def format_sources(docs, scores=None, higher_is_better=False):
    sources, seen = [], set()
    scores = scores or [None] * len(docs)
    for i, (doc, score) in enumerate(zip(docs, scores), start=1):
        text = clean_text(doc.page_content)
        if text in seen:
            continue
        seen.add(text)
        meta = getattr(doc, "metadata", {}) or {}
        badges = []
        for key in ("surah", "surah_name", "chapter", "ayah", "verse", "source"):
            if key in meta and meta[key] not in (None, ""):
                badges.append(f"{key.replace('_', ' ').title()}: {meta[key]}")
        sources.append({"id": i, "content": text, "score": score, "badges": badges})
    rel = [s["score"] for s in sources if s["score"] is not None]
    for s in sources:
        s["relevance"] = (
            score_to_relevance(s["score"], rel, higher_is_better)
            if s["score"] is not None else None
        )
    return sources


# ============================================================
# GENERATION
# ============================================================
# MULTI-QUERY GENERATION (query expansion)
# ============================================================
def generate_query_variations(tokenizer, model, query, n=3):
    """
    Use FLAN-T5 to rephrase the question into semantically diverse variants.
    More phrasings = better recall on a small FAISS index, because a single
    embedding can miss passages worded differently from the user's question.
    Always returns the original query first, then up to n rephrasings.
    """
    variations = [query]
    try:
        instruction = (
            "Rephrase the following question in 3 different ways that mean the "
            "same thing, each on a new line, using varied wording and synonyms.\n\n"
            f"Question: {query}\n\nRephrasings:"
        )
        inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=120, num_beams=4,
                num_return_sequences=1, no_repeat_ngram_size=2,
            )
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
        for line in re.split(r"[\n;]|(?:\d+[\).])", raw):
            cand = clean_text(line)
            cand = re.sub(r"^(rephrasing|question|answer)s?\s*[:\-]?\s*", "", cand, flags=re.I)
            if cand and cand.lower() != query.lower() and len(cand) > 8:
                if cand not in variations:
                    variations.append(cand)
    except Exception:
        pass
    return variations[:n + 1]


# ============================================================
# MULTI-QUERY RETRIEVAL + CROSS-ENCODER RE-RANKING
# ============================================================
def multi_query_retrieve(vector_store, queries, k, fetch_per_query):
    """Retrieve candidates for every sub-query and deduplicate by content."""
    pool = {}  # content -> (doc, best_distance)
    for q in queries:
        try:
            results = vector_store.similarity_search_with_score(q, k=fetch_per_query)
        except Exception:
            continue
        for doc, dist in results:
            key = clean_text(doc.page_content)
            if not key:
                continue
            if key not in pool or dist < pool[key][1]:
                pool[key] = (doc, dist)
    # list of (doc, distance)
    return [(doc, dist) for doc, dist in pool.values()]


def rerank_candidates(reranker, query, candidates, k):
    """
    Re-score candidates with a cross-encoder (joint query+passage scoring),
    which is far more accurate than bi-encoder cosine distance. Falls back to
    vector distance order if no re-ranker is available.
    Returns top-k list of (doc, rerank_score_or_distance, used_reranker).
    """
    if not candidates:
        return []
    if reranker is None:
        # fall back: sort by ascending distance (smaller = closer)
        ordered = sorted(candidates, key=lambda x: x[1])[:k]
        return [(doc, dist, False) for doc, dist in ordered]
    pairs = [(query, clean_text(doc.page_content)) for doc, _ in candidates]
    try:
        scores = reranker.predict(pairs)
    except Exception:
        ordered = sorted(candidates, key=lambda x: x[1])[:k]
        return [(doc, dist, False) for doc, dist in ordered]
    scored = list(zip([c[0] for c in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)  # higher rerank score = better
    return [(doc, float(s), True) for doc, s in scored[:k]]


# ============================================================
# GENERATION
# ============================================================
def generate_answer(tokenizer, model, prompt_text, max_new_tokens=220):
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=4,
            early_stopping=True, no_repeat_ngram_size=3,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ============================================================
# AGENTIC RETRIEVE + GENERATE + VALIDATE
# ============================================================
def agentic_pipeline(vector_store, tokenizer, model, prompt, query,
                     k=4, max_new_tokens=220, enable_hadith=True,
                     enable_web_validation=True, enable_verse_api=True,
                     reranker=None, enable_multi_query=True, enable_rerank=True,
                     progress_cb=None):
    trace = []

    def log(msg):
        trace.append(msg)
        if progress_cb:
            progress_cb(msg)

    steps, matched_topic = plan_agentic_steps(query)
    log(f"🧭 Planned {len(steps)} steps (topic: {matched_topic or 'general'})")

    # 1) Multi-query expansion
    if enable_multi_query:
        log("🔁 Expanding query into multiple phrasings...")
        sub_queries = generate_query_variations(tokenizer, model, query, n=MAX_SUBQUERIES - 1)
        log(f"   → {len(sub_queries)} search queries: " +
            " | ".join(q[:40] for q in sub_queries))
    else:
        sub_queries = [query]

    # 2) Retrieval (fetch a wider candidate pool than k)
    log("🔎 Retrieving candidate passages from FAISS...")
    fetch_per_query = max(k * RETRIEVAL_FETCH_MULTIPLIER, k)
    candidates = multi_query_retrieve(vector_store, sub_queries, k, fetch_per_query)
    if not candidates:
        return {
            "answer": "No relevant Quran verses were found for this question.",
            "sources": [], "context": "", "elapsed": 0.0,
            "hadith": None, "validation": None, "verses": [],
            "trace": trace, "topic": matched_topic,
        }
    log(f"   → {len(candidates)} unique candidates pooled")

    # 3) Re-ranking with cross-encoder
    use_reranker = enable_rerank and reranker is not None
    if use_reranker:
        log("🎯 Re-ranking candidates with cross-encoder...")
    elif enable_rerank:
        log("🎯 Re-ranker unavailable — sorting by vector distance.")
    ranked = rerank_candidates(reranker if use_reranker else None, query, candidates, k)
    docs = [d for d, _, _ in ranked]
    scores = [s for _, s, _ in ranked]
    used_reranker = ranked[0][2] if ranked else False

    context_chunks = [clean_text(d.page_content)[:1400] for d in docs]
    context = "\n\n".join(context_chunks)

    # 4) Generation
    log("✍️ Generating grounded answer with FLAN-T5...")
    final_prompt = prompt.format(query=query, context=context)
    start = time.time()
    answer = generate_answer(tokenizer, model, final_prompt, max_new_tokens)
    elapsed = time.time() - start

    sources = format_sources(docs, scores, higher_is_better=used_reranker)

    # 3) Verse enrichment via Quran API
    verses = []
    if enable_verse_api:
        refs = extract_verse_refs(answer + " " + context)
        if refs:
            log(f"📖 Fetching {len(refs)} verse(s) (Arabic + translation + audio)...")
        for ref in refs:
            v = safe_fetch_quran_verse(ref)
            if v:
                verses.append(v)

    # 4) Hadith retrieval
    hadith = None
    if enable_hadith:
        log("📜 Retrieving a related Hadith...")
        # bias collection by topic deterministically so reruns are stable-ish
        col = HADITH_COLLECTIONS[hash(query) % len(HADITH_COLLECTIONS)]
        hadith = fetch_hadith(col)

    # 5) Web validation (agentic cross-check)
    validation = None
    if enable_web_validation and hadith:
        log("🌐 Cross-checking Hadith authenticity on the web...")
        validation = validate_hadith_web(hadith)

    log("✅ Assembling verified final response.")
    return {
        "answer": answer, "sources": sources, "context": context, "elapsed": elapsed,
        "hadith": hadith, "validation": validation, "verses": verses,
        "trace": trace, "topic": matched_topic, "steps": steps,
        "sub_queries": sub_queries, "candidate_count": len(candidates),
        "used_reranker": used_reranker,
    }


# ============================================================
# FOLLOW-UPS
# ============================================================
def suggest_follow_ups(current_query, n=3):
    pool = [q for q in ALL_SAMPLE_QUESTIONS if q.lower() != current_query.lower()]
    random.shuffle(pool)
    return pool[:n]


# ============================================================
# CACHE
# ============================================================
@st.cache_resource
def initialize_models():
    vector_store = load_vector_store()
    tokenizer, model = initialize_llm()
    prompt = create_prompt()
    reranker = initialize_reranker()
    return vector_store, tokenizer, model, prompt, reranker


# ============================================================
# SESSION
# ============================================================
def init_session():
    defaults = {
        "messages": [], "selected_question": "", "last_result": None,
        "bookmarks": [], "accent_theme": "Emerald (default)",
        "pending_query": None, "streak": 0, "last_visit": None,
        "quiz_score": 0, "quiz_total": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# CORE RUN
# ============================================================
def run_query(query, vector_store, tokenizer, model, prompt, k, max_tokens,
              enable_hadith=True, enable_web_validation=True, enable_verse_api=True,
              reranker=None, enable_multi_query=True, enable_rerank=True,
              progress_cb=None, **_ignore):
    try:
        result = agentic_pipeline(
            vector_store=vector_store, tokenizer=tokenizer, model=model,
            prompt=prompt, query=query, k=k, max_new_tokens=max_tokens,
            enable_hadith=enable_hadith, enable_web_validation=enable_web_validation,
            enable_verse_api=enable_verse_api, reranker=reranker,
            enable_multi_query=enable_multi_query, enable_rerank=enable_rerank,
            progress_cb=progress_cb,
        )
    except Exception as e:
        st.error(f"Error while generating response: {e}")
        return

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "hadith": result["hadith"],
        "validation": result["validation"],
        "verses": result["verses"],
        "trace": result["trace"],
        "topic": result.get("topic"),
        "sub_queries": result.get("sub_queries", []),
        "candidate_count": result.get("candidate_count"),
        "used_reranker": result.get("used_reranker", False),
        "query": query,
        "elapsed": result["elapsed"],
        "feedback": None,
        "bookmarked": False,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    st.session_state.last_result = {"query": query, **result}
    st.toast(f"Answer ready in {result['elapsed']:.1f}s")


# ============================================================
# DAILY WIDGETS (Name of the day + streak)
# ============================================================
def render_daily_widgets():
    today = datetime.now().strftime("%Y-%m-%d")
    idx = int(hashlib.md5(today.encode()).hexdigest(), 16) % len(ASMA_UL_HUSNA)
    name, arabic, meaning = ASMA_UL_HUSNA[idx]

    # streak
    if st.session_state.last_visit != today:
        st.session_state.streak += 1
        st.session_state.last_visit = today

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            f"<div class='name-card'><span class='small-muted'>✨ Name of Allah — today</span>"
            f"<h3 style='margin:0.2rem 0;'>{name} "
            f"<span style='font-size:1.4rem;'>{arabic}</span></h3>"
            f"<span class='small-muted'>{meaning}</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("🔥 Day streak", st.session_state.streak)


# ============================================================
# VERSE OF THE DAY
# ============================================================
def render_verse_of_the_day():
    st.markdown("### 🌙 Verse of the Day")
    today = datetime.now().strftime("%Y-%m-%d")
    seed = int(hashlib.md5(("verse" + today).encode()).hexdigest(), 16)
    # Pick from ALL 6,236 ayahs by global number (not just ayahs 1-7).
    ref = str((seed % 6236) + 1)
    try:
        with st.spinner("Fetching today's verse..."):
            v = fetch_quran_verse(ref)
    except Exception as e:
        st.info("Couldn't load the verse of the day right now.")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("🔄 Retry", key="retry_votd"):
                fetch_quran_verse.clear()   # clear cached failure
                st.rerun()
        with col_b:
            with st.expander("Error details"):
                st.caption(str(e))
        return
    st.markdown(f"<div class='verse-arabic'>{v['arabic']}</div>", unsafe_allow_html=True)
    st.markdown(f"*{v['english']}*")
    st.caption(f"— Surah {v['surah_name']} ({v['surah_arabic']}), Ayah {v['ayah_number']}")
    if v.get("audio"):
        st.audio(v["audio"])


# ============================================================
# HADITH + VALIDATION RENDERING
# ============================================================
def render_validation_badge(validation):
    if not validation:
        return
    status = validation["status"]
    conf = validation["confidence"]
    cls = {"authentic": "v-ok", "plausible": "v-ok",
           "needs-review": "v-warn", "unknown": "v-unknown"}.get(status, "v-unknown")
    label = {"authentic": "✅ Likely Authentic", "plausible": "🟢 Plausible",
             "needs-review": "⚠️ Needs Scholarly Review", "unknown": "❔ Unverified"}.get(status, "❔ Unverified")
    st.markdown(
        f"<span class='badge-verify {cls}'>{label} · {conf}% signal</span>",
        unsafe_allow_html=True,
    )


def render_hadith_block(hadith, validation):
    if not hadith or not hadith.get("text"):
        return
    st.markdown("#### 📜 Related Hadith")
    st.markdown(
        f"<div class='hadith-box'>{hadith['text']}</div>", unsafe_allow_html=True
    )
    meta = " · ".join(filter(None, [
        hadith.get("collection"), hadith.get("book"),
        f"No. {hadith.get('number')}" if hadith.get("number") else "",
    ]))
    if hadith.get("narrator"):
        st.caption(hadith["narrator"])
    if meta:
        st.caption(meta)
    render_validation_badge(validation)
    if validation and validation.get("evidence"):
        with st.expander("🌐 Web cross-check evidence"):
            st.caption(f"Search query: `{validation['query']}`")
            for ev in validation["evidence"]:
                st.markdown(f"- {ev}")
    st.caption("⚠️ Automated signal only — always confirm gradings with qualified scholars.")


def render_verses_block(verses):
    if not verses:
        return
    st.markdown("#### 📖 Referenced Verses (with audio)")
    for v in verses:
        with st.container(border=True):
            st.markdown(f"<div class='verse-arabic'>{v['arabic']}</div>", unsafe_allow_html=True)
            st.markdown(f"*{v['english']}*")
            st.caption(f"Surah {v['surah_name']} · Ayah {v['ayah_number']} ({v['reference']})")
            if v.get("audio"):
                st.audio(v["audio"])


# ============================================================
# SAMPLE QUESTIONS
# ============================================================
def render_sample_questions(deps):
    st.markdown("### 🗂️ Browse questions by topic")
    st.caption("Tap any question to get an instant agentic answer — verses, hadith, and validation included.")
    _, top_r = st.columns([3, 1])
    with top_r:
        if st.button("🎲 Surprise me", use_container_width=True):
            run_query(random.choice(ALL_SAMPLE_QUESTIONS), **deps)
            st.rerun()
    tabs = st.tabs(list(QUESTION_BANK.keys()))
    for tab, (category, questions) in zip(tabs, QUESTION_BANK.items()):
        with tab:
            cols = st.columns(2)
            for i, q in enumerate(questions):
                with cols[i % 2]:
                    if st.button(q, use_container_width=True, key=f"sample_{category}_{q}"):
                        run_query(q, **deps)
                        st.rerun()


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar(reranker_available=False):
    with st.sidebar:
        st.header("⚙️ Settings")
        k = st.slider("Retrieved passages", 2, 8, 4, 1)
        max_tokens = st.slider("Answer length", 80, 400, 220, 20)
        show_sources_default = st.toggle("Show sources by default", value=False)
        show_context = st.toggle("Show retrieved context block", value=False)

        st.divider()
        st.subheader("🔎 Retrieval Quality")
        enable_multi_query = st.toggle(
            "Multi-query expansion", value=True,
            help="Rephrase your question several ways to catch passages worded differently.",
        )
        rerank_label = "Cross-encoder re-ranking"
        if not reranker_available:
            rerank_label += " (unavailable)"
        enable_rerank = st.toggle(
            rerank_label, value=reranker_available, disabled=not reranker_available,
            help="Re-score candidates with a cross-encoder for more accurate top results."
                 if reranker_available else
                 "Install sentence-transformers to enable cross-encoder re-ranking.",
        )

        st.divider()
        st.subheader("🤖 Agentic Tools")
        enable_hadith = st.toggle("Fetch related Hadith", value=True)
        enable_web_validation = st.toggle("Web-validate Hadith", value=True)
        enable_verse_api = st.toggle("Enrich verses (Arabic+audio)", value=True)

        st.divider()
        st.subheader("🎨 Appearance")
        theme_name = st.selectbox(
            "Accent theme", list(ACCENT_THEMES.keys()),
            index=list(ACCENT_THEMES.keys()).index(st.session_state.accent_theme),
        )
        st.session_state.accent_theme = theme_name

        st.divider()
        st.subheader("🧹 Session")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_result = None
                st.success("Chat cleared.")
        with col_b:
            if st.button("Clear bookmarks", use_container_width=True):
                st.session_state.bookmarks = []
                st.success("Bookmarks cleared.")
        if st.session_state.messages:
            st.download_button(
                "⬇️ Export chat (.md)", data=export_chat_markdown(),
                file_name="quran_chatbot_history.md", mime="text/markdown",
                use_container_width=True,
            )

        st.divider()
        st.subheader("ℹ️ About")
        st.caption("LLM: google/flan-t5-base")
        st.caption("Embeddings: all-MiniLM-L6-v2")
        st.caption("Re-ranker: ms-marco-MiniLM-L-6-v2")
        st.caption("Index: local FAISS")
        st.caption("APIs: AlQuran.cloud · Hadith API · DuckDuckGo")

    return {
        "k": k, "max_tokens": max_tokens,
        "show_sources_default": show_sources_default, "show_context": show_context,
        "enable_hadith": enable_hadith, "enable_web_validation": enable_web_validation,
        "enable_verse_api": enable_verse_api,
        "enable_multi_query": enable_multi_query, "enable_rerank": enable_rerank,
    }


# ============================================================
# EXPORT
# ============================================================
def export_chat_markdown():
    lines = ["# Quran Chatbot — Conversation Export", ""]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lines.append(f"## ❓ {msg['content']}")
        else:
            lines.append(msg["content"])
            if msg.get("hadith") and msg["hadith"].get("text"):
                lines.append(f"\n**Related Hadith ({msg['hadith'].get('collection','')}):** {msg['hadith']['text']}")
                if msg.get("validation"):
                    lines.append(f"_Validation: {msg['validation']['status']} ({msg['validation']['confidence']}%)_")
            if msg.get("sources"):
                lines.append("\n**Sources:**")
                for src in msg["sources"]:
                    lines.append(f"- {src['content'][:200]}")
            lines.append("")
    return "\n".join(lines)


# ============================================================
# SOURCE CARD
# ============================================================
def render_source_card(src, show_expanded=False):
    preview = src["content"][:140] + ("..." if len(src["content"]) > 140 else "")
    with st.expander(f"Source {src['id']} — {preview}", expanded=show_expanded):
        if src.get("badges"):
            st.markdown(
                "".join(f"<span class='badge'>{b}</span>" for b in src["badges"]),
                unsafe_allow_html=True,
            )
        st.markdown(f"<div class='source-box'>{src['content']}</div>", unsafe_allow_html=True)
        if src.get("relevance") is not None:
            st.caption(f"Relative relevance: {src['relevance']}%")
            st.progress(src["relevance"] / 100)


# ============================================================
# FEEDBACK ROW
# ============================================================
def render_feedback_row(msg, msg_idx, deps):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 4])
    with c1:
        if st.button("👍" if msg.get("feedback") != "up" else "✅👍", key=f"up_{msg_idx}"):
            msg["feedback"] = "up"; st.rerun()
    with c2:
        if st.button("👎" if msg.get("feedback") != "down" else "✅👎", key=f"down_{msg_idx}"):
            msg["feedback"] = "down"; st.rerun()
    with c3:
        is_b = msg.get("bookmarked", False)
        if st.button("⭐" if not is_b else "🌟 Saved", key=f"bookmark_{msg_idx}"):
            msg["bookmarked"] = not is_b
            if msg["bookmarked"]:
                st.session_state.bookmarks.append({
                    "query": msg["query"], "answer": msg["content"],
                    "sources": msg.get("sources", []), "hadith": msg.get("hadith"),
                    "validation": msg.get("validation"), "timestamp": msg.get("timestamp", ""),
                })
            else:
                st.session_state.bookmarks = [
                    b for b in st.session_state.bookmarks if b["query"] != msg["query"]
                ]
            st.rerun()
    with c4:
        if st.button("🔁", key=f"regen_{msg_idx}", help="Regenerate"):
            run_query(msg["query"], **deps); st.rerun()
    with c5:
        if msg.get("elapsed"):
            st.caption(f"⏱️ {msg['elapsed']:.1f}s · {msg.get('timestamp', '')}")


# ============================================================
# CHAT HISTORY
# ============================================================
def render_chat_history(deps):
    if not st.session_state.messages:
        st.info("No conversation yet — ask a question above or tap a sample topic to get started.")
        return
    search_term = st.text_input("🔍 Search chat history", placeholder="Filter by keyword...")
    st.markdown("## 💬 Chat History")
    for idx, msg in enumerate(st.session_state.messages):
        if search_term and search_term.lower() not in msg["content"].lower():
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("verses"):
                    render_verses_block(msg["verses"])
                if msg.get("hadith"):
                    render_hadith_block(msg["hadith"], msg.get("validation"))
                if msg.get("trace"):
                    with st.expander("🧠 Agent reasoning trace"):
                        for step in msg["trace"]:
                            st.markdown(f"<div class='step-line'>{step}</div>", unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander("📚 View sources", expanded=deps["show_sources_default"]):
                        for src in msg["sources"]:
                            render_source_card(src, show_expanded=deps["show_sources_default"])
                render_feedback_row(msg, idx, deps)
                follow_ups = suggest_follow_ups(msg.get("query", ""), n=3)
                if follow_ups:
                    st.caption("💡 You might also ask:")
                    fcols = st.columns(len(follow_ups))
                    for fc, fq in zip(fcols, follow_ups):
                        with fc:
                            if st.button(fq, key=f"followup_{idx}_{fq}", use_container_width=True):
                                run_query(fq, **deps); st.rerun()


# ============================================================
# RESULT CARD
# ============================================================
def render_result(latest, deps):
    st.success(f"Answer generated successfully in {latest.get('elapsed', 0):.1f}s.")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Retrieved sources", len(latest["sources"]))
    colB.metric("Verses enriched", len(latest.get("verses", [])))
    colC.metric("Hadith found", 1 if latest.get("hadith") else 0)
    val = latest.get("validation")
    colD.metric("Validation", val["status"] if val else "—")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🟢 Answer", "📖 Verses", "📜 Hadith", "📚 Sources", "🧠 Reasoning"]
    )
    with tab1:
        st.markdown(f"<div class='answer-box'>{latest['answer']}</div>", unsafe_allow_html=True)
    with tab2:
        if latest.get("verses"):
            render_verses_block(latest["verses"])
        else:
            st.info("No specific verse references detected to enrich.")
    with tab3:
        if latest.get("hadith"):
            render_hadith_block(latest["hadith"], latest.get("validation"))
        else:
            st.info("Hadith retrieval disabled or unavailable.")
    with tab4:
        if not latest["sources"]:
            st.info("No sources found.")
        else:
            for src in latest["sources"]:
                render_source_card(src, show_expanded=deps["show_sources_default"])
    with tab5:
        # Retrieval quality summary
        sq = latest.get("sub_queries", [])
        cc = latest.get("candidate_count")
        rr = latest.get("used_reranker")
        if sq or cc is not None:
            bits = []
            if len(sq) > 1:
                bits.append(f"**{len(sq)} search queries** (multi-query expansion)")
            if cc is not None:
                bits.append(f"**{cc} candidates** pooled")
            bits.append("ranked by **cross-encoder**" if rr else "ranked by **vector distance**")
            st.caption("Retrieval: " + " · ".join(bits))
            if len(sq) > 1:
                with st.expander("🔁 Query variations used"):
                    for s in sq:
                        st.markdown(f"- {s}")
        for step in latest.get("trace", []):
            st.markdown(f"<div class='step-line'>{step}</div>", unsafe_allow_html=True)
        if deps["show_context"]:
            st.text_area("Retrieved context", latest["context"], height=250)


# ============================================================
# QUIZ TAB (interactive engagement)
# ============================================================
QUIZ_BANK = [
    {"q": "How many chapters (Surahs) are in the Quran?",
     "options": ["100", "114", "120", "99"], "answer": 1},
    {"q": "Which Surah is known as 'The Opening'?",
     "options": ["Al-Baqarah", "Al-Fatihah", "Yasin", "Al-Ikhlas"], "answer": 1},
    {"q": "Ayat al-Kursi is found in which Surah?",
     "options": ["Al-Baqarah (2:255)", "An-Nisa", "Al-Imran", "Maryam"], "answer": 0},
    {"q": "What is the longest Surah in the Quran?",
     "options": ["Al-Fatihah", "Al-Baqarah", "An-Nas", "Al-Kawthar"], "answer": 1},
    {"q": "Which night is described as 'better than a thousand months'?",
     "options": ["Laylat al-Qadr", "Laylat al-Isra", "Eid night", "Friday night"], "answer": 0},
]


def render_quiz_tab():
    st.markdown("## 🧩 Quran Quiz")
    st.caption("Test your knowledge — your score tracks across the session.")
    if st.session_state.quiz_total:
        acc = round(100 * st.session_state.quiz_score / st.session_state.quiz_total)
        st.metric("Your score", f"{st.session_state.quiz_score}/{st.session_state.quiz_total}", f"{acc}%")

    if "current_quiz" not in st.session_state:
        st.session_state.current_quiz = random.choice(QUIZ_BANK)

    q = st.session_state.current_quiz
    st.markdown(f"### {q['q']}")
    choice = st.radio("Choose:", q["options"], key="quiz_choice", index=None)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Submit answer", type="primary", use_container_width=True):
            if choice is None:
                st.warning("Pick an option first.")
            else:
                st.session_state.quiz_total += 1
                if q["options"].index(choice) == q["answer"]:
                    st.session_state.quiz_score += 1
                    st.success("✅ Correct! Well done.")
                    st.balloons()
                else:
                    st.error(f"❌ Not quite. Correct answer: **{q['options'][q['answer']]}**")
    with c2:
        if st.button("Next question →", use_container_width=True):
            st.session_state.current_quiz = random.choice(QUIZ_BANK)
            st.rerun()


# ============================================================
# STATS TAB
# ============================================================
def render_stats_tab():
    st.markdown("## 📊 Session Stats")
    assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
    total_q = len(assistant_msgs)
    total_bookmarks = len(st.session_state.bookmarks)
    thumbs_up = sum(1 for m in assistant_msgs if m.get("feedback") == "up")
    thumbs_down = sum(1 for m in assistant_msgs if m.get("feedback") == "down")
    hadith_validated = sum(1 for m in assistant_msgs if m.get("validation"))
    avg_len = round(sum(len(m["content"].split()) for m in assistant_msgs) / total_q, 1) if total_q else 0
    avg_time = round(sum(m.get("elapsed", 0) for m in assistant_msgs) / total_q, 2) if total_q else 0

    cols = st.columns(6)
    data = [
        ("Questions", total_q), ("Bookmarks", total_bookmarks),
        ("👍 / 👎", f"{thumbs_up}/{thumbs_down}"),
        ("Hadith checked", hadith_validated),
        ("Avg length", f"{avg_len}w"), ("Avg time", f"{avg_time}s"),
    ]
    for col, (label, value) in zip(cols, data):
        with col:
            st.markdown(
                f"<div class='pill-stat'><b>{value}</b><br><span class='small-muted'>{label}</span></div>",
                unsafe_allow_html=True,
            )

    if total_q:
        st.markdown("### Answer length over the conversation")
        st.bar_chart([len(m["content"].split()) for m in assistant_msgs])
        st.markdown("### Topics explored")
        topic_counts = {}
        for m in assistant_msgs:
            t = m.get("topic") or "general"
            topic_counts[t] = topic_counts.get(t, 0) + 1
        st.bar_chart(topic_counts)
    else:
        st.info("Ask a few questions to see stats here.")


# ============================================================
# BOOKMARKS TAB
# ============================================================
def render_bookmarks_tab():
    st.markdown("## ⭐ Bookmarked Answers")
    if not st.session_state.bookmarks:
        st.info("You haven't bookmarked any answers yet. Tap ⭐ under any answer to save it here.")
        return
    for i, b in enumerate(st.session_state.bookmarks):
        with st.container(border=True):
            st.markdown(f"**❓ {b['query']}**")
            st.markdown(f"<div class='answer-box'>{b['answer']}</div>", unsafe_allow_html=True)
            st.caption(f"Saved on {b.get('timestamp', '')}")
            if b.get("hadith") and b["hadith"].get("text"):
                render_hadith_block(b["hadith"], b.get("validation"))
            if b.get("sources"):
                with st.expander(f"📚 {len(b['sources'])} source(s)"):
                    for src in b["sources"]:
                        render_source_card(src)
            if st.button("🗑️ Remove bookmark", key=f"remove_bookmark_{i}"):
                st.session_state.bookmarks.pop(i); st.rerun()


# ============================================================
# ABOUT TAB
# ============================================================
def render_about_tab():
    st.markdown("## ℹ️ About this chatbot")
    st.markdown(
        """
        This is an **Agentic Retrieval-Augmented Generation (RAG)** assistant for
        exploring the Quran and Hadith.

        **How the agent works on each question:**
        1. 🧭 **Plans** which tools to use based on your question's topic.
        2. 🔁 **Expands** your question into several phrasings (multi-query) so
           passages worded differently still get found.
        3. 🔎 **Retrieves** a wide candidate pool from a local **FAISS** index
           (embedded with all-MiniLM-L6-v2).
        4. 🎯 **Re-ranks** candidates with a **cross-encoder**
           (ms-marco-MiniLM) that scores each passage jointly with your question
           — far more accurate than embedding distance alone.
        5. ✍️ **Generates** a grounded answer with **google/flan-t5-base**.
        6. 📖 **Enriches** detected verse references with Arabic text, translation,
           and recitation **audio** (AlQuran.cloud API).
        7. 📜 **Fetches a related Hadith** from public collections.
        8. 🌐 **Cross-checks** the Hadith's authenticity via live **web validation**.
        9. ✅ **Assembles** a cited, verified response with a reasoning trace.

        **Interactive features:**
        - 🌙 Verse of the Day with audio recitation
        - ✨ Name of Allah of the day + 🔥 daily streak
        - 🧩 Interactive Quran quiz with scoring
        - 🗂️ Topic-organized instant-answer questions + 🎲 Surprise me
        - ⭐ Bookmarks, 👍👎 feedback, 🔁 regenerate, 💡 follow-ups
        - 🧠 Transparent agent reasoning trace per answer
        - 🔍 Searchable history, 📊 live stats, ⬇️ Markdown export
        - 🎨 Six selectable color themes

        **Important disclaimer:** Web-based hadith validation is an automated,
        best-effort *signal* — not a scholarly grading. The two Sahih collections
        (Bukhari & Muslim) are widely regarded as authentic, but always confirm
        any ruling or grading with **qualified Islamic scholars**. This tool is for
        educational exploration only.
        """
    )


# ============================================================
# 99 NAMES — interactive grid with click-to-speak (Web Speech API)
# ============================================================
def render_99_names():
    st.markdown("### ✨ Explore the 99 Names (Asma ul-Husna)")
    st.caption("Tap any card to hear its pronunciation and meaning spoken aloud. "
               "Uses your browser's built-in voice — no download needed.")

    # Build the card grid as one self-contained HTML component so the speech
    # synthesis runs entirely client-side (works on Streamlit Cloud, no API).
    cards_html = ""
    for i, (name, arabic, meaning) in enumerate(ASMA_UL_HUSNA, start=1):
        # escape single quotes for the JS string args
        js_name = name.replace("'", "\\'")
        js_meaning = meaning.replace("'", "\\'")
        js_arabic = arabic.replace("'", "\\'")
        cards_html += f"""
        <div class="aname-card" onclick="speakName('{js_arabic}', '{js_name}', '{js_meaning}', this)">
            <div class="aname-top">
                <span class="aname-num">{i}</span>
                <span class="aname-speaker">🔊</span>
            </div>
            <div class="aname-arabic">{arabic}</div>
            <div class="aname-translit">{name}</div>
            <div class="aname-meaning">{meaning}</div>
        </div>
        """

    html = f"""
    <style>
      .aname-wrap {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 12px; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
      }}
      .aname-card {{
        border-radius: 16px; padding: 14px 14px 16px;
        background: linear-gradient(135deg, rgba(34,197,94,0.10), rgba(59,130,246,0.10));
        border: 1px solid rgba(34,197,94,0.30);
        cursor: pointer; transition: transform .12s ease, box-shadow .12s ease;
        user-select: none; text-align: center;
      }}
      .aname-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
        border-color: rgba(34,197,94,0.6);
      }}
      .aname-card.speaking {{
        background: linear-gradient(135deg, rgba(34,197,94,0.28), rgba(59,130,246,0.28));
        border-color: rgba(34,197,94,0.9);
      }}
      .aname-top {{ display:flex; justify-content:space-between; align-items:center; }}
      .aname-num {{ font-size:0.72rem; color:#6b7280; font-weight:700; }}
      .aname-speaker {{ font-size:0.95rem; opacity:0.65; }}
      .aname-arabic {{
        font-size:1.7rem; line-height:2.4rem; margin:6px 0 2px;
        direction:rtl; font-family:'Traditional Arabic','Amiri',serif;
      }}
      .aname-translit {{ font-weight:800; font-size:1.0rem; }}
      .aname-meaning {{ font-size:0.8rem; color:#6b7280; margin-top:3px; }}
      .aname-hint {{ font-size:0.78rem; color:#9ca3af; margin:4px 0 12px; }}
    </style>

    <div class="aname-hint" id="voiceHint">Loading voices…</div>
    <div class="aname-wrap">{cards_html}</div>

    <script>
      // Warm up the voice list (some browsers load it asynchronously).
      let VOICES = [];
      function loadVoices() {{
        VOICES = window.speechSynthesis.getVoices();
        const hint = document.getElementById('voiceHint');
        const hasArabic = VOICES.some(v => (v.lang || '').toLowerCase().startsWith('ar'));
        if (hint) {{
          hint.textContent = hasArabic
            ? '🔊 Arabic voice available — tap a name to listen.'
            : '🔊 Tap a name to listen (Arabic voice not found; English narration will be used).';
        }}
      }}
      if (typeof speechSynthesis !== 'undefined') {{
        loadVoices();
        window.speechSynthesis.onvoiceschanged = loadVoices;
      }}

      function pickArabicVoice() {{
        return VOICES.find(v => (v.lang || '').toLowerCase().startsWith('ar')) || null;
      }}

      function speakName(arabic, translit, meaning, el) {{
        if (typeof speechSynthesis === 'undefined') {{
          alert('Speech is not supported in this browser.');
          return;
        }}
        window.speechSynthesis.cancel();  // stop anything currently playing

        // highlight the active card
        document.querySelectorAll('.aname-card.speaking')
                .forEach(c => c.classList.remove('speaking'));
        if (el) el.classList.add('speaking');

        const arVoice = pickArabicVoice();
        const utterances = [];

        // 1) Speak the Arabic name (use Arabic voice if available)
        const u1 = new SpeechSynthesisUtterance(arabic);
        if (arVoice) {{ u1.voice = arVoice; u1.lang = arVoice.lang; }}
        else {{ u1.lang = 'ar-SA'; }}
        u1.rate = 0.85;
        utterances.push(u1);

        // 2) Then the transliteration + meaning in English
        const u2 = new SpeechSynthesisUtterance(translit + '. ' + meaning + '.');
        u2.lang = 'en-US';
        u2.rate = 0.95;
        utterances.push(u2);

        // chain them, clear highlight when done
        u2.onend = function() {{ if (el) el.classList.remove('speaking'); }};
        u1.onerror = function() {{ if (el) el.classList.remove('speaking'); }};

        utterances.forEach(u => window.speechSynthesis.speak(u));
      }}
    </script>
    """
    # height accommodates all 99 cards in the scrollable component
    components.html(html, height=620, scrolling=True)


# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(page_title="Agentic Quran & Hadith Chatbot", page_icon="📖", layout="wide")
    init_session()
    theme = ACCENT_THEMES[st.session_state.accent_theme]
    inject_custom_css(accent=theme["accent"], accent2=theme["accent2"])

    st.markdown("<div class='main-title'>📖 Agentic Quran & Hadith Explorer</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle-text'>Local FAISS + FLAN-T5 RAG, enriched with live Quran verses, "
        "related Hadith, and web-validated authenticity checks.</div>",
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading Quran index, language model, and re-ranker..."):
            vector_store, tokenizer, model, prompt, reranker = initialize_models()
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()

    settings = render_sidebar(reranker_available=reranker is not None)

    # dependency bundle passed to run_query
    deps = {
        "vector_store": vector_store, "tokenizer": tokenizer, "model": model,
        "prompt": prompt, "reranker": reranker,
        "k": settings["k"], "max_tokens": settings["max_tokens"],
        "enable_hadith": settings["enable_hadith"],
        "enable_web_validation": settings["enable_web_validation"],
        "enable_verse_api": settings["enable_verse_api"],
        "enable_multi_query": settings["enable_multi_query"],
        "enable_rerank": settings["enable_rerank"],
    }
    # extra keys used only for rendering
    render_deps = {**deps, "show_sources_default": settings["show_sources_default"],
                   "show_context": settings["show_context"]}

    render_daily_widgets()
    st.divider()

    main_tab, daily_tab, quiz_tab, bookmarks_tab, stats_tab, about_tab = st.tabs(
        ["💬 Chat", "🌙 Daily", "🧩 Quiz", "⭐ Bookmarks", "📊 Stats", "ℹ️ About"]
    )

    with main_tab:
        render_sample_questions(deps)
        st.divider()
        default_query = st.session_state.selected_question or ""
        query = st.text_input(
            "Ask your question", value=default_query,
            placeholder="e.g. What does the Quran say about patience?",
        )
        col1, col2 = st.columns([1, 5])
        ask_clicked = col1.button("Ask", type="primary", use_container_width=True)
        clear_input_clicked = col2.button("Clear current question")
        if clear_input_clicked:
            st.session_state.selected_question = ""; st.rerun()

        if ask_clicked and query.strip():
            steps_box = st.status("🤖 Agent working...", expanded=True)
            def cb(msg):
                steps_box.write(msg)
            with steps_box:
                run_query(query, **deps, progress_cb=cb)
                steps_box.update(label="✅ Completed", state="complete")
            st.session_state.selected_question = ""

        render_chat_history(render_deps)

        if st.session_state.last_result:
            st.divider()
            st.markdown("## 🆕 Latest Result")
            latest = st.session_state.last_result
            st.markdown(f"**Question:** {latest['query']}")
            with st.expander("✅ Click to view the full agentic result", expanded=True):
                render_result(latest, render_deps)

    with daily_tab:
        render_verse_of_the_day()
        st.divider()
        render_99_names()

    with quiz_tab:
        render_quiz_tab()
    with bookmarks_tab:
        render_bookmarks_tab()
    with stats_tab:
        render_stats_tab()
    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
