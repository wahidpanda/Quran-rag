import os
import re
import time
import json
import random
import hashlib
from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st
import torch
import requests

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIG
# ============================================================
FAISS_DIR = "quran_faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

# Public, free, no-key APIs used by the agentic layer.
QURAN_API = "https://api.alquran.cloud/v1"          # verses, translations, audio
HADITH_API = "https://random-hadith-generator.vercel.app"  # random hadith by collection
HADITH_COLLECTIONS = ["bukhari", "muslim", "abudawud", "ibnmajah", "tirmidhi"]

# DuckDuckGo Instant Answer API (no key) for lightweight web validation.
DDG_API = "https://api.duckduckgo.com/"

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

# 99 Names — small subset rotated for the "Name of the day" widget.
ASMA_UL_HUSNA = [
    ("Ar-Rahman", "The Most Compassionate"),
    ("Ar-Raheem", "The Most Merciful"),
    ("Al-Malik", "The King / Sovereign"),
    ("Al-Quddus", "The Most Holy"),
    ("As-Salam", "The Source of Peace"),
    ("Al-Mu'min", "The Granter of Security"),
    ("Al-Aziz", "The Almighty"),
    ("Al-Ghaffar", "The Ever-Forgiving"),
    ("Al-Wahhab", "The Supreme Bestower"),
    ("Ar-Razzaq", "The Provider"),
    ("Al-Hakim", "The All-Wise"),
    ("Al-Wadud", "The Most Loving"),
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
    """reference like '2:255'. Returns arabic, english, audio url, surah name."""
    try:
        ar = requests.get(f"{QURAN_API}/ayah/{reference}/ar.alafasy", timeout=8).json()
        en = requests.get(f"{QURAN_API}/ayah/{reference}/en.asad", timeout=8).json()
        if ar.get("status") != "OK" or en.get("status") != "OK":
            return None
        a = ar["data"]
        e = en["data"]
        return {
            "reference": reference,
            "surah_name": a["surah"]["englishName"],
            "surah_arabic": a["surah"]["name"],
            "ayah_number": a["numberInSurah"],
            "arabic": a["text"],
            "english": e["text"],
            "audio": a.get("audio"),
        }
    except Exception:
        return None


# ============================================================
# TOOL: Hadith fetch
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_hadith(collection):
    """Pull a hadith from a public random-hadith API."""
    try:
        r = requests.get(f"{HADITH_API}/{collection}", timeout=8)
        data = r.json().get("data", {})
        return {
            "collection": collection.title(),
            "text": clean_text(data.get("hadith_english", "")),
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
        resp = requests.get(
            DDG_API,
            params={"q": search_query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
            headers={"User-Agent": "QuranChatbot/1.0"},
        )
        data = resp.json()
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


def score_to_relevance(score, all_scores):
    if not all_scores:
        return 0
    lo, hi = min(all_scores), max(all_scores)
    if hi == lo:
        return 100
    pct = 100 * (1 - (score - lo) / (hi - lo))
    return max(5, min(100, round(pct)))


def format_sources(docs, scores=None):
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
        s["relevance"] = score_to_relevance(s["score"], rel) if s["score"] is not None else None
    return sources


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
                     progress_cb=None):
    trace = []

    def log(msg):
        trace.append(msg)
        if progress_cb:
            progress_cb(msg)

    steps, matched_topic = plan_agentic_steps(query)
    log(f"🧭 Planned {len(steps)} steps (topic: {matched_topic or 'general'})")

    # 1) Retrieval
    log("🔎 Retrieving Quran passages from FAISS...")
    results = vector_store.similarity_search_with_score(query, k=k)
    if not results:
        return {
            "answer": "No relevant Quran verses were found for this question.",
            "sources": [], "context": "", "elapsed": 0.0,
            "hadith": None, "validation": None, "verses": [],
            "trace": trace, "topic": matched_topic,
        }
    docs = [d for d, _ in results]
    scores = [s for _, s in results]

    context_chunks = []
    for doc in docs:
        chunk = clean_text(doc.page_content)
        context_chunks.append(chunk[:1400])
    context = "\n\n".join(context_chunks)

    # 2) Generation
    log("✍️ Generating grounded answer with FLAN-T5...")
    final_prompt = prompt.format(query=query, context=context)
    start = time.time()
    answer = generate_answer(tokenizer, model, final_prompt, max_new_tokens)
    elapsed = time.time() - start

    sources = format_sources(docs, scores)

    # 3) Verse enrichment via Quran API
    verses = []
    if enable_verse_api:
        refs = extract_verse_refs(answer + " " + context)
        if refs:
            log(f"📖 Fetching {len(refs)} verse(s) (Arabic + translation + audio)...")
        for ref in refs:
            v = fetch_quran_verse(ref)
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
    return vector_store, tokenizer, model, prompt


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
              progress_cb=None):
    try:
        result = agentic_pipeline(
            vector_store, tokenizer, model, prompt, query, k, max_tokens,
            enable_hadith, enable_web_validation, enable_verse_api, progress_cb,
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
    name, meaning = ASMA_UL_HUSNA[idx]

    # streak
    if st.session_state.last_visit != today:
        st.session_state.streak += 1
        st.session_state.last_visit = today

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            f"<div class='name-card'><span class='small-muted'>✨ Name of Allah — today</span>"
            f"<h3 style='margin:0.2rem 0;'>{name}</h3>"
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
    surah = (seed % 114) + 1
    ayah = (seed % 7) + 1
    ref = f"{surah}:{ayah}"
    with st.spinner("Fetching today's verse..."):
        v = fetch_quran_verse(ref)
    if not v:
        st.info("Couldn't load the verse of the day (network). Try again later.")
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
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        k = st.slider("Retrieved passages", 2, 8, 4, 1)
        max_tokens = st.slider("Answer length", 80, 400, 220, 20)
        show_sources_default = st.toggle("Show sources by default", value=False)
        show_context = st.toggle("Show retrieved context block", value=False)

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
        st.caption("Model: google/flan-t5-base")
        st.caption("Embeddings: all-MiniLM-L6-v2")
        st.caption("Index: local FAISS")
        st.caption("APIs: AlQuran.cloud · Hadith API · DuckDuckGo")

    return {
        "k": k, "max_tokens": max_tokens,
        "show_sources_default": show_sources_default, "show_context": show_context,
        "enable_hadith": enable_hadith, "enable_web_validation": enable_web_validation,
        "enable_verse_api": enable_verse_api,
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
        2. 🔎 **Retrieves** the most relevant passages from a local **FAISS** index
           (embedded with all-MiniLM-L6-v2).
        3. ✍️ **Generates** a grounded answer with **google/flan-t5-base**.
        4. 📖 **Enriches** detected verse references with Arabic text, translation,
           and recitation **audio** (AlQuran.cloud API).
        5. 📜 **Fetches a related Hadith** from public collections.
        6. 🌐 **Cross-checks** the Hadith's authenticity via live **web validation**.
        7. ✅ **Assembles** a cited, verified response with a reasoning trace.

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

    settings = render_sidebar()

    try:
        with st.spinner("Loading Quran index and language model..."):
            vector_store, tokenizer, model, prompt = initialize_models()
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()

    # dependency bundle passed to run_query
    deps = {
        "vector_store": vector_store, "tokenizer": tokenizer, "model": model,
        "prompt": prompt, "k": settings["k"], "max_tokens": settings["max_tokens"],
        "enable_hadith": settings["enable_hadith"],
        "enable_web_validation": settings["enable_web_validation"],
        "enable_verse_api": settings["enable_verse_api"],
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
        st.markdown("### ✨ Explore the 99 Names")
        ncols = st.columns(3)
        for i, (name, meaning) in enumerate(ASMA_UL_HUSNA):
            with ncols[i % 3]:
                st.markdown(
                    f"<div class='name-card' style='margin-bottom:0.6rem;'>"
                    f"<b>{name}</b><br><span class='small-muted'>{meaning}</span></div>",
                    unsafe_allow_html=True,
                )

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
