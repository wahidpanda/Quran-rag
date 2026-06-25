import os
import re
import time
import random
from datetime import datetime

import streamlit as st
import torch

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

ACCENT_THEMES = {
    "Emerald (default)": {"accent": "34, 197, 94", "accent2": "59, 130, 246"},
    "Royal Blue": {"accent": "59, 130, 246", "accent2": "168, 85, 247"},
    "Amethyst": {"accent": "168, 85, 247", "accent2": "236, 72, 153"},
    "Amber / Gold": {"accent": "217, 119, 6", "accent2": "34, 197, 94"},
    "Rose": {"accent": "236, 72, 153", "accent2": "59, 130, 246"},
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
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }}
        .subtle-text {{
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }}
        .answer-box {{
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: rgba(var(--accent), 0.08);
            border: 1px solid rgba(var(--accent), 0.25);
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }}
        .source-box {{
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background: rgba(var(--accent2), 0.06);
            border: 1px solid rgba(var(--accent2), 0.18);
        }}
        .small-muted {{
            color: #6b7280;
            font-size: 0.85rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: rgba(var(--accent), 0.15);
            color: rgba(var(--accent), 1);
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 0.3rem;
            margin-bottom: 0.3rem;
        }}
        .pill-stat {{
            padding: 0.6rem 0.9rem;
            border-radius: 12px;
            background: rgba(var(--accent2), 0.08);
            border: 1px solid rgba(var(--accent2), 0.18);
            text-align: center;
        }}
        .feedback-row button {{
            margin-right: 0.25rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOAD SAVED FAISS INDEX
# ============================================================
def load_vector_store():
    if not os.path.exists(FAISS_DIR):
        raise FileNotFoundError(
            f"FAISS folder '{FAISS_DIR}' not found.\n"
            f"Expected files:\n"
            f"- {FAISS_DIR}/index.faiss\n"
            f"- {FAISS_DIR}/index.pkl"
        )

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = FAISS.load_local(
        FAISS_DIR,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_store


# ============================================================
# LOAD FLAN-T5 MODEL + TOKENIZER
# ============================================================
def initialize_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    return tokenizer, model


# ============================================================
# PROMPT TEMPLATE
# ============================================================
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
- If possible, structure the answer in short paragraphs or bullet points.

User Question:
{query}

Relevant Quran Context:
{context}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["query", "context"]
    )


# ============================================================
# CLEAN / FORMAT SOURCES
# ============================================================
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_to_relevance(score, all_scores):
    """Convert a FAISS L2 distance into a relative 0-100 relevance score
    among the docs returned for this query. Lower distance = higher relevance."""
    if not all_scores:
        return 0
    lo, hi = min(all_scores), max(all_scores)
    if hi == lo:
        return 100
    # invert so smaller distance -> higher percentage
    pct = 100 * (1 - (score - lo) / (hi - lo))
    return max(5, min(100, round(pct)))


def format_sources(docs, scores=None):
    sources = []
    seen = set()
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

        sources.append({
            "id": i,
            "content": text,
            "score": score,
            "badges": badges,
        })

    relevances = [s["score"] for s in sources if s["score"] is not None]
    for s in sources:
        if s["score"] is not None:
            s["relevance"] = score_to_relevance(s["score"], relevances)
        else:
            s["relevance"] = None

    return sources


# ============================================================
# GENERATE RESPONSE FROM FLAN-T5
# ============================================================
def generate_answer(tokenizer, model, prompt_text, max_new_tokens=220):
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()


# ============================================================
# RETRIEVAL + GENERATION
# ============================================================
def retrieve_and_generate(vector_store, tokenizer, model, prompt, query, k=4, max_new_tokens=220):
    # similarity_search_with_score gives us a distance we can turn into a
    # relevance indicator for the UI.
    results = vector_store.similarity_search_with_score(query, k=k)

    if not results:
        return "No relevant Quran verses were found for this question.", [], "", 0.0

    docs = [doc for doc, _ in results]
    scores = [score for _, score in results]

    context_chunks = []
    for doc in docs:
        chunk = clean_text(doc.page_content)
        if len(chunk) > 1400:
            chunk = chunk[:1400]
        context_chunks.append(chunk)

    context = "\n\n".join(context_chunks)
    final_prompt = prompt.format(query=query, context=context)

    start = time.time()
    answer = generate_answer(
        tokenizer=tokenizer,
        model=model,
        prompt_text=final_prompt,
        max_new_tokens=max_new_tokens
    )
    elapsed = time.time() - start

    sources = format_sources(docs, scores)
    return answer, sources, context, elapsed


# ============================================================
# FOLLOW-UP SUGGESTIONS (lightweight keyword heuristic)
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
# SESSION STATE
# ============================================================
def init_session():
    defaults = {
        "messages": [],
        "selected_question": "",
        "last_result": None,
        "bookmarks": [],
        "accent_theme": "Emerald (default)",
        "pending_query": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# CORE: RUN A QUERY END TO END
# ============================================================
def run_query(query, vector_store, tokenizer, model, prompt, k, max_tokens):
    try:
        answer, sources, context, elapsed = retrieve_and_generate(
            vector_store=vector_store,
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            query=query,
            k=k,
            max_new_tokens=max_tokens
        )
    except Exception as e:
        st.error(f"Error while generating response: {e}")
        return

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "query": query,
        "elapsed": elapsed,
        "feedback": None,
        "bookmarked": False,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    st.session_state.last_result = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "context": context,
        "elapsed": elapsed,
    }

    st.toast(f"Answer ready in {elapsed:.1f}s")


# ============================================================
# SAMPLE QUESTIONS (CATEGORIZED + SURPRISE ME)
# ============================================================
def render_sample_questions(vector_store, tokenizer, model, prompt, k, max_tokens):
    st.markdown("### 🗂️ Browse questions by topic")
    st.caption("Tap any question to get an instant answer — no extra click needed.")

    top_l, top_r = st.columns([3, 1])
    with top_r:
        if st.button("🎲 Surprise me", use_container_width=True):
            random_q = random.choice(ALL_SAMPLE_QUESTIONS)
            run_query(random_q, vector_store, tokenizer, model, prompt, k, max_tokens)
            st.rerun()

    tabs = st.tabs(list(QUESTION_BANK.keys()))
    for tab, (category, questions) in zip(tabs, QUESTION_BANK.items()):
        with tab:
            cols = st.columns(2)
            for i, q in enumerate(questions):
                with cols[i % 2]:
                    if st.button(q, use_container_width=True, key=f"sample_{category}_{q}"):
                        run_query(q, vector_store, tokenizer, model, prompt, k, max_tokens)
                        st.rerun()


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")

        k = st.slider("Retrieved passages", min_value=2, max_value=8, value=4, step=1)
        max_tokens = st.slider("Answer length", min_value=80, max_value=400, value=220, step=20)
        show_sources_default = st.toggle("Show sources by default", value=False)
        show_context = st.toggle("Show retrieved context block", value=False)

        st.divider()
        st.subheader("🎨 Appearance")
        theme_name = st.selectbox(
            "Accent theme",
            list(ACCENT_THEMES.keys()),
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
                "⬇️ Export chat (.md)",
                data=export_chat_markdown(),
                file_name="quran_chatbot_history.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.divider()
        st.subheader("ℹ️ About")
        st.caption("Model: google/flan-t5-base")
        st.caption("Embeddings: all-MiniLM-L6-v2")
        st.caption("Index: local FAISS")

    return k, max_tokens, show_sources_default, show_context


# ============================================================
# EXPORT CHAT AS MARKDOWN
# ============================================================
def export_chat_markdown():
    lines = ["# Quran Chatbot — Conversation Export", ""]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lines.append(f"## ❓ {msg['content']}")
        else:
            lines.append(msg["content"])
            if msg.get("sources"):
                lines.append("\n**Sources:**")
                for src in msg["sources"]:
                    lines.append(f"- {src['content'][:200]}")
            lines.append("")
    return "\n".join(lines)


# ============================================================
# SOURCE RENDERING (shared by chat history + result panel)
# ============================================================
def render_source_card(src, show_expanded=False):
    preview = src["content"][:140] + ("..." if len(src["content"]) > 140 else "")
    with st.expander(f"Source {src['id']} — {preview}", expanded=show_expanded):
        if src.get("badges"):
            badge_html = "".join(f"<span class='badge'>{b}</span>" for b in src["badges"])
            st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown(
            f"<div class='source-box'>{src['content']}</div>",
            unsafe_allow_html=True
        )
        if src.get("relevance") is not None:
            st.caption(f"Relative relevance: {src['relevance']}%")
            st.progress(src["relevance"] / 100)


# ============================================================
# FEEDBACK / BOOKMARK ROW FOR A MESSAGE
# ============================================================
def render_feedback_row(msg, msg_idx, vector_store, tokenizer, model, prompt, k, max_tokens):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 4])

    with c1:
        up_label = "👍" if msg.get("feedback") != "up" else "✅👍"
        if st.button(up_label, key=f"up_{msg_idx}"):
            msg["feedback"] = "up"
            st.rerun()

    with c2:
        down_label = "👎" if msg.get("feedback") != "down" else "✅👎"
        if st.button(down_label, key=f"down_{msg_idx}"):
            msg["feedback"] = "down"
            st.rerun()

    with c3:
        is_bookmarked = msg.get("bookmarked", False)
        star_label = "⭐" if not is_bookmarked else "🌟 Saved"
        if st.button(star_label, key=f"bookmark_{msg_idx}"):
            msg["bookmarked"] = not is_bookmarked
            if msg["bookmarked"]:
                st.session_state.bookmarks.append({
                    "query": msg["query"],
                    "answer": msg["content"],
                    "sources": msg.get("sources", []),
                    "timestamp": msg.get("timestamp", ""),
                })
            else:
                st.session_state.bookmarks = [
                    b for b in st.session_state.bookmarks if b["query"] != msg["query"]
                ]
            st.rerun()

    with c4:
        if st.button("🔁", key=f"regen_{msg_idx}", help="Regenerate this answer"):
            run_query(msg["query"], vector_store, tokenizer, model, prompt, k, max_tokens)
            st.rerun()

    with c5:
        if msg.get("elapsed"):
            st.caption(f"⏱️ {msg['elapsed']:.1f}s · {msg.get('timestamp', '')}")


# ============================================================
# RENDER PREVIOUS CHAT
# ============================================================
def render_chat_history(show_sources_default, vector_store, tokenizer, model, prompt, k, max_tokens):
    if not st.session_state.messages:
        st.info("No conversation yet — ask a question above or tap a sample topic to get started.")
        return

    search_term = st.text_input("🔍 Search chat history", placeholder="Filter by keyword...")

    st.markdown("## 💬 Chat History")
    for idx, msg in enumerate(st.session_state.messages):
        if search_term:
            haystack = msg["content"].lower()
            if search_term.lower() not in haystack:
                continue

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                if msg.get("sources"):
                    with st.expander("📚 View sources", expanded=show_sources_default):
                        for src in msg["sources"]:
                            render_source_card(src, show_expanded=show_sources_default)

                render_feedback_row(msg, idx, vector_store, tokenizer, model, prompt, k, max_tokens)

                follow_ups = suggest_follow_ups(msg.get("query", ""), n=3)
                if follow_ups:
                    st.caption("💡 You might also ask:")
                    fcols = st.columns(len(follow_ups))
                    for fc, fq in zip(fcols, follow_ups):
                        with fc:
                            if st.button(fq, key=f"followup_{idx}_{fq}", use_container_width=True):
                                run_query(fq, vector_store, tokenizer, model, prompt, k, max_tokens)
                                st.rerun()


# ============================================================
# RESULT CARD
# ============================================================
def render_result(latest, show_sources_default=False, show_context=False):
    st.success(f"Answer generated successfully in {latest.get('elapsed', 0):.1f}s.")

    colA, colB, colC = st.columns(3)
    colA.metric("Retrieved sources", len(latest["sources"]))
    colB.metric("Question length", len(latest["query"].split()))
    colC.metric("Answer length", len(latest["answer"].split()))

    tab1, tab2, tab3 = st.tabs(["🟢 Answer", "📚 Sources", "🧠 Retrieval Context"])

    with tab1:
        st.markdown(
            f"<div class='answer-box'>{latest['answer']}</div>",
            unsafe_allow_html=True
        )

    with tab2:
        if not latest["sources"]:
            st.info("No sources found.")
        else:
            st.caption("Click a source to inspect the retrieved Quran/Tafsir chunk and its relevance.")
            for src in latest["sources"]:
                render_source_card(src, show_expanded=show_sources_default)

    with tab3:
        if show_context:
            st.text_area("Retrieved context used for generation", latest["context"], height=300)
        else:
            st.info("Enable **'Show retrieved context block'** from the sidebar to inspect the full RAG context.")


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
    avg_len = round(sum(len(m["content"].split()) for m in assistant_msgs) / total_q, 1) if total_q else 0
    avg_time = round(sum(m.get("elapsed", 0) for m in assistant_msgs) / total_q, 2) if total_q else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in zip(
        (c1, c2, c3, c4, c5),
        ("Questions asked", "Bookmarks", "👍 / 👎", "Avg. answer length", "Avg. response time"),
        (total_q, total_bookmarks, f"{thumbs_up} / {thumbs_down}", f"{avg_len} words", f"{avg_time}s"),
    ):
        with col:
            st.markdown(f"<div class='pill-stat'><b>{value}</b><br><span class='small-muted'>{label}</span></div>", unsafe_allow_html=True)

    if total_q:
        st.markdown("### Answer length over the conversation")
        st.bar_chart([len(m["content"].split()) for m in assistant_msgs])

        st.markdown("### Topics asked")
        topic_counts = {}
        for m in assistant_msgs:
            q = m.get("query", "")
            for category, questions in QUESTION_BANK.items():
                if q in questions:
                    topic_counts[category] = topic_counts.get(category, 0) + 1
        if topic_counts:
            st.bar_chart(topic_counts)
        else:
            st.caption("Ask one of the suggested topic questions to populate this chart.")
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
            if b.get("sources"):
                with st.expander(f"📚 {len(b['sources'])} source(s)"):
                    for src in b["sources"]:
                        render_source_card(src)
            if st.button("🗑️ Remove bookmark", key=f"remove_bookmark_{i}"):
                st.session_state.bookmarks.pop(i)
                st.rerun()


# ============================================================
# ABOUT TAB
# ============================================================
def render_about_tab():
    st.markdown("## ℹ️ About this chatbot")
    st.markdown(
        """
        This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer
        questions about the Quran:

        1. Your question is embedded with **all-MiniLM-L6-v2**.
        2. The most relevant passages are retrieved from a local **FAISS** index.
        3. **google/flan-t5-base** generates an answer grounded strictly in the
           retrieved passages.

        **New in this version:**
        - 🗂️ Topic-organized sample questions that answer instantly when tapped
        - 🎲 "Surprise me" random question button
        - ⭐ Bookmark your favorite answers
        - 👍 👎 Feedback on each answer
        - 🔁 One-click regenerate for any answer
        - 💡 Follow-up question suggestions
        - 🔍 Search across your chat history
        - 📊 A live stats dashboard for the session
        - ⬇️ Export the full conversation as Markdown
        - 🎨 Selectable accent color themes
        - Relevance bars showing how closely each source matches your question

        **Disclaimer:** This tool is for educational exploration only and is not
        a substitute for guidance from qualified Islamic scholars.
        """
    )


# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(
        page_title="Quran Chatbot",
        page_icon="📖",
        layout="wide"
    )

    init_session()
    theme = ACCENT_THEMES[st.session_state.accent_theme]
    inject_custom_css(accent=theme["accent"], accent2=theme["accent2"])

    st.markdown("<div class='main-title'>📖 Quran Chatbot</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle-text'>Ask questions about Islam and the Quran using a local FAISS index + FLAN-T5 answer generation.</div>",
        unsafe_allow_html=True
    )

    # sidebar settings
    k, max_tokens, show_sources_default, show_context = render_sidebar()

    # initialize models
    try:
        with st.spinner("Loading Quran index and language model..."):
            vector_store, tokenizer, model, prompt = initialize_models()
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()

    main_tab, bookmarks_tab, stats_tab, about_tab = st.tabs(
        ["💬 Chat", "⭐ Bookmarks", "📊 Stats", "ℹ️ About"]
    )

    with main_tab:
        render_sample_questions(vector_store, tokenizer, model, prompt, k, max_tokens)
        st.divider()

        default_query = st.session_state.selected_question if st.session_state.selected_question else ""
        query = st.text_input(
            "Ask your question",
            value=default_query,
            placeholder="e.g. What does the Quran say about patience?"
        )

        col1, col2 = st.columns([1, 5])
        ask_clicked = col1.button("Ask", type="primary", use_container_width=True)
        clear_input_clicked = col2.button("Clear current question", use_container_width=False)

        if clear_input_clicked:
            st.session_state.selected_question = ""
            st.rerun()

        should_run = ask_clicked and query.strip()

        if should_run:
            with st.status("Running Quran retrieval and answer generation...", expanded=True) as status:
                st.write("🔎 Searching the FAISS index for relevant Quran passages...")
                st.write("✍️ Generating the final answer from the retrieved context...")
                run_query(query, vector_store, tokenizer, model, prompt, k, max_tokens)
                status.update(label="Completed successfully", state="complete")
            st.session_state.selected_question = ""

        render_chat_history(show_sources_default, vector_store, tokenizer, model, prompt, k, max_tokens)

        if st.session_state.last_result:
            st.divider()
            st.markdown("## 🆕 Latest Result")
            latest = st.session_state.last_result
            st.markdown(f"**Question:** {latest['query']}")
            with st.expander("✅ Click to view the generated answer", expanded=True):
                render_result(latest, show_sources_default=show_sources_default, show_context=show_context)

    with bookmarks_tab:
        render_bookmarks_tab()

    with stats_tab:
        render_stats_tab()

    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
