import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import re
import warnings
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer, util

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="e-Nurse",
    page_icon="⚕",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .chat-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .chat-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #35408e;
        margin-bottom: 0.2rem;
    }
    .chat-header p {
        font-size: 0.95rem;
        color: #6b7280;
        margin-top: 0;
    }

    .disclaimer {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #92400e;
        margin-bottom: 1rem;
    }

    .stChatMessage {
        border-radius: 16px;
        padding: 0.5rem 1rem;
    }

    .badge {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        margin-bottom: 6px;
    }

    .urgency-banner {
        background: #fee2e2;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        font-size: 0.82rem;
        color: #991b1b;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .confidence-badge {
        display: inline-block;
        background: #f0fdf4;
        color: #166534;
        font-size: 0.7rem;
        padding: 1px 8px;
        border-radius: 999px;
        margin-bottom: 6px;
    }
    .urgency-badge {
        display: inline-block;
        font-size: 0.7rem;
        padding: 1px 8px;
        border-radius: 999px;
        margin-bottom: 6px;
    }
    .urgency-high { background: #fef2f2; color: #991b1b; }
    .urgency-mod  { background: #fffbeb; color: #92400e; }
    .urgency-low  { background: #f0fdf4; color: #166534; }

    .sidebar-info {
        font-size: 0.85rem;
        color: #4b5563;
    }
    .sidebar-info b {
        color: #0f766e;
    }

    .stTextInput > div > div > input {
        border-radius: 12px;
        font-size: 0.95rem;
    }

    /* Follow-up suggestion chips */
    .followup-label {
        font-size: 0.78rem;
        color: #6b7280;
        margin-bottom: 4px;
        font-weight: 500;
    }
    /* Justify assistant chat responses */
    [data-testid="stChatMessage"][aria-label="assistant"] p,
    [data-testid="stChatMessage"][aria-label="assistant"] li {
        text-align: justify;
        line-height: 1.6;
        margin-bottom: 0.6rem;
    }
    div[data-testid="stButton"] > button {
        background-color: #f0f9ff;
        color: #0369a1;
        border: 1px solid #bae6fd;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #0ea5e9;
        color: white;
        border-color: #0ea5e9;
    }
    /* ── Sidebar Blue Theme (#35408e) ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #35408e 0%, #2a3270 100%);
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] h3,
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] li {
        color: #e0e7ff !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] strong {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] a {
        color: #93c5fd !important;
        text-decoration: none;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] a:hover {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stImage"] img {
        filter: brightness(0) invert(1);
    }
    /* Sidebar buttons */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        padding: 4px 10px;
        font-size: 0.72rem;
        line-height: 1.2;
        border-radius: 16px;
        min-height: 0;
        background: rgba(255,255,255,0.15);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.25);
        font-weight: 500;
        transition: all 0.2s;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: #ffffff;
        color: #35408e;
        border-color: #ffffff;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
        gap: 0.2rem;
    }
    /* Clear chat button - red accent */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:last-of-type > button {
        background: rgba(220,38,38,0.2);
        color: #fca5a5;
        border-color: rgba(220,38,38,0.3);
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"]:last-of-type > button:hover {
        background: #dc2626;
        color: #ffffff;
        border-color: #dc2626;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>⚕ e-(NU)rse</h1>
    <p>Your campus health information and guidance assistant</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚕ <b>Disclaimer:</b> This tool provides general medical information from verified health datasets. 
    It is <b>NOT</b> a substitute for professional medical diagnosis or prescription. 
    Please visit the school clinic or consult a physician for individual health evaluations.
</div>
""", unsafe_allow_html=True)

# ── Search defaults (no user-facing settings) ─────────────────
threshold = 0.60
max_words = 1000

# ── Sidebar ──────────────────────────────────────────────────
GUIDE_TOPICS = [
    ("Asthma", "What causes Asthma ?"),
    ("Acne", "What is (are) Acne ?"),
    ("Anxiety", "What is (are) Anxiety Disorders ?"),
    ("Depression", "What are the symptoms of Depression ?"),
    ("Headache", "What causes Headache ?"),
    ("Back Pain", "What are the treatments for Back Pain ?"),
    ("Sleep Apnea", "What are the symptoms of Sleep Apnea ?"),
    ("Concussion", "What is (are) Concussion ?"),
    ("Sports Injuries", "What are the treatments for Sports Injuries ?"),
    ("Low Vision", "Who is at risk for Low Vision ?"),
]

with st.sidebar:
    
    st.markdown(
        "A RAG Health guidance assistant built on the "
        "[MedQuAD](https://github.com/abachaa/MedQuAD) dataset\n "
    )
    st.markdown(
        "- TF-IDF+N-gram and Linear SVC: Intent Prediction (main model).\n"
        "- SBERT: sentence lookup in the dataset."
    )
    st.markdown("**Objectives:**")
    st.markdown(
        "- Low cost; No LLM, No Limit.\n"
        "- Suggest follow-up questions."
    )
    st.markdown("**Limitations:**")
    st.markdown(
        "- Limited to information of dataset (13k records).\n"
    )
    st.markdown("---")

    st.markdown("### Quick Topics")
    # 2-column grid layout for compact display
    for i in range(0, len(GUIDE_TOPICS), 2):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            topic, query = GUIDE_TOPICS[i]
            if st.button(topic, key=f"guide_{i}"):
                st.session_state.pending_query = query
                st.rerun()
        if i + 1 < len(GUIDE_TOPICS):
            with col2:
                topic, query = GUIDE_TOPICS[i + 1]
                if st.button(topic, key=f"guide_{i+1}"):
                    st.session_state.pending_query = query
                    st.rerun()

    st.markdown("---")
    st.markdown("### Settings")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown('<p style="color:#c7d2fe; font-size:0.82rem; margin:0;">Dev: <b style="color:#ffffff;">Manalo, Caleb</b></p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#c7d2fe; font-size:0.82rem; margin:0;">Dev: <b style="color:#ffffff;">Paule, Moises</b></p>', unsafe_allow_html=True)
# ── Urgency Lexicon & Conversational Filter ───────────────────
URGENCY_LEXICON = {
    # High urgency (score: -2)
    "emergency": -2, "severe": -2, "bleeding": -2, "overdose": -2,
    "chest pain": -2, "difficulty breathing": -2, "can't breathe": -2,
    "unconscious": -2, "seizure": -2, "stroke": -2, "heart attack": -2,
    "anaphylaxis": -2, "suicide": -2, "self-harm": -2, "paralysis": -2,
    "fainting": -2, "sudden": -2, "rapid": -2, "acute": -2,

    # Moderate urgency (score: -1)
    "pain": -1, "help": -1, "chronic": -1, "fever": -1, "infection": -1,
    "swelling": -1, "dizzy": -1, "nausea": -1, "vomiting": -1,
    "diarrhea": -1, "cough": -1, "wheezing": -1, "rash": -1,
    "injury": -1, "fracture": -1, "burn": -1, "headache": -1,
    "fatigue": -1, "weakness": -1, "numbness": -1, "blurred vision": -1,
    "sore": -1, "throbbing": -1, "cramps": -1, "spasm": -1,
    "inflamed": -1, "infected": -1, "wound": -1, "lesion": -1,
    "anxiety": -1, "depression": -1, "insomnia": -1, "palpitations": -1,

    # Mild / improving (score: +1)
    "relieved": 1, "better": 1, "manageable": 1, "improving": 1,
    "mild": 1, "stable": 1, "recovering": 1, "healing": 1,
    "tolerable": 1, "subside": 1, "subsided": 1, "normal": 1,
}

GREETINGS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "who are you", "what can you do", "help", "kumusta", "magandang araw"
}

def is_greeting(text: str) -> bool:
    clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    return clean in GREETINGS

def score_urgency(text: str, lexicon=URGENCY_LEXICON) -> int:
    text_lower = text.lower()
    total = 0

    # Check multi-word phrases first (longest match)
    multi_word = {k: v for k, v in lexicon.items() if " " in k}
    for phrase, score in sorted(multi_word.items(), key=lambda x: -len(x[0])):
        if phrase in text_lower:
            total += score

    # Check single words (skip punctuation)
    words = text_lower.split()
    for word in words:
        clean = re.sub(r'[^\w\s]', '', word)
        if clean in lexicon:
            total += lexicon[clean]

    return total

# ── Text cleanup ─────────────────────────────────────────────
def clean_response(text: str) -> str:
    text = re.sub(r'\(Watch the video.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(To enlarge the video.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'See this graphic.*?(?=\.)\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'See a glossary.*?(?=\.)\.?', '', text, flags=re.IGNORECASE)
    # Preserve double newlines as paragraph breaks, collapse single whitespace
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    # Convert single newlines to space (only within paragraphs)
    paragraphs = text.split('\n\n')
    text = '\n\n'.join(p.replace('\n', ' ') for p in paragraphs)
    return text.strip()


def semantic_truncate(text: str, query: str, embedder, max_words: int = 200,
                      min_words: int = 50, truncation_stride: int = 5) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    query_embedding = embedder.encode(query, convert_to_tensor=True)
    best_truncation = text
    highest_similarity = -1

    for i in range(len(words) - min_words, max_words - 1, -truncation_stride):
        current_truncated_text = " ".join(words[:i])
        if not current_truncated_text:
            continue
        truncated_embedding = embedder.encode(current_truncated_text, convert_to_tensor=True)
        similarity = util.cos_sim(query_embedding, truncated_embedding).item()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_truncation = current_truncated_text

    if len(best_truncation.split()) > max_words:
        return " ".join(words[:max_words]) + "..."
    return best_truncation + "..."

# ── Load artifacts (cached) ──────────────────────────────────
@st.cache_resource
def load_assets():
    base = os.path.dirname(os.path.abspath(__file__))
    intent_model = joblib.load(os.path.join(base, "intent_pipeline_model.joblib"))
    corpus_embeddings = joblib.load(os.path.join(base, "corpus_embeddings.joblib"))

    # Load the pre-split training data directly (matches corpus_embeddings row-for-row)
    train_df = pd.read_csv(os.path.join(base, "train_data.csv")).reset_index(drop=True)

    # Load SBERT fresh from HuggingFace (pickled version is incompatible)
    sbert_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    return intent_model, sbert_embedder, corpus_embeddings, train_df


with st.spinner("Loading medical models..."):
    intent_model, sbert_embedder, corpus_embeddings, train_df = load_assets()

# ── Response Logic ───────────────────────────────────────────
def get_response(user_query: str, threshold: float = 0.60, max_words: int = 200):
    clean_query = user_query.strip().lower()

    # Step 1: Guardrail for small talk and greetings
    if is_greeting(clean_query):
        return {
            "intent": "general_greeting",
            "confidence": 1.0,
            "urgency_score": 0,
            "matched_focus": "General Assistant",
            "response": (
                "Hello! I am **e-Nurse**, your school health guidance assistant. "
                "How can I help you today? You can ask me about common symptoms, health conditions, or exams."
            ),
        }

    # Step 2: Score urgency
    urgency_score = score_urgency(user_query)

    # Step 3: Predict intent
    predicted_intent = intent_model.predict([user_query])[0]

    # Step 4: Dense search query formulation (matches notebook: intent + query)
    search_query = f"{predicted_intent} {user_query}"
    query_vec = sbert_embedder.encode(search_query, convert_to_tensor=True)

    # Step 5: Semantic search using util.cos_sim (matches notebook)
    scores = util.cos_sim(query_vec, corpus_embeddings)[0]
    best_idx = int(scores.argmax().item())
    best_score = float(scores[best_idx].item())

    # Step 6: Strict confidence guardrail
    if best_score < threshold:
        return {
            "intent": predicted_intent,
            "confidence": best_score,
            "urgency_score": urgency_score,
            "matched_focus": None,
            "response": (
                "I couldn't find a sufficiently reliable match in our verified medical database. "
                "For accurate guidance, please describe your question in more detail or visit the school clinic."
            ),
        }

    # Step 7: Fetch and format verified document
    matched_row = train_df.iloc[best_idx]
    raw_response = str(matched_row["response"])
    clean_raw = clean_response(raw_response)
    final_answer = semantic_truncate(clean_raw, user_query, sbert_embedder, max_words=max_words)

    # Step 8: Urgency intervention
    if urgency_score < -1:
        final_answer = (
            "⚠ **Urgent Note:** Your query indicates potential severe symptoms. "
            "Please proceed to the nearest school clinic or emergency room immediately.\n\n"
            + final_answer
        )

    return {
        "intent": predicted_intent,
        "confidence": best_score,
        "urgency_score": urgency_score,
        "matched_focus": str(matched_row.get("focus_area", "")),
        "response": final_answer,
    }


# ── Follow-up question generator ─────────────────────────────
def generate_follow_ups(focus: str, current_intent: str) -> list[str]:
    """Suggest follow-ups using real queries from the dataset for the same topic."""
    if not focus:
        return []

    # Find other rows with the same focus_area but different intent
    same_topic = train_df[
        (train_df["focus_area"] == focus) & (train_df["intent"] != current_intent)
    ]

    if same_topic.empty:
        return []

    # Pick up to 3 unique queries from different intents
    seen = set()
    suggestions = []
    for _, row in same_topic.iterrows():
        q = str(row["query"]).strip()
        q_lower = q.lower()
        if q_lower not in seen and len(q) < 80:
            seen.add(q_lower)
            suggestions.append(q)
        if len(suggestions) >= 3:
            break

    return suggestions

# ── Chat History Rendering ───────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "focus" in msg and msg["focus"]:
            urgency = msg.get("urgency", 0)
            urg_cls = "urgency-high" if urgency <= -2 else "urgency-mod" if urgency < 0 else "urgency-low"
            st.markdown(
                f'<span class="badge">📋 {msg["focus"]}</span> '
                f'<span class="badge">{msg["intent"]}</span> '
                f'<span class="confidence-badge">confidence: {msg["confidence"]:.2f}</span> '
                f'<span class="urgency-badge {urg_cls}">urgency: {urgency}</span>',
                unsafe_allow_html=True,
            )
        # Render response with paragraph breaks
        response_html = msg["content"].replace('\n\n', '</p><p>')
        st.markdown(f'<p>{response_html}</p>', unsafe_allow_html=True)

# ── User Input & Processing ──────────────────────────────────
pending = st.session_state.pop("pending_query", None)
prompt = st.chat_input("Ask a health or symptom question...")

# Override with pending follow-up if any
if pending:
    prompt = pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving verified health information..."):
            result = get_response(prompt, threshold, max_words)

        focus = result["matched_focus"]
        urgency = result["urgency_score"]
        if focus:
            urg_cls = "urgency-high" if urgency <= -2 else "urgency-mod" if urgency < 0 else "urgency-low"
            st.markdown(
                f'<span class="badge">📋 {focus}</span> '
                f'<span class="badge">{result["intent"]}</span> '
                f'<span class="confidence-badge">confidence: {result["confidence"]:.2f}</span> '
                f'<span class="urgency-badge {urg_cls}">urgency: {urgency}</span>',
                unsafe_allow_html=True,
            )

        # Render response with paragraph breaks
        response_html = result["response"].replace('\n\n', '</p><p>')
        st.markdown(f'<p>{response_html}</p>', unsafe_allow_html=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "focus": focus,
            "intent": result["intent"],
            "confidence": result["confidence"],
            "urgency": result["urgency_score"],
        })

# ── Follow-up suggestions (outside chat messages) ───────────
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "assistant" and last_msg.get("focus") and last_msg.get("intent") != "general_greeting":
        follow_ups = generate_follow_ups(last_msg["focus"], last_msg["intent"])
        if follow_ups:
            st.markdown('<p class="followup-label">Suggested follow-ups:</p>', unsafe_allow_html=True)
            cols = st.columns(len(follow_ups))
            for i, (col, suggestion) in enumerate(zip(cols, follow_ups)):
                with col:
                    if st.button(suggestion, key=f"fu_{len(st.session_state.messages)}_{i}"):
                        st.session_state.pending_query = suggestion
                        st.rerun()