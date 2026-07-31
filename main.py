"""
Agentic Document Understanding System
======================================

Architecture
------------
Instead of a fixed pipeline that always runs every analysis step in the
same order, this version gives an LLM orchestrator a toolbox of functions
(summarize, extract keywords, classify, evaluate, chunk long documents,
compare documents, etc.) and lets it DECIDE which tools to call, in what
order, and when to stop — using Google Gemini's native function/tool
calling loop (via the google-genai SDK). Gemini has a generous free tier,
so this runs without a paid API key.

Every tool call the model makes is logged and shown to the user as an
"Agent Trace", so you can see the reasoning path, not just the output.

If no API key is provided, the app falls back to a deterministic
rule-based pipeline (clearly labeled as fallback mode) so it still works
without a key — it just isn't agentic in that mode.

Install: pip install streamlit pdfplumber pymupdf python-docx scikit-learn pandas google-genai
Run:     streamlit run app.py
"""

import re

import pandas as pd
import pdfplumber
import streamlit as st
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Agentic Document Understanding System",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 88% !important; padding-left: 3rem !important; padding-right: 3rem !important; }
    div[data-testid="stAlert"] { width: 100% !important; max-width: 100% !important; }
    .agent-trace-step {
        border-left: 3px solid #6c5ce7;
        padding: 0.4rem 0.8rem;
        margin-bottom: 0.4rem;
        background: rgba(108, 92, 231, 0.06);
        border-radius: 0 6px 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("📄 Project Info")
st.sidebar.write("**Agentic Document Understanding System**")
st.sidebar.write(
    "An LLM orchestrator autonomously chooses which analysis tools to run "
    "on each document — extraction, summarization, keyword extraction, "
    "classification, chunking for long docs, and cross-document comparison "
    "— rather than following a fixed script."
)

st.sidebar.write("### Supported Files")
st.sidebar.write("- PDF\n- TXT\n- DOCX")

st.sidebar.divider()
st.sidebar.write("### Agent Mode")
st.sidebar.write(
    "Enter a Google Gemini API key (free tier available at "
    "aistudio.google.com/apikey) to enable the agent orchestrator. "
    "Without a key, the app runs a deterministic rule-based fallback "
    "pipeline instead (no autonomous tool selection)."
)

gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    help="Used only for this session; not stored.",
)

llm_model = st.sidebar.text_input(
    "Gemini Model",
    value="gemini-2.5-flash",
    help="Check aistudio.google.com or ai.google.dev for the current list of "
    "available model names if this one errors out.",
)

max_agent_turns = st.sidebar.slider(
    "Max agent tool-call rounds",
    min_value=2,
    max_value=10,
    value=6,
    help="Safety cap on how many rounds of tool calls the agent can make per document.",
)

agent_mode_active = bool(gemini_api_key) and genai is not None
ANALYSIS_CACHE_VERSION = 6

st.title("📄 Agentic Document Understanding System")
st.write(
    "Upload PDF, TXT, or DOCX documents. An LLM agent decides which analysis "
    "tools to run on each document and in what order, then produces a "
    "structured final analysis."
)

if agent_mode_active:
    st.success("🤖 Agent mode active — the LLM orchestrator will decide the analysis plan per document.")
else:
    st.warning(
        "⚙️ Deterministic fallback mode — enter a Gemini API key in the sidebar to enable "
        "autonomous agent orchestration."
    )

with st.expander("ℹ️ How does the agent work?"):
    st.write(
        """
        1. The document is extracted and cleaned.
        2. The LLM orchestrator is given a **toolbox**: rule-based summary, keyword
           extraction, classification, rule-based evaluation, document chunking
           (for long documents), and cross-document comparison.
        3. The model decides for itself which tools it needs, in which order —
           for example, it may chunk a long document before summarizing it, or
           skip classification if the category is already obvious from the summary.
        4. Every tool call is logged as an **Agent Trace** step, visible below each document.
        5. The agent finishes by calling a `finalize_analysis` tool with a structured
           result (summary, keywords, category, confidence, quality score, and its
           own reasoning) — including the option to **override** the rule-based
           category if it disagrees, with justification.
        6. Without an API key, a deterministic rule-based pipeline runs instead,
           clearly labeled as fallback mode.
        """
    )


# -----------------------------
# TEXT EXTRACTION
# -----------------------------

def extract_page_text_column_aware(page):
    """
    pdfplumber's default extract_text() reads lines roughly top-to-bottom, which
    scrambles two-column layouts (common in academic papers) by interleaving
    fragments from both columns mid-sentence. This detects a likely two-column
    layout and, if found, extracts the left column fully, then the right column,
    instead of reading across both.
    """
    words = page.extract_words()
    if not words:
        return page.extract_text() or ""

    mid = page.width / 2
    left_words = [w for w in words if w["x1"] <= mid + 5]
    right_words = [w for w in words if w["x0"] >= mid - 5]
    crossing_words = [w for w in words if w not in left_words and w not in right_words]

    looks_like_two_columns = (
        len(left_words) > 10
        and len(right_words) > 10
        and len(crossing_words) < 0.15 * len(words)
    )

    if not looks_like_two_columns:
        return page.extract_text() or ""

    left_crop = page.crop((0, 0, mid, page.height))
    right_crop = page.crop((mid, 0, page.width, page.height))
    left_text = left_crop.extract_text() or ""
    right_text = right_crop.extract_text() or ""
    return (left_text + "\n" + right_text).strip()


def extract_text_from_pdf_pdfplumber(file):
    """Fallback extractor: pdfplumber with a column-aware heuristic for two-column body text."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = extract_page_text_column_aware(page)
                if page_text:
                    text += page_text + "\n"
    except Exception as error:
        st.error(f"PDF text extraction error: {error}")
        return ""
    return text


def extract_text_from_pdf(file):
    # PyMuPDF's block-based reading-order algorithm handles complex layouts (title
    # pages with multiple text boxes, tables, multi-column body text) far more
    # reliably than pdfplumber's line-clustering approach, which can splice
    # unrelated text fragments together mid-word/mid-sentence on such pages
    # (e.g. cover pages with a title block, author list, and guide name positioned
    # as separate text boxes rather than one flowing column).
    if fitz is not None:
        try:
            file_bytes = file.read()
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_parts = []
            for page in pdf_doc:
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text)
            pdf_doc.close()
            text = "\n".join(text_parts)
            if text.strip():
                return text
        except Exception as error:
            st.warning(f"PyMuPDF extraction failed, falling back to pdfplumber: {error}")
        finally:
            try:
                file.seek(0)
            except Exception:
                pass

    return extract_text_from_pdf_pdfplumber(file)


def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception as error:
        st.error(f"TXT text extraction error: {error}")
        return ""


def extract_text_from_docx(file):
    text = ""
    try:
        document = Document(file)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as error:
        st.error(f"DOCX text extraction error: {error}")
        return ""
    return text


def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file)
    return ""


def preprocess_text(text):
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    # Drop standalone page-number lines (e.g. a lone "16" from a page footer/header) —
    # left in place, these glue onto the sentence or heading before/after them once
    # newlines get collapsed during summarization, corrupting both.
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$\n?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# -----------------------------
# CORE ANALYSIS FUNCTIONS
# (these become the agent's TOOLS)
# -----------------------------

SUMMARY_SECTION_BOUNDARIES = [
    "abstract",
    "introduction",
    "background",
    "methodology",
    "proposed system",
    "system architecture",
    "implementation",
    "results",
    "evaluation",
    "discussion",
    "conclusion",
    "future scope",
    "monitoring agent",
    "trend and risk agent",
    "intervention agent",
    "response protocol",
    "privacy",
]

SUMMARY_SOURCE_HEADINGS = {
    "abstract": 3.0,
    "executive summary": 3.0,
    "overview": 2.6,
    "introduction": 2.5,
    "problem statement": 2.4,
    "objective": 2.4,
    "objectives": 2.4,
    "proposed system": 2.5,
    "system architecture": 2.2,
    "methodology": 2.0,
    "implementation": 1.7,
    "results": 1.8,
    "evaluation": 1.8,
    "discussion": 1.7,
    "conclusion": 2.3,
    "future scope": 1.5,
}

SUMMARY_STOP_HEADINGS = {
    "acknowledgement",
    "acknowledgements",
    "certificate",
    "declaration",
    "references",
    "bibliography",
    "appendix",
    "table of contents",
    "list of figures",
    "list of tables",
}

SUMMARY_INCOMPLETE_STARTS = {
    "ction",
    "tions",
    "sion",
    "ment",
    "ments",
    "ility",
    "ities",
    "ance",
    "ences",
}

SUMMARY_CAPTION_PATTERNS = [
    r"\brisk score\s*(?:of|:)?\s*\d+\s*/\s*\d+",
    r"\bscore\s*:\s*\d+",
    r"\bfigure\s+\d+",
    r"\btable\s+\d+",
    r"\bstatus panel\b",
    r"\bdashboard\b",
]

SUMMARY_BOILERPLATE_PATTERNS = [
    r"\bproject report\b",
    r"\bsubmitted in partial\b",
    r"\bpartial fulfil",
    r"\bbachelor of\b",
    r"\bdegree of\b",
    r"\bunder the guidance\b",
    r"\bdepartment of\b",
    r"\baffiliated\b",
    r"\buniversity\b",
    r"\bprofessor\b",
    r"\backnowledg",
    r"\bcertificate\b",
    r"\bdeclaration\b",
]


def split_long_summary_fragment(fragment, max_words=60):
    words = fragment.split()
    if len(words) <= max_words:
        return [fragment]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        if end < len(words):
            lower_bound = max(start + 25, end - 15)
            for split_at in range(end, lower_bound, -1):
                if re.search(r"[,;:]$", words[split_at - 1]):
                    end = split_at
                    break
        chunks.append(" ".join(words[start:end]))
        start = end
    return chunks


def clean_summary_fragment(fragment):
    fragment = re.sub(r"\s+", " ", fragment)
    fragment = re.sub(r"\s+([,.;:!?])", r"\1", fragment)
    fragment = re.sub(r"([.!?]){2,}", r"\1", fragment)
    return fragment.strip(" -,:;")


def boilerplate_hit_count(fragment):
    lower = fragment.lower()
    return sum(1 for pattern in SUMMARY_BOILERPLATE_PATTERNS if re.search(pattern, lower))


def normalize_heading_text(line):
    line = re.sub(r"^\s*(chapter\s+)?\d+(\.\d+)*\s*", "", line.lower())
    line = re.sub(r"[^a-z ]+", " ", line)
    return re.sub(r"\s+", " ", line).strip()


def match_summary_heading(line):
    normalized = normalize_heading_text(line)
    if not normalized:
        return None

    for heading in SUMMARY_SOURCE_HEADINGS:
        if normalized == heading:
            return heading
        if normalized.startswith(heading + " ") and len(normalized.split()) <= 12:
            return heading

    for heading in SUMMARY_STOP_HEADINGS:
        if normalized == heading:
            return heading

    return None


def looks_like_unrecognized_heading(line):
    """
    Detects short, title/caps-style lines that are almost certainly a section
    heading even if it's not in our known heading list (e.g. "RESEARCH GAP",
    "4.2 Testing Approach"). Used to end the current section's line-accumulation
    at ANY heading boundary, not just recognized ones — otherwise an unrecognized
    heading's line gets appended as body text, gluing it onto the surrounding
    sentence and corrupting both.
    """
    stripped = line.strip()
    if not stripped or stripped[-1] in ".!?,;:":
        return False

    words = stripped.split()
    if not words or len(words) > 8:
        return False

    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    if not alpha_words:
        return False

    def is_heading_case(word):
        letters = re.sub(r"[^A-Za-z]", "", word)
        if not letters:
            return True
        return letters.isupper() or letters[0].isupper()

    return all(is_heading_case(w) for w in alpha_words)


def select_summary_source_text(text, max_words=1800):
    lines = [clean_summary_fragment(line) for line in text.splitlines()]
    sections = []
    current_heading = None
    current_lines = []

    def flush_section():
        if current_heading and current_lines and current_heading in SUMMARY_SOURCE_HEADINGS:
            section_text = " ".join(current_lines)
            sections.append((current_heading, section_text))

    for line in lines:
        if not line:
            continue

        heading = match_summary_heading(line)
        if heading:
            flush_section()
            current_lines = []
            current_heading = heading
            continue

        if current_heading in SUMMARY_SOURCE_HEADINGS and looks_like_unrecognized_heading(line):
            # An unrecognized heading (not in our whitelist) still ends the current
            # section — otherwise its line gets glued onto the surrounding sentence.
            flush_section()
            current_lines = []
            current_heading = None
            continue

        if current_heading in SUMMARY_SOURCE_HEADINGS:
            current_lines.append(line)

    flush_section()

    if not sections:
        return text

    sections.sort(key=lambda item: SUMMARY_SOURCE_HEADINGS[item[0]], reverse=True)
    selected = []
    selected_words = 0
    seen_headings = set()
    for heading, section_text in sections:
        if heading in seen_headings:
            continue
        seen_headings.add(heading)

        section_words = section_text.split()
        if not section_words:
            continue

        remaining = max_words - selected_words
        if remaining <= 0:
            break

        selected.append(" ".join(section_words[:remaining]))
        selected_words += min(len(section_words), remaining)

    return "\n".join(selected) if selected else text


DANGLING_END_WORDS = {
    "and", "or", "but", "the", "a", "an", "of", "in", "on", "with", "to", "for",
    "as", "by", "at", "from", "that", "which", "is", "are", "was", "were",
}


def looks_like_broken_pdf_fragment(fragment):
    words = fragment.split()
    if not words:
        return True

    first = re.sub(r"[^A-Za-z]", "", words[0]).lower()
    last = re.sub(r"[^A-Za-z]", "", words[-1]).lower()

    if first in SUMMARY_INCOMPLETE_STARTS:
        return True
    if words[0].endswith(",") and first not in {"however", "therefore", "moreover", "finally"}:
        return True
    if last and len(last) <= 2 and last.upper() not in {"ai", "ml", "ui"}:
        return True
    if re.search(r"\b[a-z]{1,2}\.$", fragment):
        return True
    if last in DANGLING_END_WORDS:
        # A sentence that ends "...transparent, and." is truncated — a real
        # sentence never legitimately ends on a bare connective/article.
        return True

    lower = fragment.lower()
    if any(re.search(pattern, lower) for pattern in SUMMARY_CAPTION_PATTERNS):
        return True

    return False


def summary_relevance_score(fragment):
    lower = fragment.lower()
    signals = [
        "aim",
        "objective",
        "propose",
        "present",
        "develop",
        "design",
        "system",
        "framework",
        "method",
        "architecture",
        "agent",
        "monitor",
        "detect",
        "classif",
        "analy",
        "evaluate",
        "result",
        "privacy",
        "patient",
        "health",
        "risk",
        "guidance",
    ]
    return sum(1 for signal in signals if signal in lower)


def is_useful_summary_candidate(fragment):
    words = fragment.split()
    if len(words) < 8 or len(words) > 80:
        return False

    if looks_like_broken_pdf_fragment(fragment):
        return False

    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    if len(alpha_words) / max(len(words), 1) < 0.65:
        return False

    short_word_ratio = sum(1 for word in alpha_words if len(word) <= 2) / max(len(alpha_words), 1)
    if short_word_ratio > 0.35:
        return False

    digit_ratio = sum(1 for char in fragment if char.isdigit()) / max(len(fragment), 1)
    if digit_ratio > 0.16:
        return False

    if boilerplate_hit_count(fragment) >= 2:
        return False

    return True


# Detects a page-break artifact glued mid-sentence: a PDF page's footer/header
# (an optional page number followed by a short run of ALL-CAPS words, e.g. a
# running section title like "RESEARCH GAP") that got extracted with no
# punctuation separating it from the surrounding prose, because the original
# sentence spanned a page boundary. Endpoint-only checks (first/last word of a
# fragment) can't catch this since the artifact sits in the *middle* of an
# otherwise clean-looking fragment — this forces a sentence break there instead.
GENERIC_HEADING_INJECTION_PATTERN = re.compile(
    r"(?<=[a-z0-9,;:\)])\s+(?:\d{1,3}\s+)?[A-Z]{2,}(?:\s+[A-Z]{2,}){1,4}(?=\s+[A-Z][a-z])"
)


def build_summary_candidates(text):
    working = GENERIC_HEADING_INJECTION_PATTERN.sub(". ", text)

    for heading in SUMMARY_SECTION_BOUNDARIES:
        working = re.sub(
            rf"\s+({re.escape(heading)})\b",
            r". \1",
            working,
            flags=re.IGNORECASE,
        )

    fragments = re.split(r"(?<=[.!?])\s+|\n+|(?:\s+\*\s+)", working)
    candidates = []
    seen = set()
    for fragment in fragments:
        fragment = clean_summary_fragment(fragment)
        if not fragment:
            continue
        for piece in split_long_summary_fragment(fragment):
            piece = clean_summary_fragment(piece)
            key = piece.lower()
            if key in seen or not is_useful_summary_candidate(piece):
                continue
            seen.add(key)
            candidates.append(piece)
    return candidates


def format_summary_sentence(fragment):
    fragment = clean_summary_fragment(fragment)
    if not fragment:
        return ""
    fragment = fragment[0].upper() + fragment[1:]
    if fragment[-1] not in ".!?":
        fragment += "."
    return fragment


def limit_summary_words(summary, max_words=140):
    summary = clean_summary_fragment(summary)
    words = summary.split()
    if len(words) <= max_words:
        return summary

    sentences = re.split(r"(?<=[.!?])\s+", summary)
    selected = []
    selected_word_count = 0
    for sentence in sentences:
        sentence = clean_summary_fragment(sentence)
        if not sentence:
            continue
        sentence_word_count = len(sentence.split())
        if selected and selected_word_count + sentence_word_count > max_words:
            break
        if not selected and sentence_word_count > max_words:
            return " ".join(words[:max_words]).rstrip(" ,;:") + "..."
        selected.append(format_summary_sentence(sentence))
        selected_word_count += sentence_word_count

    if selected:
        return " ".join(selected)
    return " ".join(words[:max_words]).rstrip(" ,;:") + "..."


def generate_simple_summary(text, max_sentences=3):
    try:
        max_sentences = int(max_sentences)
    except (TypeError, ValueError):
        max_sentences = 3
    max_sentences = max(1, min(max_sentences, 3))
    source_text = select_summary_source_text(text)
    candidate_sentences = build_summary_candidates(source_text)
    if len(candidate_sentences) < max_sentences and source_text != text:
        candidate_sentences = build_summary_candidates(text)

    if not candidate_sentences:
        return "The text is too short to generate a meaningful summary."

    if len(candidate_sentences) <= max_sentences:
        return limit_summary_words(" ".join(format_summary_sentence(sentence) for sentence in candidate_sentences))

    try:
        # Score each sentence by how information-dense it is (sum of TF-IDF weights
        # of its words against the rest of the document), rather than just taking
        # the first few sentences — this picks the most representative content
        # from across the whole document, not just the introduction.
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95)
        tfidf_matrix = vectorizer.fit_transform(candidate_sentences)
        raw_scores = tfidf_matrix.sum(axis=1).A1
        candidate_lengths = [len(sentence.split()) for sentence in candidate_sentences]
        scores = []
        for index, raw_score in enumerate(raw_scores):
            score = raw_score / (candidate_lengths[index] ** 0.35)
            relevance = summary_relevance_score(candidate_sentences[index])
            score += relevance * 0.25
            if relevance == 0:
                score *= 0.6
            boilerplate_hits = boilerplate_hit_count(candidate_sentences[index])
            if boilerplate_hits:
                score *= 0.35 ** boilerplate_hits
            scores.append(score)

        # Greedy Maximal Marginal Relevance selection: pick the highest-scoring
        # sentence first, then repeatedly pick whichever remaining sentence best
        # balances its own score against similarity to what's already selected.
        # Plain top-N by score tends to pick several near-duplicate sentences on
        # documents that repeat the same core terminology throughout (e.g. every
        # top sentence being about "the multi-agent architecture") — this spreads
        # picks across different parts/aspects of the document instead.
        max_score = max(scores) if scores else 0.0
        normalized_scores = [s / max_score if max_score > 0 else 0.0 for s in scores]
        similarity_matrix = cosine_similarity(tfidf_matrix)
        mmr_lambda = 0.75  # weight toward relevance (1.0) vs. diversity (0.0)

        selected_indices = []
        remaining_indices = set(range(len(candidate_sentences)))
        for _ in range(min(max_sentences, len(candidate_sentences))):
            best_index, best_mmr_value = None, float("-inf")
            for i in remaining_indices:
                redundancy = max((similarity_matrix[i][j] for j in selected_indices), default=0.0)
                mmr_value = mmr_lambda * normalized_scores[i] - (1 - mmr_lambda) * redundancy
                if mmr_value > best_mmr_value:
                    best_index, best_mmr_value = i, mmr_value
            selected_indices.append(best_index)
            remaining_indices.discard(best_index)

        top_indices = sorted(selected_indices)  # restore original reading order for coherence

        return limit_summary_words(" ".join(format_summary_sentence(candidate_sentences[i]) for i in top_indices))
    except Exception:
        return limit_summary_words(" ".join(format_summary_sentence(sentence) for sentence in candidate_sentences[:max_sentences]))


def extract_keywords(text, top_n=8):
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return [k for k, _ in keyword_scores]
    except Exception:
        words = re.findall(r"\b\w+\b", text.lower())
        words = [w for w in words if len(w) > 4]
        unique_words = []
        for w in words:
            if w not in unique_words:
                unique_words.append(w)
        return unique_words[:top_n]


CATEGORY_KEYWORDS = {
    "Academic Paper": ["abstract", "methodology", "references", "literature review", "experiment", "research paper", "hypothesis", "citation"],
    "Lecture Note": ["lecture", "chapter", "course", "lesson", "definition", "example", "concept", "explains", "basic concepts", "notes", "öğrenme", "ders", "konu"],
    "Report": ["report", "analysis", "findings", "recommendation", "summary", "result", "conclusion", "rapor"],
    "Business Document": ["business", "market", "customer", "sales", "strategy", "company", "marketing", "revenue"],
    "Technical Document": ["system", "software", "algorithm", "model", "database", "architecture", "implementation", "api", "python", "machine learning", "artificial intelligence", "dataset", "datasets", "classification", "regression", "prediction", "training", "testing", "evaluation", "supervised learning", "unsupervised learning", "data", "model evaluation"],
    "News Article": ["news", "announced", "reported", "according to", "said", "breaking", "article"],
}


def classify_document(text):
    text_lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += 2 if category == "Technical Document" else 1
        scores[category] = score
    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "Other", "No strong category keywords were found."
    return best_category, f"The document contains keywords related to {best_category}."


def evaluate_output(text, summary, keywords, category):
    score = 0
    feedback = []

    if len(text.split()) > 20:
        score += 1
        feedback.append("Text extraction is successful.")
    else:
        feedback.append("Extracted text is too short.")

    if summary and "too short" not in summary.lower():
        score += 1
        feedback.append("Summary is generated.")
    else:
        feedback.append("Summary quality is weak.")

    if len(keywords) >= 3:
        score += 1
        feedback.append("Keywords are extracted.")
    else:
        feedback.append("Not enough keywords are extracted.")

    if category != "Other":
        score += 1
        feedback.append("Document category is identified.")
    else:
        feedback.append("Document category is uncertain.")

    if len(text.split()) > 50:
        score += 1
        feedback.append("Document has enough content for analysis.")
    else:
        feedback.append("Document is short, so analysis may be limited.")

    return score, feedback


def get_quality_label(score):
    if score == 5:
        return "🟢 Excellent"
    elif score == 4:
        return "🟢 Good"
    elif score == 3:
        return "🟡 Acceptable"
    return "🔴 Needs Improvement"


def chunk_text(text, chunk_size_words=250):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size_words]) for i in range(0, len(words), chunk_size_words)]
    return chunks


def interpret_similarity(score):
    if score >= 0.70:
        return "High Similarity"
    elif score >= 0.40:
        return "Medium Similarity"
    return "Low Similarity"


def calculate_pair_similarity(text_a, text_b):
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        score = round(float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]), 3)
        return score, interpret_similarity(score)
    except Exception:
        return None, "Could not be computed"


def calculate_similarity_matrix(documents):
    texts = [doc["clean_text"] for doc in documents]
    names = [doc["file_name"] for doc in documents]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                score = round(similarity_matrix[i][j], 2)
                rows.append({
                    "Document 1": names[i],
                    "Document 2": names[j],
                    "Similarity Score": score,
                    "Interpretation": interpret_similarity(score),
                })
        return pd.DataFrame(rows)
    except Exception as error:
        st.warning(f"Similarity analysis could not be completed: {error}")
        return pd.DataFrame()


# -----------------------------
# AGENT ORCHESTRATOR
# -----------------------------

# Gemini function-declaration schemas (OpenAPI-subset dicts; type names are
# uppercase per google.genai.types.Type).
TOOL_DECLARATIONS_RAW = [
    {
        "name": "get_rule_based_summary",
        "description": "Generate an extractive (rule-based) summary of the document by pulling its most informative sentences.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "max_sentences": {"type": "INTEGER", "description": "How many sentences to include, default 3."}
            },
            "required": [],
        },
    },
    {
        "name": "get_keywords",
        "description": "Extract the top TF-IDF keywords from the document.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "top_n": {"type": "INTEGER", "description": "How many keywords to return, default 8."}
            },
            "required": [],
        },
    },
    {
        "name": "classify_document_rule_based",
        "description": "Classify the document into a category using keyword matching (Academic Paper, Lecture Note, Report, Business Document, Technical Document, News Article, or Other).",
        "parameters": {"type": "OBJECT", "properties": {}, "required": []},
    },
    {
        "name": "rule_based_evaluate",
        "description": "Score the analysis quality (0-5) using rule-based heuristics on text length, summary, keywords, and category.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "summary": {"type": "STRING", "description": "The summary text to evaluate."},
                "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
                "category": {"type": "STRING"},
            },
            "required": ["summary", "keywords", "category"],
        },
    },
    {
        "name": "chunk_document",
        "description": "Split a long document into word chunks and get a lightweight overview (chunk count + a short preview of the first couple of chunks). Use this before summarizing if the document is long (e.g. over ~800 words). To read a specific chunk's full text, call get_chunk afterward with a chunk_index.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "chunk_size_words": {"type": "INTEGER", "description": "Words per chunk, default 500."}
            },
            "required": [],
        },
    },
    {
        "name": "get_chunk",
        "description": "Retrieve the full text of one specific chunk by index (0-based), after chunk_document has been called. Use this selectively — only for chunks you actually need to read in detail, not all of them.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "chunk_index": {"type": "INTEGER", "description": "0-based index of the chunk to retrieve."}
            },
            "required": ["chunk_index"],
        },
    },
    {
        "name": "compare_with_document",
        "description": "Compute TF-IDF cosine similarity between the current document and another already-uploaded document by file name. Only useful when multiple documents are uploaded.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "other_file_name": {"type": "STRING", "description": "Exact file name of the other uploaded document."}
            },
            "required": ["other_file_name"],
        },
    },
    {
        "name": "finalize_analysis",
        "description": "Submit the final structured analysis for this document. Call this exactly once, as your last action.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "summary": {"type": "STRING", "description": "Final summary in your own words: 2-3 specific sentences, maximum 140 words. Focus on the document's actual purpose, method, and findings; ignore title-page metadata, names, roll numbers, acknowledgements, and other boilerplate unless central to the content."},
                "keywords": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Final list of keywords."},
                "category": {"type": "STRING", "description": "Final chosen category."},
                "category_overridden": {"type": "BOOLEAN", "description": "True if you disagreed with the rule-based classifier's category."},
                "override_reason": {"type": "STRING", "description": "If overridden, why. Empty string otherwise."},
                "quality_score": {"type": "INTEGER", "description": "Your own 0-5 quality/confidence assessment of the analysis."},
                "reasoning": {"type": "STRING", "description": "Brief explanation of the plan you followed and why (2-4 sentences)."},
            },
            "required": ["summary", "keywords", "category", "category_overridden", "override_reason", "quality_score", "reasoning"],
        },
    },
]


def build_gemini_tools():
    declarations = [genai_types.FunctionDeclaration(**decl) for decl in TOOL_DECLARATIONS_RAW]
    return [genai_types.Tool(function_declarations=declarations)]

SYSTEM_PROMPT = """You are an autonomous document-analysis agent. You have a toolbox of \
functions for analyzing a document: rule-based summarization, keyword extraction, \
classification, evaluation, chunking for long documents (chunk_document + get_chunk), \
and comparison against other uploaded documents.

Decide for yourself which tools to call and in what order. You do not need to call every \
tool. For example:
- If the document is long (roughly 800+ words), call chunk_document first to see how many \
chunks there are, then selectively call get_chunk on a small number of representative \
chunks (e.g. the first, middle, and last, or a handful spread across the document) — do \
NOT try to read every chunk one by one, that wastes calls and produces a bloated trace. \
Use the chunk previews plus your own judgment to synthesize a summary that covers the \
whole document, not just the truncated excerpt you were given directly.
- Final summaries must be concise and specific: 2-3 sentences, maximum 140 words. \
Do not paste raw extracted text, title pages, acknowledgements, names, roll numbers, \
or institutional boilerplate unless those details are central to the user's content.
- If the rule-based classifier's category looks wrong given the content, you may override it \
in your final answer, but you must explain why.
- Only call compare_with_document if multiple documents are available and comparison is useful.
- Be efficient: prefer a small number of well-chosen tool calls over exhaustively calling \
everything.

When you are done, call finalize_analysis exactly once with your final structured result. \
Do not call finalize_analysis until you have gathered enough information to be confident."""


def run_agent(client, model_name, doc_index, documents, max_turns):
    """
    Runs the tool-calling agent loop for documents[doc_index] using Gemini.
    Returns (final_result_dict, trace_list).
    """
    current_doc = documents[doc_index]
    text = current_doc["clean_text"]
    other_docs = {d["file_name"]: d["clean_text"] for i, d in enumerate(documents) if i != doc_index}

    trace = []
    chunk_state = {"chunks": None}

    def dispatch(name, args):
        if name == "get_rule_based_summary":
            result = generate_simple_summary(text, args.get("max_sentences", 3))
        elif name == "get_keywords":
            result = extract_keywords(text, args.get("top_n", 8))
        elif name == "classify_document_rule_based":
            category, reason = classify_document(text)
            result = {"category": category, "reason": reason}
        elif name == "rule_based_evaluate":
            score, feedback = evaluate_output(text, args.get("summary", ""), list(args.get("keywords", [])), args.get("category", ""))
            result = {"score": score, "feedback": feedback}
        elif name == "chunk_document":
            chunks = chunk_text(text, args.get("chunk_size_words", 500))
            chunk_state["chunks"] = chunks
            preview_count = min(2, len(chunks))
            result = {
                "num_chunks": len(chunks),
                "preview_of_first_chunks": [chunks[i][:200] for i in range(preview_count)],
                "note": "Call get_chunk(chunk_index) to read a specific chunk's full text if needed.",
            }
        elif name == "get_chunk":
            chunks = chunk_state["chunks"]
            if not chunks:
                result = {"error": "No chunks available yet — call chunk_document first."}
            else:
                idx = int(args.get("chunk_index", 0))
                if idx < 0 or idx >= len(chunks):
                    result = {"error": f"chunk_index {idx} out of range (0-{len(chunks) - 1})."}
                else:
                    result = {"chunk_index": idx, "text": chunks[idx][:1500]}
        elif name == "compare_with_document":
            other_name = args.get("other_file_name")
            if other_name not in other_docs:
                result = {"error": f"'{other_name}' not found among uploaded documents.", "available": list(other_docs.keys())}
            else:
                score, interpretation = calculate_pair_similarity(text, other_docs[other_name])
                result = {"similarity_score": score, "interpretation": interpretation}
        else:
            result = {"error": f"Unknown tool: {name}"}
        return result

    word_count = len(text.split())
    tools = build_gemini_tools()
    config = genai_types.GenerateContentConfig(tools=tools, system_instruction=SYSTEM_PROMPT)

    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=(
                f"Document: {current_doc['file_name']}\n"
                f"Word count: {word_count}\n"
                f"Other uploaded documents available for comparison: {list(other_docs.keys()) or 'none'}\n\n"
                f"Document text (may be truncated):\n{text[:6000]}"
            ))],
        )
    ]

    final_result = None

    for turn in range(max_turns):
        response = client.models.generate_content(model=model_name, contents=contents, config=config)

        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content else []
        function_calls = [p for p in parts if getattr(p, "function_call", None)]

        if not function_calls:
            trace.append({"step": turn + 1, "type": "message", "content": getattr(response, "text", "") or ""})
            break

        # Echo the model's turn (including its function_call parts) back into the conversation.
        contents.append(candidate.content)

        response_parts = []
        for part in function_calls:
            name = part.function_call.name
            args = dict(part.function_call.args) if part.function_call.args else {}

            if name == "finalize_analysis":
                final_result = {
                    "summary": limit_summary_words(str(args.get("summary", ""))),
                    "keywords": [str(k) for k in args.get("keywords", [])],
                    "category": str(args.get("category", "Other")),
                    "category_overridden": bool(args.get("category_overridden", False)),
                    "override_reason": str(args.get("override_reason", "")),
                    "quality_score": int(args.get("quality_score", 0)),
                    "reasoning": str(args.get("reasoning", "")),
                }
                trace.append({"step": turn + 1, "type": "finalize", "tool": name, "args": final_result})
                response_parts.append(genai_types.Part.from_function_response(name=name, response={"status": "received"}))
                continue

            result = dispatch(name, args)
            trace.append({"step": turn + 1, "type": "tool_call", "tool": name, "args": args, "result": result})
            response_parts.append(genai_types.Part.from_function_response(name=name, response={"result": result}))

        contents.append(genai_types.Content(role="user", parts=response_parts))

        if final_result is not None:
            break

    if final_result is None:
        # Agent ran out of turns without finalizing — build a safe fallback from whatever tools ran.
        summary = generate_simple_summary(text)
        keywords = extract_keywords(text)
        category, _ = classify_document(text)
        final_result = {
            "summary": summary,
            "keywords": keywords,
            "category": category,
            "category_overridden": False,
            "override_reason": "",
            "quality_score": 2,
            "reasoning": "Agent did not finalize within the turn limit; used a safe deterministic fallback.",
        }
        trace.append({"step": len(trace) + 1, "type": "fallback", "content": "Turn limit reached before finalize_analysis."})

    return final_result, trace


def run_deterministic_pipeline(current_doc):
    """Fallback path used when no API key / no google-genai package is available."""
    text = current_doc["clean_text"]
    # Keep fallback summaries concise while filtering noisy PDF boilerplate.
    summary_sentence_count = 3
    summary = generate_simple_summary(text, max_sentences=summary_sentence_count)
    keywords = extract_keywords(text)
    category, reason = classify_document(text)
    score, feedback = evaluate_output(text, summary, keywords, category)
    return {
        "summary": summary,
        "keywords": keywords,
        "category": category,
        "category_overridden": False,
        "override_reason": "",
        "quality_score": score,
        "reasoning": f"Deterministic fallback pipeline (no agent). {reason}",
    }, feedback


# -----------------------------
# CROSS-DOCUMENT SEARCH INDEX (for chat)
# -----------------------------

def build_search_index(processed_docs, chunk_size_words=200):
    """Builds one TF-IDF index over chunks from ALL uploaded documents, for chat retrieval."""
    chunks_meta = []
    for doc in processed_docs:
        chunks = chunk_text(doc["clean_text"], chunk_size_words)
        for i, c in enumerate(chunks):
            chunks_meta.append({"file_name": doc["file_name"], "chunk_index": i, "text": c})

    if not chunks_meta:
        return None

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c["text"] for c in chunks_meta])
    return {"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks_meta}


def query_search_index(index, query, top_k=5):
    if not index:
        return []
    query_vec = index["vectorizer"].transform([query])
    scores = cosine_similarity(query_vec, index["matrix"])[0]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in ranked:
        c = index["chunks"][i]
        results.append({
            "file_name": c["file_name"],
            "chunk_index": c["chunk_index"],
            "score": round(float(scores[i]), 3),
            "preview": c["text"][:300],
        })
    return results


# -----------------------------
# CHAT AGENT (chat with your documents)
# -----------------------------

CHAT_TOOL_DECLARATIONS_RAW = [
    {
        "name": "list_documents",
        "description": "List all uploaded documents along with their existing analysis: word count, category, keywords, and summary. Use this first for questions about which documents exist or their high-level content.",
        "parameters": {"type": "OBJECT", "properties": {}, "required": []},
    },
    {
        "name": "search_documents",
        "description": "Keyword/TF-IDF search across all uploaded documents. Returns the most relevant passages with document name, chunk index, similarity score, and a short preview. Use this to locate where an answer might be before reading full text.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Search query, phrased as the key terms of the user's question."},
                "top_k": {"type": "INTEGER", "description": "How many top matching passages to return, default 5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_chunk_text",
        "description": "Retrieve the full text of a specific passage found via search_documents, by file name and chunk index.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_name": {"type": "STRING", "description": "Exact file name, as returned by search_documents or list_documents."},
                "chunk_index": {"type": "INTEGER", "description": "0-based chunk index, as returned by search_documents."},
            },
            "required": ["file_name", "chunk_index"],
        },
    },
]


def build_chat_tools():
    declarations = [genai_types.FunctionDeclaration(**decl) for decl in CHAT_TOOL_DECLARATIONS_RAW]
    return [genai_types.Tool(function_declarations=declarations)]


CHAT_SYSTEM_PROMPT = """You are a helpful assistant answering questions about a set of \
uploaded documents. You have three tools: list_documents (existing analysis: category, \
keywords, summary for each document), search_documents (find relevant passages across all \
documents by query), and get_chunk_text (read the full text of one specific passage).

Ground every answer in the documents. For questions about specific content, call \
search_documents first, then get_chunk_text on the most promising 1-3 results if the \
preview isn't enough detail — don't fetch every result. For questions about what documents \
exist or their general topic, list_documents is usually enough on its own.

Always mention which document(s) your answer is based on by file name. If the documents \
don't contain the answer, say so plainly rather than guessing. Be concise. Once you have \
enough information, just respond in plain text — there is no special "finalize" tool for \
chat, a normal text reply ends your turn."""


def run_chat_agent(client, model_name, user_question, chat_contents, search_idx, processed_docs, max_turns):
    """
    Runs one turn of the document-chat agent. Mutates chat_contents in place (appends the
    new turns) so conversation memory persists across calls. Returns (answer_text, trace).
    """
    trace = []

    def dispatch(name, args):
        if name == "list_documents":
            result = [
                {
                    "file_name": d["file_name"],
                    "word_count": len(d["clean_text"].split()),
                    "category": d["result"]["category"],
                    "keywords": d["result"]["keywords"],
                    "summary": d["result"]["summary"][:300],
                }
                for d in processed_docs
                if d.get("clean_text")
            ]
        elif name == "search_documents":
            result = query_search_index(search_idx, args.get("query", ""), args.get("top_k", 5))
        elif name == "get_chunk_text":
            file_name = args.get("file_name")
            idx = int(args.get("chunk_index", 0))
            match = None
            if search_idx:
                match = next(
                    (c for c in search_idx["chunks"] if c["file_name"] == file_name and c["chunk_index"] == idx),
                    None,
                )
            if not match:
                result = {"error": f"No chunk {idx} found for '{file_name}'."}
            else:
                result = {"text": match["text"][:1500]}
        else:
            result = {"error": f"Unknown tool: {name}"}
        return result

    tools = build_chat_tools()
    config = genai_types.GenerateContentConfig(tools=tools, system_instruction=CHAT_SYSTEM_PROMPT)

    chat_contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=user_question)]))

    answer_text = ""
    for turn in range(max_turns):
        response = client.models.generate_content(model=model_name, contents=chat_contents, config=config)

        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content else []
        function_calls = [p for p in parts if getattr(p, "function_call", None)]

        if not function_calls:
            answer_text = getattr(response, "text", "") or "I couldn't generate a response."
            if candidate and candidate.content:
                chat_contents.append(candidate.content)
            break

        chat_contents.append(candidate.content)

        response_parts = []
        for part in function_calls:
            name = part.function_call.name
            args = dict(part.function_call.args) if part.function_call.args else {}
            result = dispatch(name, args)
            trace.append({"step": turn + 1, "tool": name, "args": args, "result": result})
            response_parts.append(genai_types.Part.from_function_response(name=name, response={"result": result}))

        chat_contents.append(genai_types.Content(role="user", parts=response_parts))

    if not answer_text:
        answer_text = "I ran out of tool-call turns before finishing — try asking a more specific question."

    return answer_text, trace


# -----------------------------
# REPORT GENERATION
# -----------------------------

def generate_analysis_report(file_name, clean_text, result, trace, mode):
    report = f"""DOCUMENT ANALYSIS REPORT
========================
Mode: {mode}

File Name:
{file_name}

Basic Document Information:
- Text Length: {len(clean_text)} characters
- Word Count: {len(clean_text.split())} words

Final Category: {result['category']}
Category Overridden By Agent: {result['category_overridden']}
Override Reason: {result['override_reason'] or 'N/A'}

Final Summary:
{result['summary']}

Final Keywords:
{", ".join(result['keywords'])}

Quality Score: {result['quality_score']}/5

Agent Reasoning:
{result['reasoning']}

Agent Trace:
"""
    for step in trace:
        if step["type"] == "tool_call":
            report += f"- Step {step['step']}: called {step['tool']}({step['args']}) -> {step['result']}\n"
        elif step["type"] == "finalize":
            report += f"- Step {step['step']}: called finalize_analysis\n"
        elif step["type"] == "fallback":
            report += f"- Step {step['step']}: {step['content']}\n"
        else:
            report += f"- Step {step['step']}: message\n"

    report += "\nEnd of Report\n========================\n"
    return report


def generate_similarity_report(similarity_df):
    report = "SIMILARITY ANALYSIS REPORT\n==========================\n\n"
    if similarity_df.empty:
        report += "No similarity result was generated.\n"
    else:
        for _, row in similarity_df.iterrows():
            report += f"Document 1: {row['Document 1']}\n"
            report += f"Document 2: {row['Document 2']}\n"
            report += f"Similarity Score: {row['Similarity Score']}\n"
            report += f"Interpretation: {row['Interpretation']}\n"
            report += "--------------------------\n"
    report += "\nEnd of Similarity Report\n========================\n"
    return report


# -----------------------------
# STREAMLIT INTERFACE
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

    # Pass 1: extract & clean all documents up front so the agent can
    # reference other documents (e.g. for comparison) while processing each one.
    documents = []
    for uploaded_file in uploaded_files:
        extracted_text = extract_text(uploaded_file)
        clean_text = preprocess_text(extracted_text)
        documents.append({"file_name": uploaded_file.name, "clean_text": clean_text})

    client = genai.Client(api_key=gemini_api_key) if agent_mode_active else None

    # Cache analysis + chat state in session_state, keyed on the uploaded fileset and
    # settings, so re-running the (expensive) per-document agent pipeline only happens
    # when the documents or mode actually change — not on every chat message.
    uploaded_signature = tuple((f.name, f.size) for f in uploaded_files)
    settings_changed = (
        st.session_state.get("doc_signature") != uploaded_signature
        or st.session_state.get("doc_agent_mode") != agent_mode_active
        or st.session_state.get("doc_model") != llm_model
        or st.session_state.get("analysis_cache_version") != ANALYSIS_CACHE_VERSION
    )

    if settings_changed or st.session_state.get("processed_docs") is None:
        st.session_state.doc_signature = uploaded_signature
        st.session_state.doc_agent_mode = agent_mode_active
        st.session_state.doc_model = llm_model
        st.session_state.analysis_cache_version = ANALYSIS_CACHE_VERSION
        st.session_state.chat_history = []
        st.session_state.chat_contents = []

        progress_status = st.status("Analyzing documents...", expanded=False)

        for index, current_doc in enumerate(documents):
            clean_text = current_doc["clean_text"]

            if not clean_text:
                current_doc["result"] = None
                current_doc["trace"] = []
                current_doc["mode_label"] = "N/A"
                continue

            progress_status.write(f"Processing {current_doc['file_name']}...")

            if agent_mode_active:
                try:
                    result, trace = run_agent(client, llm_model, index, documents, max_agent_turns)
                    mode_label = "Agent mode (autonomous tool-calling)"
                except Exception as error:
                    progress_status.write(f"⚠️ Agent run failed for {current_doc['file_name']}, using fallback: {error}")
                    result, feedback = run_deterministic_pipeline(current_doc)
                    trace = []
                    mode_label = "Deterministic fallback (agent error)"
            else:
                result, feedback = run_deterministic_pipeline(current_doc)
                trace = []
                mode_label = "Deterministic fallback (no API key)"

            current_doc["result"] = result
            current_doc["trace"] = trace
            current_doc["mode_label"] = mode_label

        progress_status.update(label="Analysis complete.", state="complete")

        st.session_state.processed_docs = documents
        st.session_state.search_index = build_search_index(
            [d for d in documents if d["clean_text"]]
        )
    else:
        documents = st.session_state.processed_docs

    # -----------------------------
    # RENDER (always from cached results, never re-triggers the agent)
    # -----------------------------

    st.header("📌 Analysis Results")

    for index, current_doc in enumerate(documents):
        st.divider()
        st.subheader(f"📄 File: {current_doc['file_name']}")

        clean_text = current_doc["clean_text"]
        result = current_doc.get("result")
        trace = current_doc.get("trace") or []
        mode_label = current_doc.get("mode_label", "N/A")

        if not clean_text or result is None:
            st.warning("No readable text could be extracted from this file.")
            continue

        quality_label = get_quality_label(min(max(int(result.get("quality_score", 0)), 0), 5))

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Basic Document Info")
            st.write(f"**Mode:** {mode_label}")
            st.write(f"**Text length:** {len(clean_text)} characters")
            st.write(f"**Word count:** {len(clean_text.split())} words")

            st.write("### Category")
            st.write(f"**Category:** {result['category']}")
            if result.get("category_overridden"):
                st.info(f"🤖 Agent overrode the rule-based category. Reason: {result['override_reason']}")

            st.write("### Summary")
            st.info(result["summary"])

        with col2:
            st.write("### Keywords")
            st.write(", ".join(result["keywords"]))

            st.write("### Quality / Confidence Score")
            st.metric(label="Quality", value=f"{result['quality_score']}/5", delta=quality_label)

            st.write("### Reasoning")
            st.write(result["reasoning"])

        if trace:
            st.write("### 🧭 Agent Trace")

            def short(obj, limit=300):
                text_repr = str(obj)
                if len(text_repr) <= limit:
                    return text_repr, False
                return text_repr[:limit] + "…", True

            for step in trace:
                if step["type"] == "tool_call":
                    args_str, _ = short(step["args"], 200)
                    result_str, result_truncated = short(step["result"], 300)
                    st.markdown(
                        f"<div class='agent-trace-step'><b>Step {step['step']}:</b> "
                        f"called <code>{step['tool']}</code>({args_str})<br>"
                        f"→ {result_str}</div>",
                        unsafe_allow_html=True,
                    )
                    if result_truncated:
                        with st.expander(f"Full result for step {step['step']} ({step['tool']})"):
                            st.json(step["result"] if isinstance(step["result"], (dict, list)) else str(step["result"]))
                elif step["type"] == "finalize":
                    st.markdown(
                        f"<div class='agent-trace-step'><b>Step {step['step']}:</b> "
                        f"called <code>finalize_analysis</code></div>",
                        unsafe_allow_html=True,
                    )
                elif step["type"] == "fallback":
                    st.markdown(
                        f"<div class='agent-trace-step'><b>Step {step['step']}:</b> {step['content']}</div>",
                        unsafe_allow_html=True,
                    )

        report_text = generate_analysis_report(current_doc["file_name"], clean_text, result, trace, mode_label)
        st.download_button(
            label="📥 Download Analysis Report",
            data=report_text,
            file_name=f"{current_doc['file_name']}_analysis_report.txt",
            mime="text/plain",
            key=f"download_report_{index}_{current_doc['file_name']}",
        )

        st.write("### Extracted Text Preview")
        st.text_area(
            "First 1500 characters",
            clean_text[:1500],
            height=200,
            key=f"preview_{index}_{current_doc['file_name']}",
        )

    readable_documents = [d for d in documents if d["clean_text"]]
    if len(readable_documents) > 1:
        st.divider()
        st.header("📊 Cross-Document Similarity Analysis")
        st.caption(
            "This full pairwise comparison runs deterministically across all documents. "
            "The agent can also call compare_with_document for targeted, on-demand comparisons "
            "during its own analysis (see Agent Trace above)."
        )

        similarity_df = calculate_similarity_matrix(readable_documents)

        if not similarity_df.empty:
            st.dataframe(similarity_df)
            similarity_report = generate_similarity_report(similarity_df)
            st.download_button(
                label="📥 Download Similarity Report",
                data=similarity_report,
                file_name="similarity_analysis_report.txt",
                mime="text/plain",
                key="download_similarity_report",
            )
        else:
            st.warning("Similarity analysis could not generate a result.")

    # -----------------------------
    # CHAT WITH YOUR DOCUMENTS
    # -----------------------------

    st.divider()
    st.header("💬 Chat with your documents")

    if not agent_mode_active:
        st.info("Enter a Gemini API key in the sidebar to chat with your uploaded documents.")
    elif not readable_documents:
        st.info("Upload at least one readable document to start chatting.")
    else:
        st.caption(
            "The chat agent can list your documents, search across all of them, and pull up "
            "specific passages — it decides which of those to use per question."
        )

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_question = st.chat_input("Ask a question about your uploaded documents...")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Searching and reasoning across your documents..."):
                    try:
                        answer, chat_trace = run_chat_agent(
                            client,
                            llm_model,
                            user_question,
                            st.session_state.chat_contents,
                            st.session_state.search_index,
                            documents,
                            max_agent_turns,
                        )
                    except Exception as error:
                        answer = f"Sorry, the chat agent hit an error: {error}"
                        chat_trace = []

                st.write(answer)

                if chat_trace:
                    with st.expander("🧭 Chat agent trace"):
                        for step in chat_trace:
                            st.write(f"**Step {step['step']}:** called `{step['tool']}`({step['args']})")
                            st.json(step["result"] if isinstance(step["result"], (dict, list)) else str(step["result"]))

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.info("Please upload at least one document.")