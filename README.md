# Agentic Document Understanding System

A Streamlit app that analyzes PDF, TXT, and DOCX documents. With a Gemini API
key, an LLM agent decides which analysis tools to run on each document
(summarize, extract keywords, classify, chunk, compare) instead of following
a fixed pipeline. Without a key, it falls back to a deterministic rule-based
pipeline, so the app still works either way.

## Features

- **Agent mode** — a Gemini-powered agent picks its own analysis plan per
  document and shows its reasoning as a step-by-step trace.
- **Deterministic fallback** — works without any API key using TF-IDF
  keyword extraction, rule-based classification, and extractive summarization.
- **Multi-document support** — upload several files at once and compare them.
- **Cross-document similarity** — pairwise TF-IDF similarity between all
  uploaded documents.
- **Chat with your documents** — ask questions across everything you've
  uploaded; the chat agent searches and cites which document it pulled from.
- **Downloadable reports** — per-document analysis report and a similarity
  report, both as plain text.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Using agent mode (optional but recommended)

1. Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Paste it into the "Gemini API Key" field in the sidebar
3. Upload a document — the agent will decide its own analysis plan

Without a key, the app still runs in deterministic fallback mode.

## Supported file types

- PDF (`.pdf`)
- Plain text (`.txt`)
- Word documents (`.docx`)

## Notes

- Gemini's free tier is rate-limited (requests per minute/day). If you hit
  errors while testing multiple documents or chatting a lot, that's the
  free-tier limit, not a bug.
- Analysis results are cached per session — re-uploading the same files
  won't re-run the agent unless the files or settings change.