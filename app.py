# =============================================================
# 1. Install required packages (Colab)
# =============================================================
import importlib.util
import subprocess
import sys


def ensure(pip_name, module_name):
    """Install a package only if it is missing."""
    if importlib.util.find_spec(module_name) is None:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


ensure("PyMuPDF", "fitz")
ensure("google-genai", "google.genai")
ensure("sentence-transformers", "sentence_transformers")
ensure("gradio", "gradio")

import glob
import os
import re
import fitz  # PyMuPDF
import gradio as gr
import numpy as np
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# =============================================================
# 2. Configuration & API Key
# =============================================================
API_KEY = ""  # Paste your API key here, OR save it as 'API_KEY' in Colab Secrets

MODEL_NAMES = ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 100      # words per chunk
CHUNK_OVERLAP = 20    # words shared between neighbouring chunks
TOP_K = 3
TEMPERATURE = 0.2

ANSWER_RULES = (
    "You are a campus regulations assistant. Answer ONLY using the context "
    "provided. Cite the file name and page number for each fact. If the "
    "answer is not in the context, say you could not find it in the handbooks."
)


def get_api_key():
    """Check API_KEY variable first, then environment variables, then Colab Secrets."""
    if API_KEY and API_KEY.strip():
        return API_KEY.strip()
    for name in ("API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        key = os.environ.get(name)
        if key:
            return key
    try:
        from google.colab import userdata
        for name in ("API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            try:
                key = userdata.get(name)
                if key:
                    return key
            except Exception:
                continue
    except Exception:
        pass
    return None


# =============================================================
# 3. Load and clean all PDF files from the Colab folder
# =============================================================
pdf_files = glob.glob("*.pdf")
print(f"📁 Found these PDF files: {pdf_files}")

if not pdf_files:
    raise FileNotFoundError("No PDF files found. Upload your handbooks to the Colab file panel first.")


def clean_text(text):
    text = text.replace("\u00ad", "")               # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)    # re-join words split across lines
    text = re.sub(r"\s*\n\s*", " ", text)           # line breaks -> spaces
    text = re.sub(r"[ \t]+", " ", text)             # repeated spaces
    return text.strip()


PAGES = []
for filename in pdf_files:
    doc = fitz.open(filename)
    empty_pages = []
    for page_index in range(len(doc)):
        text = clean_text(doc[page_index].get_text("text", sort=True))
        if len(text) > 20:
            PAGES.append({"file": filename, "page": page_index + 1, "text": text})
        else:
            empty_pages.append(page_index + 1)
    print(f"📖 '{filename}': {len(doc)} pages, {len(doc) - len(empty_pages)} with readable text")
    if empty_pages:
        print(f"   ⚠️ No readable text on pages {empty_pages} (likely scanned images).")
    doc.close()

if not PAGES:
    raise ValueError("No readable text found in any PDF. The files are probably scanned images.")

print(f"🎉 Success! Total pages loaded: {len(PAGES)}")

# =============================================================
# 4. Embedding model, LLM helper, and RAG knowledge base
# =============================================================
embedder = SentenceTransformer(EMBED_MODEL_NAME)
knowledge_base = {"name": None, "chunks": [], "vectors": None}


def split_into_chunks(pages):
    chunks = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for p in pages:
        words = p["text"].split()
        for start in range(0, len(words), step):
            chunk_words = words[start:start + CHUNK_SIZE]
            if len(chunk_words) > 5:
                chunks.append({
                    "file": p["file"],
                    "page": p["page"],
                    "text": " ".join(chunk_words),
                })
            if start + CHUNK_SIZE >= len(words):
                break
    return chunks


def build_knowledge_base(pages, name):
    chunks = split_into_chunks(pages)
    if not chunks:
        raise ValueError("No text was found. The PDF may contain scanned images only.")
    vectors = embedder.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )
    knowledge_base.update(name=name, chunks=chunks, vectors=vectors)
    return f"Loaded **{name}**: {len(pages)} pages, {len(chunks)} chunks."


def retrieve(question, top_k=TOP_K):
    question_vector = embedder.encode([question], normalize_embeddings=True)[0]
    similarities = knowledge_base["vectors"] @ question_vector
    best = np.argsort(similarities)[::-1][:top_k]
    return [{**knowledge_base["chunks"][i], "score": float(similarities[i])} for i in best]


def generate(prompt, system_text, demo_reply):
    """Call Gemma through the Gemini API; fall back to demo_reply if no API_KEY is set."""
    api_key = get_api_key()
    if not api_key:
        return demo_reply

    client = genai.Client(api_key=api_key)
    last_error = None
    for model_name in MODEL_NAMES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_text}\n\n{prompt}",
                config=types.GenerateContentConfig(temperature=TEMPERATURE),
            )
            return (response.text or "").strip()
        except Exception as e:
            last_error = e
    return f"⚠️ The model call failed: {last_error}"


def answer(question):
    if not question or not question.strip():
        return "Please type a question first.", []

    found = retrieve(question)
    context = "\n\n".join(f"[{c['file']} - page {c['page']}] {c['text']}" for c in found)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    top = found[0]
    demo_reply = (
        "**Demonstration mode:** no `API_KEY` is set, so Gemma did not write an answer. "
        f"Retrieval completed; the most relevant passage is in **{top['file']}, page {top['page']}**:\n\n"
        f"> {top['text'][:400]}..."
    )
    reply = generate(prompt, system_text=ANSWER_RULES, demo_reply=demo_reply)
    return reply, found


print(build_knowledge_base(PAGES, "Official College Handbooks"))

# =============================================================
# 5. Gradio UI
# =============================================================
gr.close_all()


def on_ask(question):
    if not question.strip():
        raise gr.Error("Enter a question first.")
    reply, found = answer(question)
    sources = "\n\n".join(
        f"**{c['file']} — Page {c['page']}** (similarity {c['score']:.2f})\n> {c['text']}"
        for c in found
    )
    return reply, sources


custom_css = """
body {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Inter', sans-serif;
}
.gradio-container {
    max-width: 900px !important;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    margin-top: 30px;
    margin-bottom: 30px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="blue"), css=custom_css) as demo:
    gr.Markdown("# 🎓 Campus Regulations Assistant")
    gr.Markdown("### *Your AI-powered guide to official college handbooks, rules, and policies.*")
    gr.Markdown(f"✅ *Connected to your uploaded college rulebooks ({len(PAGES)} pages loaded).*")

    question_box = gr.Textbox(
        label="Question",
        placeholder="For example: What is the minimum attendance required?",
    )
    ask_btn = gr.Button("Ask", variant="primary")

    gr.Examples(
        [
            "What is the minimum attendance required?",
            "Are re-evaluations allowed and what is the deadline?",
            "What are the hostel visiting hours?",
            "Who won the 2022 World Cup?",
        ],
        inputs=question_box,
    )

    answer_md = gr.Markdown()
    with gr.Accordion("Retrieved passages (the text given to Gemma)", open=False):
        sources_md = gr.Markdown()

    ask_btn.click(on_ask, inputs=question_box, outputs=[answer_md, sources_md])
    question_box.submit(on_ask, inputs=question_box, outputs=[answer_md, sources_md])

demo.launch(share=True)
  
