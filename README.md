# Campus Regulations Assistant (RAG)

A Retrieval-Augmented Generation (RAG) chatbot built with **Gemma 4**, **SentenceTransformers**, **PyMuPDF**, and **Gradio** that answers student queries strictly from official college handbook PDFs with page-level citations.

## How It Works
1. Extracts and cleans text from uploaded PDF handbooks using PyMuPDF.
2. Splits text into overlapping word chunks and embeds them with `all-MiniLM-L6-v2`.
3. Retrieves top matching passages via cosine similarity and grounds Gemma 4's response strictly in those sources.

## Run Locally / Colab
```bash
pip install -r requirements.txt
export API_KEY="your_api_key_here"
python app.py
```
