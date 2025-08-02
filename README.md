# phi3-chatbot

A Gradio-based PDF Q&A chatbot using LangChain + FAISS + Hugging Face Hub.

## Features
- Upload a PDF
- Ask questions
- Uses Mistral 7B via Hugging Face

## Setup
Make sure `HUGGINGFACE_TOKEN` is set as a secret in Hugging Face Space.

## Dev
```bash
pip install -r requirements.txt
python app.py
