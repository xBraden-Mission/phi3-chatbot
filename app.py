import os
import tempfile
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Config
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "google/flan-t5-base"

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("❌ HUGGINGFACE_TOKEN not set in environment.")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

def process_pdf(file):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, file.name)
        with open(pdf_path, "wb") as f:
            f.write(file.read())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectordb = FAISS.from_documents(splits, embeddings)

        llm = HuggingFaceHub(
            repo_id=LLM_REPO_ID,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 256},
        )

        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

        return qa

state = gr.State()

def upload_and_prepare(pdf):
    try:
        qa_chain = process_pdf(pdf)
        state.value = qa_chain
        return "✅ PDF processed. Ask your question!"
    except Exception as e:
        return f"❌ Error: {e}"

def ask_question(question):
    if state.value is None:
        return "❌ Please upload a PDF first."
    try:
        return state.value.run(question)
    except Exception as e:
        return f"❌ Failed to answer: {e}"

with gr.Blocks(title="Mission Engineering Chatbot") as demo:
    gr.Markdown("# 📘 Mission Engineering Chatbot")
    with gr.Row():
        pdf_file = gr.File(file_types=[".pdf"], label="Upload PDF")
        upload_btn = gr.Button("Process PDF")
    status = gr.Textbox(label="Status")

    question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="Answer")
    ask_btn = gr.Button("Submit")

    upload_btn.click(upload_and_prepare, inputs=pdf_file, outputs=status)
    ask_btn.click(ask_question, inputs=question, outputs=answer)

demo.launch(share=True)
