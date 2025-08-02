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
LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Auth
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HUGGINGFACE_TOKEN not set in environment")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

def process_pdf(file, question):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            loader = PyPDFLoader(tmp_file.name)
            docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.from_documents(chunks, embeddings)

        llm = HuggingFaceHub(
            repo_id=LLM_REPO_ID,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
        )
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
        return qa.run(question)

    except Exception as e:
        return f"❌ Error: {str(e)}"

gr.Interface(
    fn=process_pdf,
    inputs=[
        gr.File(label="Upload a PDF"),
        gr.Textbox(label="Ask a question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Phi-3 Chatbot",
    allow_flagging="never"
).launch()
