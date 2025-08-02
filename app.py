import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HUGGINGFACE_TOKEN is not set. Set it in the Space’s secrets.")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

def process_pdf(file):
    try:
        loader = PyPDFLoader(file.name)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(split_docs, embeddings)
        retriever = db.as_retriever()
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
        )
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return chain
    except Exception as e:
        return f"Error: {str(e)}"

def ask_question(file, question):
    chain = process_pdf(file)
    if isinstance(chain, str):  # error message
        return chain
    return chain.run(question)

gr.Interface(
    fn=ask_question,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a Question")],
    outputs="text",
    title="Mission Engineering Chatbot",
    allow_flagging="never"
).launch()
