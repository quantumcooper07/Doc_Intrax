import streamlit as st
import tempfile
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import fitz  # PyMuPDF
import os

# --- Configure Google Gemini ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# --- Gemini Wrapper ---
def ask_gemini(prompt):
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# --- PDF Loader ---
def load_pdf(file_path):
    doc = fitz.open(file_path)
    page_texts = [page.get_text() for page in doc]
    full_text = " ".join(page_texts)
    return full_text, page_texts

# --- Text Chunking ---
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# --- Vector DB Creation with Chroma ---
def create_vector_db(chunks):
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_directory)

# --- Ask Question with Context ---
def ask_question_with_context(vector_db, question):
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
Use the following context to answer the question:

Context:
{context}

Question:
{question}
"""
    return ask_gemini(prompt)

# --- Streamlit App UI ---
st.set_page_config(page_title="PDF Q&A with Gemini", layout="wide")
st.title("Ask Questions About Your PDF using Google Gemini")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Processing PDF...")

    try:
        full_text, page_texts = load_pdf(tmp_path)
        chunks = chunk_text(full_text)

        st.success(f"PDF processed: {len(page_texts)} pages, {len(chunks)} chunks")

        st.session_state["page_texts"] = page_texts
        st.session_state["vector_db"] = create_vector_db(chunks)

    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")

# --- Question Input ---
if "vector_db" in st.session_state:
    st.header("Ask a Question")
    query = st.text_input("Your question:")

    if query:
        with st.spinner("Thinking..."):
            try:
                answer = ask_question_with_context(st.session_state["vector_db"], query)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error during response: {str(e)}")
