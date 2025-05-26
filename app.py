import streamlit as st
import tempfile
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from langchain.vectorstores import Chroma
import os


# Initialize Gemini
GEMINI_API_KEY = "AIzaSyBVcNicvBrN18eUxxJJ-wH8PfiXIgoA-PE"
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


# Gemini wrapper function
def ask_gemini(prompt):
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# Load PDF and return full text + page-wise text
def load_pdf(file_path):
    doc = fitz.open(file_path)
    page_texts = [page.get_text() for page in doc]
    full_text = " ".join(page_texts)
    return full_text, page_texts

# Split into chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

#Create Vector DB
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "./chroma_db"
    return Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_directory)


# Ask question using Gemini with retrieved context
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

# ---- Streamlit App ----
st.set_page_config(page_title="Document Q&A with Google Gemini", layout="wide")
st.title("Ask Questions About Your PDF with Google Gemini")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Extracting and processing PDF...")

    try:
        full_text, page_texts = load_pdf(tmp_path)
        chunks = chunk_text(full_text)

        st.success(f" PDF processed: {len(page_texts)} pages, {len(chunks)} chunks")

        # Store in session state
        st.session_state["page_texts"] = page_texts
        st.session_state["vector_db"] = create_vector_db(chunks)

        # # Page-wise viewer
        # st.header(" Page Viewer")
        # selected_page = st.slider("Select a page to view", 1, len(page_texts), 1)
        # st.text_area(f"Page {selected_page} content:", page_texts[selected_page - 1], height=300)

    except Exception as e:
        st.error(f" Failed to process PDF: {str(e)}")

# Ask questions
if "vector_db" in st.session_state:
    st.header("Ask Questions About the Document")
    query = st.text_input("Type your question here:")

    if query:
        with st.spinner("Thinking..."):
            try:
                answer = ask_question_with_context(st.session_state["vector_db"], query)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error during response: {str(e)}")
