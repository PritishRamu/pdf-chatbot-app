import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# Set up Streamlit UI
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Chat with Your PDF")

# Load OpenAI key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Upload PDF
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf is not None:
    # Save to temp file
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    # Load and split PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # Embedding + Vector Store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)

    # LLM Setup
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")

    # User Query
    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Searching..."):
            relevant_docs = db.similarity_search(query)
            answer = chain.run(input_documents=relevant_docs, question=query)
            st.success(answer)
