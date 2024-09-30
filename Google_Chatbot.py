import streamlit as st
import PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain.vectorstores import FAISS
import os

st.set_page_config(page_title="EIS Virtual Assistant", layout="wide")

st.markdown ("<h1 style='text-align: center;'>EIS Virtual Assistant</h1>", unsafe_allow_html=True)
st.markdown("""
            ##Get information about admissions at EIS, Fees and IB Curreiculum Guide by chatting with EIS's virtual assistant
            ##You can ask your questions here
            """)

api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """ As a school admission counselor, your task is to provide accurate and informative answers to the questions posed by students. You should use the provided context to support your answers. If the context does not provide the answer, say "I don't know the answer to that question." and \n\n"""
    context :\n {context}?\n
    Question: \n{question} \n

    Answer:
    
    model = ChatGoogleGenerativeAI (model="gemini-1.0-pro", google_api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
    
def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


#streamlit UI Code
def main():
    st.header("EIS Virtual Assistant")

    user_question = st.text_input("Ask a question about EIS admissions, fees, or IB curriculum", key="user_question")

    if user_question and api_key:
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("EIS Virtual Assistant")
        pdf_docs = st.file_uploader("Upload your PDFs here...", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")

if _name_ == "__main__":
    main()