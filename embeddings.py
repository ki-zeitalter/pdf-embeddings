import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def setup():
    load_dotenv()

def textExtrahieren(pdfDateien):
    gesamtText = ""

    for pdf in pdfDateien:
        reader = PdfReader(pdf)
        for page in reader.pages:
            gesamtText += page.extract_text()
    
    return gesamtText


def textSplitten(text):
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)

    return splitter.split_text(text)


def erstelleEmbeddings(textTeile):
    return FAISS.from_texts(texts=textTeile, embedding=OpenAIEmbeddings())

def suche(suchtext):
    ergebnisse = st.session_state.embeddings.similarity_search(suchtext)

    st.write(ergebnisse)


def run():
    setup()

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None

    st.set_page_config(page_title="Embeddings aus deinen PDF-Dateien 📘", page_icon="📘")

    st.header("Embeddings aus deinen PDF-Dateien 📘")

    pdfDateien = st.file_uploader("PDF-Dateien hochladen (anschließend klicke auf 'Erstelle Embeddings')",type="pdf", accept_multiple_files=True)

    if st.button("Erstelle Embeddings"):
        text = textExtrahieren(pdfDateien)

        textTeile = textSplitten(text)

        embeddings = erstelleEmbeddings(textTeile)

        st.session_state.embeddings = embeddings


    suchtext = st.text_input("Suche relevante Textstellen")

    if suchtext:
        suche(suchtext)

if __name__ == '__main__':
    run()

