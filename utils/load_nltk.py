import streamlit as st
import nltk
import tiktoken
from nltk.corpus import stopwords

@st.cache_resource(show_spinner="Descargando recursos de NLTK...")
def load_nltk():
    """
    Descarga y carga los recursos necesarios de NLTK una sola vez por sesión.
    """

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    return True

@st.cache_resource()
def load_nltk_stop_words():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    stopwords_es = set(stopwords.words("spanish"))

    return stopwords_es


@st.cache_resource
def load_tokenizer():
    return tiktoken.get_encoding("cl100k_base")
