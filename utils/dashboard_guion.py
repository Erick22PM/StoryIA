import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import streamlit as st
import pandas as pd
import seaborn as sns
import re

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def get_dashboard_guion(transcripcion):
    st.markdown("---")
    st.markdown("## 📊 Análisis del guión")

    text = transcripcion

    # ------------------------------------------------------
    # Procesar texto
    # ------------------------------------------------------
    # Tokenizar y limpiar
    stopwords_es = set(stopwords.words("spanish"))
    tokens = simple_tokenize(text)
    tokens_limpios = [t for t in tokens if t not in stopwords_es]

    # Contar palabras
    conteo = Counter(tokens_limpios)

    # Crear DataFrame para visualizar
    df_freq = pd.DataFrame(conteo.most_common(20), columns=["palabra", "frecuencia"])

    # ------------------------------------------------------
    # Métricas generales
    # ------------------------------------------------------
    num_palabras = len(tokens)
    num_palabras_unicas = len(set(tokens))
    num_oraciones = text.count(".")
    densidad_lexica = round(num_palabras_unicas / num_palabras, 3) if num_palabras > 0 else 0

    st.markdown("### 📐 Métricas generales del guion")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Palabras totales", num_palabras)
    col2.metric("Palabras únicas", num_palabras_unicas)
    col3.metric("Oraciones", num_oraciones)
    col4.metric("Densidad léxica", densidad_lexica)

    st.markdown("---")

    # ------------------------------------------------------
    # Tabla de palabras más comunes
    # ------------------------------------------------------
    st.subheader("🔠 Palabras más frecuentes")
    st.dataframe(df_freq)

    # ------------------------------------------------------
    # Gráfica de barras
    # ------------------------------------------------------
    st.subheader("📊 Frecuencia de palabras")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_freq, x="frecuencia", y="palabra")
    st.pyplot(fig)

    st.markdown("---")

    # ------------------------------------------------------
    # Nube de palabras
    # ------------------------------------------------------
    st.subheader("☁️ Nube de palabras")

    wc = WordCloud(width=800, height=300, background_color="white").generate(" ".join(tokens_limpios))

    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")

    st.pyplot(fig_wc)

    st.markdown("---")

    # ------------------------------------------------------
    # Análisis de entidades con SpaCy
    # ------------------------------------------------------
    st.subheader("🏷 Entidades nombradas (NER)")

    nlp = spacy.load("es_core_news_sm")
    doc = nlp(text)

    entities = [(ent.text, ent.label_) for ent in doc.ents]

    if entities:
        df_ents = pd.DataFrame(entities, columns=["Entidad", "Tipo"])
        st.dataframe(df_ents)
    else:
        st.info("No se detectaron entidades en este guión.")

    st.markdown("---")

    # ------------------------------------------------------
    # Bigramas
    # ------------------------------------------------------
    st.subheader("🔗 Bigramas más comunes")

    bigrams = list(zip(tokens_limpios, tokens_limpios[1:]))
    conteo_bigrams = Counter(bigrams)
    df_bigrams = pd.DataFrame(conteo_bigrams.most_common(15), columns=["bigrama", "frecuencia"])

    st.dataframe(df_bigrams)

    # Gráfica
    fig_bg, ax_bg = plt.subplots(figsize=(8, 4))
    pairs = [" ".join(p) for p, _ in conteo_bigrams.most_common(15)]
    freqs = [f for _, f in conteo_bigrams.most_common(15)]

    sns.barplot(x=freqs, y=pairs, ax=ax_bg)
    st.pyplot(fig_bg)
