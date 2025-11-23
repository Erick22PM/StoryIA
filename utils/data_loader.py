import streamlit as st
import pandas as pd
import spacy
from tensorflow.keras.models import load_model
import joblib
import os
import requests

from sentence_transformers import SentenceTransformer

def load_file(file_name):
    base = os.path.dirname(__file__)         # ruta del archivo data_loader.py
    path = os.path.join(base, "..", "utils/models", file_name)
    path = os.path.abspath(path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encontró el modelo en: {path}")
    return load_model(path)

def load_cler(file_name):
    base = os.path.dirname(__file__)         # ruta del archivo data_loader.py
    path = os.path.join(base, "..", "utils/models", file_name)
    path = os.path.abspath(path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encontró el modelo en: {path}")
    return  joblib.load(path)


def load_parquet(file_name):
    base = os.path.dirname(__file__)         # ruta del archivo data_loader.py
    path = os.path.join(base, "..", "DATA/dataframes", file_name)
    path = os.path.abspath(path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encontró el modelo en: {path}")
    return  pd.read_parquet(path)

# =============================
#   embeddins
# =============================

@st.cache_resource(show_spinner="Cargando modelo de embeddings…")
def load_embedder():
    """
    Carga el modelo SentenceTransformer en memoria compartida (RAM de la sesión)
    """
    try:
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        raise RuntimeError(f"Error cargando SentenceTransformer: {e}")


# =============================
#   SPA CY (MUY PESADO)
# =============================

@st.cache_resource(show_spinner="Cargando modelo SpaCy…")
def load_spacy():
    """
    Carga el modelo SpaCy español en cache (una sola vez por sesión)
    """
    try:
        return spacy.load("es_core_news_sm", disable=["ner"])
    except Exception as e:
        raise RuntimeError(f"Error cargando SpaCy: {e}")


# ==========================================================
#   DATASETS
# ==========================================================
@st.cache_data(show_spinner="Cargando dataset principal...")
def load_main_dataset():
    return load_parquet("PROD_DATASET.parquet")


@st.cache_data(show_spinner="Cargando hashtags...")
def load_hashtags():
    return load_parquet("hashtags.parquet")

@st.cache_data(show_spinner="Cargando EDA...")
def load_eda():
    return load_parquet("2_EDA_AGENT.parquet")


@st.cache_data(show_spinner="Cargando embeddings...")
def load_embeddings():
    return load_parquet("clustering_y_embedding_guiones.parquet")


# =============================
#   MODELOS DE ML
# =============================
@st.cache_resource
def load_clas_model():
    return load_file("modelo_clasificacion.keras")

@st.cache_resource
def load_r_bajo_model():
    return load_file("modelo_regr_bajo.keras")

@st.cache_resource
def load_r_normal_model():
    return load_file("modelo_regr_normal.keras")

@st.cache_resource
def load_r_viral_model():
    return load_file("modelo_regr_viral.keras")

@st.cache_resource
def load_hdbscan_mod():
    url = "https://huggingface.co/ErickPM22/clustering/resolve/main/hdbscan_model.pkl"
    local_path = "/mount/src/modelo_hdbscan.pkl"   # lugar seguro en Streamlit Cloud

    # Descargar solo si no existe localmente
    if not os.path.exists(local_path):
        print("📥 Descargando modelo HDBSCAN desde HuggingFace...")
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

    paquete = joblib.load(local_path)
    clusterer = paquete["clusterer"]
    X_norm = paquete.get("X_norm")
    return clusterer, X_norm

# =========================
# CARGAR SCALERS
# =========================

@st.cache_resource
def load_scaler_bajo_y():
    return load_cler("scaler-bajo_y.pkl")

@st.cache_resource
def load_scaler_viral_y():
    return load_cler("scaler-viral_y.pkl")

@st.cache_resource
def load_scaler_normal_y():
    return load_cler("scaler-normal_y.pkl")
