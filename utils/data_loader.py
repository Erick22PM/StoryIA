import streamlit as st
import pandas as pd
import spacy
from tensorflow.keras.models import load_model
import joblib
import os
import requests

from sentence_transformers import SentenceTransformer

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
    return pd.read_parquet("./DATA/dataframes/PROD_DATASET.parquet")


@st.cache_data(show_spinner="Cargando embeddings...")
def load_embeddings():
    return pd.read_parquet("./DATA/dataframes/clustering_y_embedding_guiones.parquet")


# =============================
#   MODELOS DE ML
# =============================
@st.cache_resource
def load_clas_model():
    return load_model("./models/modelo_clasificacion.keras")

def load_r_bajo_model():
    return load_model("./models/modelo_regr_bajo.keras")

def load_r_normal_model():
    return load_model("./models/modelo_regr_normal.keras")

def load_r_viral_model():
    return load_model("./models/modelo_regr_viral.keras")

@st.cache_resource(show_spinner="Cargando modelo HDBSCAN...")
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
    return load_model("./models/scaler-bajo_y.pkl")

@st.cache_resource
def load_scaler_viral_y():
    return load_model("./models/scaler-viral_y.pkl")

@st.cache_resource
def load_scaler_normal_y():
    return load_model("./models/scaler-normal_y.pkl")
