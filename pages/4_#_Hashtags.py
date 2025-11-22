import streamlit as st
import pandas as pd
import utils.mostrar_datos_ingresados
from utils.hashtag_recommender import recomendar_hashtags_existentes


with st.sidebar:
    st.image("assets/logo.png", width=150)
st.title("🔖 Recomendador de Hashtags")

# =====================================================
# MOSTRAR GUION + IMAGEN (VALIDADO)
# =====================================================
utils.mostrar_datos_ingresados.mostrar_datos_ingresados(st.session_state)


# =====================================================
# CARGAR DATAFRAME DE HASHTAGS UNA SOLA VEZ
# =====================================================
if "df_hash" not in st.session_state:
    st.session_state.df_hash = pd.read_parquet("DATA/dataframes/hashtags.parquet")

# Inicializar cache de resultados
if "hashtags_existentes" not in st.session_state:
    st.session_state.hashtags_existentes = None


# =====================================================
# BOTÓN PARA RECOMENDAR
# =====================================================
if st.button("🔍 Recomendar hashtags"):

    # Validar embedding del guion
    if "guion_embedding" not in st.session_state or st.session_state.guion_embedding is None:
        st.error("❌ No hay embedding del guion generado.")
        st.stop()

    st.info("⏳ Analizando...")

    # ---------- HASHTAGS EXISTENTES ----------
    st.session_state.hashtags_existentes = utils.hashtag_recommender.recomendar_hashtags_existentes(
    embedding_texto=st.session_state.guion_embedding,
    df_hash=st.session_state.df_hash,
    texto_original=st.session_state.guion_text,
    top_k=10
    )

    st.success("✨ Recomendaciones generadas!")


# =====================================================
# MOSTRAR RESULTADOS
# =====================================================
if st.session_state.hashtags_existentes is not None:
    st.markdown("## 🧩 Hashtags existentes recomendados")

    df = st.session_state.hashtags_existentes[
        ["hashtag"]
    ]

    st.dataframe(df, width='stretch')
