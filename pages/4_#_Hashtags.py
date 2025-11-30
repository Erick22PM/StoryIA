import streamlit as st
import pandas as pd
import utils.mostrar_datos_ingresados
from utils.hashtag_openai import generar_hashtags_desde_guion   # <--- NUEVO

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

# Inicializar cache de resultados
if "hashtags_existentes" not in st.session_state:
    st.session_state.hashtags_existentes = None

# Inicializar resultados IA
if "hashtags_ia" not in st.session_state:
    st.session_state.hashtags_ia = None

# =====================================================
# BOTÓN PARA RECOMENDAR CON IA (OPENAI)
# =====================================================

# Mostrar el botón SOLO si no se han generado hashtags aún
if st.session_state.hashtags_ia is None:

    if st.button("🧠 Generar hashtags"):

        # Validar guion
        if "guion_text" not in st.session_state or not st.session_state.guion_text:
            st.error("❌ No hay guion cargado.")
        else:
            st.info("⏳ Generando hashtags con IA...")
            try:
                st.session_state.hashtags_ia = generar_hashtags_desde_guion(
                    st.session_state.guion_text
                )
                st.success("🚀 Hashtags generados")
            except Exception as e:
                st.error(f"⚠️ Error al generar hashtags con IA: {e}")


# =====================================================
# MOSTRAR RESULTADOS IA
# =====================================================
if st.session_state.hashtags_ia is not None:
    st.markdown("## 🤖 Hashtags generados con IA")

    relacionados = st.session_state.hashtags_ia.get("relacionados", [])
    viralidad = st.session_state.hashtags_ia.get("viralidad", [])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Relacionados con el guion")
        if relacionados:
            for h in relacionados:
                if not h.startswith("#"):
                    h = "#" + h.strip()
                st.write(f"- {h}")
        else:
            st.write("No se generaron hashtags relacionados.")

    with col2:
        st.markdown("### 📢 Viralidad y redes sociales")
        if viralidad:
            for h in viralidad:
                if not h.startswith("#"):
                    h = "#" + h.strip()
                st.write(f"- {h}")
        else:
            st.write("No se generaron hashtags de viralidad.")

    # (OPCIONAL) Botón de regenerar
    st.markdown("---")
    if st.button("🔄 Regenerar hashtags"):
        st.session_state.hashtags_ia = None
        st.rerun()
