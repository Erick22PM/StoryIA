import streamlit as st
import utils.audio_generator
import utils.mostrar_datos_ingresados

st.title("📄 Vista del Guión Cargado")

# Validar si hay datos almacenados
if "guion_text" not in st.session_state or st.session_state.guion_text is None:
    st.warning("⚠️ No hay guión cargado. Ve primero a la página 'Carga de archivos'.")
    if st.button("Ir a Cargar Guión"):
        st.switch_page("pages/1_Carga_de_archivos.py")
    st.stop()


# =====================================================
# MOSTRAR GUION + IMAGEN
# =====================================================

utils.mostrar_datos_ingresados.mostrar_datos_ingresados(st.session_state)


# =====================================================
# 🔊 NUEVA SECCIÓN: Generar AUDIO
# =====================================================


st.markdown("---")

st.subheader("Score narrativo (placeholder)")
st.progress(0.5)

st.subheader("Feedback inteligente (placeholder)")
st.write("💬 Aquí irá el feedback generado por IA.")

st.markdown("---")
