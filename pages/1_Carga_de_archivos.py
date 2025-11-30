import streamlit as st
from utils.text_embeddings import embed_text_robusto
from utils.procesar_guion import procesar_guion_completo

# ------------------------------------------
# CONFIGURACIÓN INICIAL
# ------------------------------------------
st.set_page_config(page_title="Cargar archivos", layout="wide")

with st.sidebar:
    st.image("assets/logo.png", width=150)

st.title("📝 Cargar de archivos")


# ------------------------------------------
# INICIALIZAR SESSION STATE
# ------------------------------------------
DEFAULTS = {
    "guion_text": None,
    "guion_image": None,
    "guion_embedding": None,
    "guion_resultados": None,
    "puntaje_modelo": None,
    "procesado": False,
    "edit_mode": True,
}

for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


# ------------------------------------------
# FUNCIONES CACHEADAS
# ------------------------------------------
@st.cache_resource
def cached_embedding(text):
    return embed_text_robusto(text)


@st.cache_resource
def cached_procesamiento(text):
    return procesar_guion_completo(text)


# ====================================================
# 🔧 MODO: EDITAR
# ====================================================
if st.session_state.edit_mode:

    st.subheader("Ingrese su guión")
    guion = st.text_area(
        "Escribe o pega tu guión aquí:",
        height=250,
        value=st.session_state.guion_text or ""
    )

    st.subheader("Subir imagen")
    imagen = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

    if st.button("Guardar datos"):
        if not guion.strip():
            st.warning("⚠️ Debes ingresar un guion antes de continuar.")
            st.stop()

        guion_cambiado = (guion != st.session_state.guion_text)

        # Guardar en session_state
        st.session_state.guion_text = guion
        if imagen:
            st.session_state.guion_image = imagen

        # Solo procesar si el guion cambió
        if guion_cambiado:
            st.info("🚀 Procesando tu guion...")

            progress = st.progress(0)
            status = st.empty()

            # 1) Embed
            status.write("🔄 Generando embedding...")
            progress.progress(10)
            embedding = cached_embedding(guion)

            st.session_state.guion_embedding = embedding
            progress.progress(40)

            # 2) Procesar con modelos
            status.write("🏗 Ejecutando modelos analíticos...")
            resultados = cached_procesamiento(guion)

            st.session_state.guion_resultados = resultados
            st.session_state.puntaje_modelo = float(resultados)
            st.session_state.procesado = True

            progress.progress(100)
            status.success("✅ Procesamiento completo")

        else:
            st.info("👍 El guion no cambió. No se reprocesó.")

        st.session_state.edit_mode = False
        st.rerun()


# ====================================================
# 📄 MODO: VISTA PREVIA
# ====================================================
else:
    st.markdown("## Contenido cargado")

    if st.session_state.guion_image:
        col_img, col_text = st.columns([1, 2])

        with col_img:
            st.subheader("Imagen cargada")
            st.image(st.session_state.guion_image, use_container_width=True)

        with col_text:
            st.subheader("Guion cargado")
            st.write(st.session_state.guion_text)
    else:
        st.subheader("Guion cargado")
        st.write(st.session_state.guion_text)

    st.markdown("---")

    if st.session_state.guion_embedding is not None:
        st.success("🧠 Embedding generado.")
        st.write(f"Dimensión embedding: {len(st.session_state.guion_embedding)}")
    else:
        st.error("❌ No se generó embedding.")

    st.markdown("---")

    # Botón volver a editar
    if st.button("✏️ Editar o cargar nuevos archivos"):
        st.session_state.edit_mode = True
        st.rerun()

    if st.button("➡ Ir a análisis del guion"):
        st.switch_page("pages/2_Guion.py")
